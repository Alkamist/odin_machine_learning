package example_chat

// Automatic download of model weights and tokenizers.
//
// Everything llm_chat needs is fetched over plain HTTPS with vendor:curl, which
// links statically against Schannel on Windows -- no DLLs, no cert bundle, no
// Python. Assets are pinned by exact byte size so a truncated or interrupted
// download is detected rather than surfacing later as a confusing parse error.
//
// Gemma weights come from Ollama's registry, which is content-addressed: the
// blob is named by its own sha256, so the URL can never silently start serving
// different bytes.

import "base:runtime"

import "core:c"
import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:strings"

import curl "vendor:curl"

Asset :: struct {
	url:  string,
	dest: string,
	size: i64, // exact expected size in bytes; also used to detect partial files
}

// Progress state threaded through curl's C callbacks.
Download_State :: struct {
	file:        ^os.File,
	ctx:         runtime.Context,
	name:        string,
	resume_from: i64,
	total:       i64,
	last_pct:    int,
	write_err:   os.Error,
}

// Formats a byte count with a unit that keeps the number readable, so a 2 MB
// tokenizer does not print as "0.00 GB" next to a 9.6 GB checkpoint.
@(require_results)
_human_size :: proc(bytes: i64, allocator := context.temp_allocator) -> string {
	KB :: 1024
	MB :: 1024 * KB
	GB :: 1024 * MB
	switch {
	case bytes >= GB: return fmt.aprintf("%.2f GB", f64(bytes) / GB, allocator = allocator)
	case bytes >= MB: return fmt.aprintf("%.1f MB", f64(bytes) / MB, allocator = allocator)
	case bytes >= KB: return fmt.aprintf("%.1f KB", f64(bytes) / KB, allocator = allocator)
	case:             return fmt.aprintf("%d B", bytes, allocator = allocator)
	}
}

@(require_results)
_asset_present :: proc(asset: Asset) -> bool {
	info, err := os.stat(asset.dest, context.temp_allocator)
	if err != nil {
		return false
	}
	return info.size == asset.size
}

// Ensures every asset exists locally, downloading the missing ones after
// confirming with the user. Returns false if the user declines or a download
// fails.
@(require_results)
ensure_assets :: proc(assets: []Asset, model_label: string) -> bool {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	missing: [dynamic]Asset
	defer delete(missing)

	total_bytes: i64
	for asset in assets {
		if !_asset_present(asset) {
			append(&missing, asset)
			total_bytes += asset.size
		}
	}
	if len(missing) == 0 {
		return true
	}

	fmt.printfln("%v is missing %d file(s), %v to download:", model_label, len(missing), _human_size(total_bytes))
	for asset in missing {
		fmt.printfln("  %v  (%v)", filepath.base(asset.dest), _human_size(asset.size))
	}
	fmt.print("Download now? [y/N] ")

	// read_line stops at the newline, so piped input keeps any following lines
	// available for the chat loop rather than swallowing them here.
	buf: [16]byte
	line, line_ok := read_line(buf[:])
	if !line_ok {
		fmt.eprintln("Aborted (no input on stdin).")
		return false
	}
	answer := strings.trim_space(line)
	if answer != "y" && answer != "Y" {
		fmt.println("Aborted. Download the files listed above manually to continue.")
		return false
	}

	curl.global_init(curl.GLOBAL_DEFAULT)
	defer curl.global_cleanup()

	for asset in missing {
		if !_download(asset) {
			return false
		}
	}
	return true
}

// Writes downloaded bytes straight to disk. Runs on curl's thread, so it
// restores the Odin context captured at call time.
_write_cb :: proc "c" (ptr: rawptr, size, nmemb: uint, userdata: rawptr) -> uint {
	state := cast(^Download_State)userdata
	context = state.ctx

	total := size * nmemb
	written, err := os.write(state.file, (cast([^]u8)ptr)[:total])
	if err != nil {
		// Returning a short count aborts the transfer with E_WRITE_ERROR.
		state.write_err = err
		return 0
	}
	return uint(written)
}

_progress_cb :: proc "c" (userdata: rawptr, dl_total, dl_now, ul_total, ul_now: i64) -> c.int {
	state := cast(^Download_State)userdata
	context = state.ctx

	done := state.resume_from + dl_now
	if state.total <= 0 {
		return 0
	}

	pct := int((done * 100) / state.total)
	if pct != state.last_pct {
		state.last_pct = pct
		fmt.printf("\r  %v  %3d%%  (%v / %v)", state.name, pct, _human_size(done), _human_size(state.total))
	}
	return 0
}

// Downloads a single asset to `dest`.
//
// Bytes land in a sibling ".part" file that is only renamed into place once the
// size matches exactly, so an interrupted run can never leave behind a file
// that looks complete. A pre-existing ".part" is resumed rather than restarted,
// which matters a great deal for the 9.6 GB Gemma weights.
@(require_results)
_download :: proc(asset: Asset) -> bool {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	name := filepath.base(asset.dest)
	part := strings.concatenate({asset.dest, ".part"}, context.temp_allocator)

	if dir := filepath.dir(asset.dest); dir != "" {
		if err := os.make_directory_all(dir); err != nil && !os.exists(dir) {
			fmt.eprintfln("FAIL: could not create %v: %v", dir, err)
			return false
		}
	}

	// Resume whatever a previous run already managed to write.
	resume_from: i64
	if info, err := os.stat(part, context.temp_allocator); err == nil && info.size < asset.size {
		resume_from = info.size
	}

	flags := os.File_Flags{.Write, .Create}
	if resume_from > 0 {
		fmt.printfln("  resuming %v at %v", name, _human_size(resume_from))
	} else {
		flags += {.Trunc}
	}

	file, open_err := os.open(part, flags)
	if open_err != nil {
		fmt.eprintfln("FAIL: could not open %v: %v", part, open_err)
		return false
	}
	// Closed explicitly before the rename below; this only covers early exits.
	file_open := true
	defer if file_open { os.close(file) }

	if resume_from > 0 {
		if _, err := os.seek(file, resume_from, .Start); err != nil {
			fmt.eprintfln("FAIL: could not seek %v: %v", part, err)
			return false
		}
	}

	state := Download_State {
		file        = file,
		ctx         = context,
		name        = name,
		resume_from = resume_from,
		total       = asset.size,
		last_pct    = -1,
	}

	handle := curl.easy_init()
	if handle == nil {
		fmt.eprintln("FAIL: curl_easy_init failed.")
		return false
	}
	defer curl.easy_cleanup(handle)

	url := strings.clone_to_cstring(asset.url, context.temp_allocator)
	curl.easy_setopt(handle, curl.option.URL, url)
	curl.easy_setopt(handle, curl.option.FOLLOWLOCATION, c.long(1))
	curl.easy_setopt(handle, curl.option.FAILONERROR, c.long(1))
	curl.easy_setopt(handle, curl.option.WRITEFUNCTION, _write_cb)
	curl.easy_setopt(handle, curl.option.WRITEDATA, &state)
	curl.easy_setopt(handle, curl.option.NOPROGRESS, c.long(0))
	curl.easy_setopt(handle, curl.option.XFERINFOFUNCTION, _progress_cb)
	curl.easy_setopt(handle, curl.option.XFERINFODATA, &state)
	if resume_from > 0 {
		curl.easy_setopt(handle, curl.option.RESUME_FROM_LARGE, resume_from)
	}

	res := curl.easy_perform(handle)
	fmt.println()

	if res != .E_OK {
		if state.write_err != nil {
			fmt.eprintfln("FAIL: could not write %v: %v", part, state.write_err)
		} else {
			fmt.eprintfln("FAIL: download of %v failed: %s", name, curl.easy_strerror(res))
			fmt.eprintfln("      partial data kept at %v; re-run to resume.", part)
		}
		return false
	}

	// Verify before publishing the file under its real name.
	written, size_err := os.file_size(file)
	if size_err != nil {
		fmt.eprintfln("FAIL: could not stat %v: %v", part, size_err)
		return false
	}
	if written != asset.size {
		fmt.eprintfln("FAIL: %v is %d bytes, expected %d.", name, written, asset.size)
		fmt.eprintfln("      partial data kept at %v; re-run to resume.", part)
		return false
	}

	os.close(file)
	file_open = false

	if err := os.rename(part, asset.dest); err != nil {
		fmt.eprintfln("FAIL: could not rename %v: %v", part, err)
		return false
	}
	return true
}

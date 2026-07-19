package fetch

import "base:builtin"
import "base:runtime"

import "core:c"
import "core:fmt"
import "core:log"
import "core:os"
import "core:path/filepath"
import "core:strings"

import curl "vendor:curl"

read_line :: proc(buffer: []byte) -> (line: string, ok: bool) {
	cursor := 0
	one: [1]byte
	for cursor < builtin.len(buffer) {
		n, err := os.read(os.stdin, one[:])
		if err != nil || n == 0 {
			if cursor == 0 {
				return "", false
			}
			break
		}
		c := one[0]
		if c == '\n' {
			break
		}
		buffer[cursor] = c
		cursor += 1
	}
	if cursor > 0 && buffer[cursor - 1] == '\r' {
		cursor -= 1
	}
	return string(buffer[:cursor]), true
}

Asset :: struct {
	url:  string,
	dest: string,
	size: i64, // exact expected size in bytes; also used to detect partial files
}

Download_State :: struct {
	file:        ^os.File,
	ctx:         runtime.Context,
	name:        string,
	resume_from: i64,
	total:       i64,
	last_pct:    int,
	write_err:   os.Error,
}

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

	buf: [16]byte
	line, line_ok := read_line(buf[:])
	if !line_ok {
		log.error("aborted: no input on stdin")
		return false
	}
	answer := strings.trim_space(line)
	if answer != "y" && answer != "Y" {
		log.warn("aborted: download the files listed above manually to continue")
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

_write_cb :: proc "c" (ptr: rawptr, size, nmemb: uint, userdata: rawptr) -> uint {
	state := cast(^Download_State)userdata
	context = state.ctx

	total := size * nmemb
	written, err := os.write(state.file, (cast([^]u8)ptr)[:total])
	if err != nil {
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

@(require_results)
_download :: proc(asset: Asset) -> bool {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	name := filepath.base(asset.dest)
	part := strings.concatenate({asset.dest, ".part"}, context.temp_allocator)

	if dir := filepath.dir(asset.dest); dir != "" {
		if err := os.make_directory_all(dir); err != nil && !os.exists(dir) {
			log.errorf("could not create %v: %v", dir, err)
			return false
		}
	}

	resume_from: i64
	if info, err := os.stat(part, context.temp_allocator); err == nil && info.size < asset.size {
		resume_from = info.size
	}

	flags := os.File_Flags{.Write, .Create}
	if resume_from > 0 {
		log.infof("resuming %v at %v", name, _human_size(resume_from))
	} else {
		flags += {.Trunc}
	}

	file, open_err := os.open(part, flags)
	if open_err != nil {
		log.errorf("could not open %v: %v", part, open_err)
		return false
	}

	file_open := true
	defer if file_open { os.close(file) }

	if resume_from > 0 {
		if _, err := os.seek(file, resume_from, .Start); err != nil {
			log.errorf("could not seek %v: %v", part, err)
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
		log.error("curl_easy_init failed")
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
			log.errorf("could not write %v: %v", part, state.write_err)
		} else {
			log.errorf("download of %v failed: %s (partial data kept at %v; re-run to resume)", name, curl.easy_strerror(res), part)
		}
		return false
	}

	written, size_err := os.file_size(file)
	if size_err != nil {
		log.errorf("could not stat %v: %v", part, size_err)
		return false
	}
	if written != asset.size {
		log.errorf("%v is %d bytes, expected %d (partial data kept at %v; re-run to resume)", name, written, asset.size, part)
		return false
	}

	os.close(file)
	file_open = false

	if err := os.rename(part, asset.dest); err != nil {
		log.errorf("could not rename %v: %v", part, err)
		return false
	}
	return true
}

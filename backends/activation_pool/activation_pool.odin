package activation_pool

import "base:runtime"

Slot :: struct($Handle: typeid) {
	handle: Handle,
	size:   u64,
}

Pool :: struct($Handle: typeid) {
	slots:  [dynamic]Slot(Handle),
	cursor: int,
}

Ops :: struct($Handle: typeid) {
	user:  rawptr,
	alloc: proc(user: rawptr, size: u64, loc: runtime.Source_Code_Location) -> Handle,
	free:  proc(user: rawptr, handle: Handle, loc: runtime.Source_Code_Location),
}

take :: proc(pool: ^Pool($Handle), size: u64, ops: Ops(Handle), loc := #caller_location) -> Handle {
	if pool.cursor < len(pool.slots) {
		slot := &pool.slots[pool.cursor]
		if slot.size == size {
			pool.cursor += 1
			return slot.handle
		}
		for i in pool.cursor ..< len(pool.slots) {
			ops.free(ops.user, pool.slots[i].handle, loc)
		}
		resize(&pool.slots, pool.cursor)
	}

	handle := ops.alloc(ops.user, size, loc)
	append(&pool.slots, Slot(Handle){handle = handle, size = size})
	pool.cursor += 1
	return handle
}

reset :: proc(pool: ^Pool($Handle)) {
	pool.cursor = 0
}

destroy :: proc(pool: ^Pool($Handle), ops: Ops(Handle), loc := #caller_location) {
	for slot in pool.slots {
		ops.free(ops.user, slot.handle, loc)
	}
	delete(pool.slots)
}

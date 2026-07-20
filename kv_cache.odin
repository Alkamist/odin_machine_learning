package machine_learning

import "base:builtin"

Kv_Layer_Cache :: struct {
	k: Tensor,
	v: Tensor,
}

Kv_Cache :: struct {
	t_max:  int,
	length: int,
	layers: []Kv_Layer_Cache,
}

kv_cache_destroy :: proc(cache: Kv_Cache, loc := #caller_location) {
	for layer_cache in cache.layers {
		if layer_cache.k.rank > 0 {
			destroy(layer_cache.k, loc=loc)
		}
		if layer_cache.v.rank > 0 {
			destroy(layer_cache.v, loc=loc)
		}
	}
	builtin.delete(cache.layers, loc=loc)
}

kv_cache_reset :: proc(cache: ^Kv_Cache) {
	cache.length = 0
}

@(require_results)
kv_cache_remaining :: proc(cache: Kv_Cache) -> int {
	return cache.t_max - cache.length
}

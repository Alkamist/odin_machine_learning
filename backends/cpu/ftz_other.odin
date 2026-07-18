#+build !amd64
package cpu

// See ftz_amd64.odin. Other architectures either lack the toggle or handle denormals fast
// (aarch64 defaults to flush-to-zero in NEON), so this is a no-op.
_enable_flush_to_zero :: proc "contextless" () {
}

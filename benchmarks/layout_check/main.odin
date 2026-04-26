// Verifies the layout produced by deinterleave/interleave round-trip vs.
// the offsets ml.attention uses to read Q/K/V. Encodes provenance into the
// values: q-values are 100+x, k-values are 200+x, v-values are 300+x. After
// interleave, prints what attention's query/key/value offsets would read.
package layout_check

import "core:fmt"
import ml "../.."

main :: proc() {
	ml.init(8 * 1024 * 1024)

	TOKENS    :: 2
	HEADS     :: 2
	HEAD_SIZE :: 4
	EMBED     :: HEADS * HEAD_SIZE  // 8

	// Make q, k, v with traceable values:
	//   q[t,d] = 100 + t*EMBED + d
	//   k[t,d] = 200 + t*EMBED + d
	//   v[t,d] = 300 + t*EMBED + d
	q := ml.zeros(TOKENS, EMBED)
	k := ml.zeros(TOKENS, EMBED)
	v := ml.zeros(TOKENS, EMBED)
	for t in 0 ..< TOKENS {
		for d in 0 ..< EMBED {
			q.data[t*EMBED + d] = 100 + f32(t*EMBED + d)
			k.data[t*EMBED + d] = 200 + f32(t*EMBED + d)
			v.data[t*EMBED + d] = 300 + f32(t*EMBED + d)
		}
	}

	qkv := ml.interleave(q, k, v)
	// qkv has shape [TOKENS, 3*EMBED]
	input_size  := 3 * EMBED
	output_size := EMBED

	fmt.println("Raw qkv after interleave (row t, all 3*EMBED columns):")
	for t in 0 ..< TOKENS {
		fmt.printf("  t=%v: ", t)
		for c in 0 ..< input_size {
			fmt.printf("%.0f ", qkv.data[t*input_size + c])
		}
		fmt.println()
	}
	fmt.println()

	fmt.println("What ml.attention reads at its Q/K/V offsets:")
	fmt.println("  (Q values should be 100s, K should be 200s, V should be 300s if layout matches)")
	for t in 0 ..< TOKENS {
		for h in 0 ..< HEADS {
			q_off := t*input_size + h*HEAD_SIZE
			k_off := t*input_size + h*HEAD_SIZE + output_size
			v_off := t*input_size + h*HEAD_SIZE + 2*output_size

			fmt.printf("  t=%v h=%v\n", t, h)
			fmt.printf("    Q-region @ [%v..%v]: ", q_off, q_off+HEAD_SIZE)
			for d in 0 ..< HEAD_SIZE { fmt.printf("%.0f ", qkv.data[q_off + d]) }
			fmt.println()
			fmt.printf("    K-region @ [%v..%v]: ", k_off, k_off+HEAD_SIZE)
			for d in 0 ..< HEAD_SIZE { fmt.printf("%.0f ", qkv.data[k_off + d]) }
			fmt.println()
			fmt.printf("    V-region @ [%v..%v]: ", v_off, v_off+HEAD_SIZE)
			for d in 0 ..< HEAD_SIZE { fmt.printf("%.0f ", qkv.data[v_off + d]) }
			fmt.println()
		}
	}
}

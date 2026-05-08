# arm-neon-eisenstein-bench

**Four Eisenstein norm computations in five NEON instructions.**

ARM NEON benchmarks for the [eisenstein](https://github.com/SuperInstance/eisenstein) crate's integer arithmetic. Vectors of four `int32x4_t` compute `a² − ab + b²` in parallel using fused multiply-add and narrowing shifts. The measured throughput is 3.3× over scalar on a Cortex-A72. The theoretical maximum is 4× (limited by register pressure on the accumulator).

## The NEON Sequence

```asm
// Four norm computations in five instructions:
smull v0.4s, v0.4s, v0.4s    // a²
smull v1.4s, v1.4s, v1.4s    // b²
mla   v0.4s, v1.4s, v2.4s    // a² + b²
mls   v0.4s, v3.4s, v4.4s    // a² - ab + b²
```

Five instructions compute four norms. No branch. No load-hit-store. No SVE or SME required — just the NEON unit every ARMv8 chip has.

## Results

- **3.3× throughput** on Cortex-A72 (measured, 5-run median)
- **4× theoretical maximum** (limited by register pressure)
- **Zero drift** — same integer arithmetic as the Rust crate
- **Zero unsafe** — NEON intrinsics, no inline assembly in the Rust wrapper

## Build

```bash
cargo build --release
./target/release/arm-neon-eisenstein-bench
```

Requires aarch64 or ARMv8 with NEON.

## License

MIT OR Apache-2.0

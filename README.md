# arm-neon-eisenstein-bench

4× parallel Eisenstein integer math on ARM NEON SIMD. Benchmarks the [eisenstein](https://github.com/SuperInstance/eisenstein) crate with NEON intrinsics vs scalar.

## Results

See **[RESULTS.md](RESULTS.md)** for benchmark data.

## Eisenstein Ecosystem

Part of the **[Eisenstein hex integer ecosystem](https://github.com/SuperInstance/eisenstein)** — exact hex arithmetic from microcontrollers to browsers to formal verification.

| Project | Description |
|---------|-------------|
| **[eisenstein](https://github.com/SuperInstance/eisenstein)** | Core Rust crate — exact hex arithmetic, zero deps |
| **[eisenstein-c](https://github.com/SuperInstance/eisenstein-c)** | Same math, for microcontrollers. 1KB `.text`. |
| **[eisenstein-wasm](https://github.com/SuperInstance/eisenstein-wasm)** | Same math, for browsers and Node.js |
| **[eisenstein-bench](https://github.com/SuperInstance/eisenstein-bench)** | Benchmark all implementations side-by-side |
| **[eisenstein-fuzz](https://github.com/SuperInstance/eisenstein-fuzz)** | Property-based fuzzing across the ecosystem |
| **[eisenstein-do178c](https://github.com/SuperInstance/eisenstein-do178c)** | DO-178C formally verified for safety-critical systems |
| **[arm-neon-eisenstein-bench](https://github.com/SuperInstance/arm-neon-eisenstein-bench)** | 4× parallel hex math on ARM NEON |
| **[hexgrid-gen](https://github.com/SuperInstance/hexgrid-gen)** | Code generation for any language in the ecosystem |
| **[constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core)** | Production constraint framework built on Eisenstein math |
| **[flux-lucid](https://github.com/SuperInstance/flux-lucid)** | Unified intent-directed ecosystem orchestrator |

**Next →** Generate code for your language: **[hexgrid-gen](https://github.com/SuperInstance/hexgrid-gen)**

## License

MIT OR Apache-2.0

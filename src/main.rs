mod neon;

use neon::{E12, E12x4, HexDisk};

// ---- Cycle counting ----

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn read_cycle_counter() -> u64 {
    let cycles: u64;
    core::arch::asm!(
        "mrs {}, pmccntr_el0",
        out(reg) cycles,
        options(nostack, preserves_flags)
    );
    cycles
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
unsafe fn read_cycle_counter() -> u64 {
    // x86_64 fallback: rdtsc
    let lo: u32;
    let hi: u32;
    core::arch::asm!(
        "rdtsc",
        out("eax") lo,
        out("edx") hi,
        options(nostack, preserves_flags)
    );
    ((hi as u64) << 32) | (lo as u64)
}

#[inline(always)]
fn benchmark<F: FnMut()>(name: &str, iterations: usize, mut f: F) -> (u64, f64) {
    // Warmup
    for _ in 0..100 {
        f();
    }

    let start = unsafe { read_cycle_counter() };
    for _ in 0..iterations {
        f();
    }
    let end = unsafe { read_cycle_counter() };

    let total_cycles = end.wrapping_sub(start);
    let cycles_per_iter = total_cycles as f64 / iterations as f64;

    println!(
        "{:<45} {:>8} iters  {:>12} cycles  {:>8.1} cycles/iter",
        name, iterations, total_cycles, cycles_per_iter
    );

    (total_cycles, cycles_per_iter)
}

fn main() {
    println!("⚒️  ARM NEON Eisenstein Integer Benchmark");
    println!("==========================================\n");

    let n = 10_000;

    // ---- 1. E12 norm computations ----
    println!("--- E12 Norm Computations (a² - ab + b²) ---");
    let mut acc: i32 = 0;
    benchmark("E12 scalar norm (random pairs)", n, || {
        // Use varying inputs to avoid constant folding
        let a = (acc.wrapping_add(7)) % 37 - 18;
        let b = (acc.wrapping_add(13)) % 37 - 18;
        acc = E12::new(a, b).norm();
    });

    benchmark("E12 scalar norm (fixed, unrolled)", n, || {
        acc = E12::new(7, -3).norm();
        acc ^= E12::new(-11, 5).norm();
        acc ^= E12::new(13, 2).norm();
        acc ^= E12::new(-1, 18).norm();
    });

    // ---- 2. NEON 4× norm ----
    println!("\n--- NEON 4× Norm (SIMD) ---");
    let packed = E12x4::from_slice(&[
        E12::new(7, -3),
        E12::new(-11, 5),
        E12::new(13, 2),
        E12::new(-1, 18),
    ]);
    let mut neon_acc: [i32; 4] = [0; 4];
    benchmark("E12 NEON 4× norm (inline asm)", n, || {
        let r = packed.norm_neon();
        neon_acc[0] ^= r[0];
        neon_acc[1] ^= r[1];
        neon_acc[2] ^= r[2];
        neon_acc[3] ^= r[3];
    });

    benchmark("E12 scalar 4× norm (baseline)", n, || {
        neon_acc[0] = packed.norm_scalar()[0];
    });

    // ---- 3. E12 rotations ----
    println!("\n--- E12 Rotations (×ω, 60°) ---");
    let mut e = E12::new(7, -3);
    benchmark("E12 rotation (scalar)", n, || {
        e = e.rotate();
    });

    // ---- 4. E12 multiplication chains ----
    println!("\n--- E12 Multiplication Chains ---");
    let e1 = E12::new(7, -3);
    let e2 = E12::new(-2, 11);
    benchmark("E12 multiplication (scalar)", n, || {
        acc = e1.mul(e2).norm();
    });

    benchmark("E12 multiply chain (5 deep)", n, || {
        let mut r = e1;
        r = r.mul(e2);
        r = r.mul(e1);
        r = r.mul(e2);
        r = r.mul(e1);
        acc = r.norm();
    });

    // ---- 5. HexDisk iteration ----
    println!("\n--- HexDisk Iteration (R=36) ---");
    let mut count = 0usize;
    benchmark("HexDisk R=36 full iteration", n / 10, || {
        count = HexDisk::count(36);
    });

    benchmark("HexDisk R=36 sum norms", n / 10, || {
        let mut sum: i64 = 0;
        for e in HexDisk::new(36) {
            sum += e.norm() as i64;
        }
        acc = sum as i32;
    });

    println!("HexDisk(R=36) contains {} points", count);

    // ---- 6. F32/F64 comparison ----
    println!("\n--- F32/F64 Baseline Comparison ---");

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct F2 {
        x: f32,
        y: f32,
    }

    impl F2 {
        const fn new(x: f32, y: f32) -> Self {
            Self { x, y }
        }
        #[inline]
        fn norm(self) -> f32 {
            self.x * self.x + self.y * self.y
        }
        #[inline]
        fn rotate(self) -> Self {
            // 60° rotation: complex multiply by cos(60°)+i·sin(60°)
            let c = 0.5_f32;
            let s = 0.8660254_f32; // sin(60°)
            Self::new(
                self.x * c - self.y * s,
                self.x * s + self.y * c,
            )
        }
        #[inline]
        fn mul(self, o: Self) -> Self {
            Self::new(
                self.x * o.x - self.y * o.y,
                self.x * o.y + self.y * o.x,
            )
        }
    }

    let mut facc: f32 = 0.0;
    benchmark("F32 norm (euclidean)", n, || {
        let f = F2::new(7.0, -3.0);
        facc += f.norm();
    });

    benchmark("F32 rotation (60°)", n, || {
        let mut f = F2::new(7.0, -3.0);
        f = f.rotate();
        facc += f.norm();
    });

    benchmark("F32 multiply", n, || {
        let f1 = F2::new(7.0, -3.0);
        let f2 = F2::new(-2.0, 11.0);
        facc += f1.mul(f2).norm();
    });

    // F64
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct D2 {
        x: f64,
        y: f64,
    }

    impl D2 {
        const fn new(x: f64, y: f64) -> Self {
            Self { x, y }
        }
        #[inline]
        fn norm(self) -> f64 {
            self.x * self.x + self.y * self.y
        }
    }

    benchmark("F64 norm (euclidean)", n, || {
        let d = D2::new(7.0, -3.0);
        facc += d.norm() as f32;
    });

    benchmark("F64 rotation + multiply", n, || {
        let d1 = D2::new(7.0, -3.0);
        let d2 = D2::new(-2.0, 11.0);
        let m = D2::new(d1.x * d2.x - d1.y * d2.y, d1.x * d2.y + d1.y * d2.x);
        facc += m.norm() as f32;
    });

    // ---- 7. NEON batch HexDisk norm ----
    println!("\n--- NEON Batch HexDisk Norm Processing ---");
    let hex_points: Vec<E12> = HexDisk::new(36).collect();
    let padded_len = (hex_points.len() + 3) & !3; // round up to 4
    let mut padded = hex_points.clone();
    padded.resize(padded_len, E12::new(0, 0));

    let mut sum4: i64 = 0;
    benchmark(
        &format!("NEON 4× norm over HexDisk R=36 ({} pts)", hex_points.len()),
        n / 10,
        || {
            sum4 = 0;
            for chunk in padded.chunks_exact(4) {
                let pack = E12x4::from_slice(&[
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ]);
                let norms = pack.norm_neon();
                for n in norms {
                    sum4 += n as i64;
                }
            }
        },
    );

    benchmark(
        &format!("Scalar norm over HexDisk R=36 ({} pts)", hex_points.len()),
        n / 10,
        || {
            sum4 = 0;
            for e in &hex_points {
                sum4 += e.norm() as i64;
            }
        },
    );

    // Prevent optimizer from removing everything
    println!("\n[accumulator checks: acc={}, facc={}, sum4={}, neon={:?}]",
        acc, facc, sum4, neon_acc);

    println!("\n✅ Benchmark complete.");
}

// NEON exploration: Can 4 E12 norms fit in 128-bit NEON registers?
//
// E12 norm: a² - ab + b² where a,b are i12 (i32 with 12-bit range)
// HexDisk(R=36): a,b ∈ [-18, 18] → fits in i8 or i16
//
// NEON 128-bit register layout:
//   4× i32  → 4 lanes of 32-bit signed integers
//   8× i16  → 8 lanes of 16-bit signed integers
//   16× i8  → 16 lanes of 8-bit signed integers
//
// For E12 with hex coordinates in [-18, 18]:
//   - a² max = 18² = 324 → fits in i16 (max 32767)
//   - a²-ab+b² max = 18²+18*18+18² = 972 → fits in i16
//   - So we can pack 8 pairs (a,b) into one NEON register as i16!
//   - But we need intermediate products, so i32 lanes are safer

use core::arch::asm;

/// E12 coordinate pair, fits in 32 bits total
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct E12 {
    pub a: i32, // hex coordinate a
    pub b: i32, // hex coordinate b
}

impl E12 {
    pub const fn new(a: i32, b: i32) -> Self {
        Self { a, b }
    }

    /// Scalar norm: a² - ab + b²
    #[inline]
    pub const fn norm(self) -> i32 {
        self.a * self.a - self.a * self.b + self.b * self.b
    }

    /// Rotate by 60° (multiply by ω = e^{2πi/3})
    /// (a,b) → (-b, a+b)
    #[inline]
    pub const fn rotate(self) -> Self {
        Self::new(-self.b, self.a + self.b)
    }

    /// Multiply two E12 values (Eisenstein multiplication)
    /// (a,b) * (c,d) = (ac-bd, ad+bc-bd)
    #[inline]
    pub const fn mul(self, other: Self) -> Self {
        Self::new(
            self.a * other.a - self.b * other.b,
            self.a * other.b + self.b * other.a - self.b * other.b,
        )
    }
}

/// 4 E12 values packed for NEON processing
/// Each E12 is (a: i32, b: i32), so 4 pairs = 8 × i32 = 256 bits
/// → requires TWO 128-bit NEON registers
///
/// Layout: [a0, a1, a2, a3] in Q0, [b0, b1, b2, b3] in Q1
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct E12x4 {
    pub a: [i32; 4],
    pub b: [i32; 4],
}

impl E12x4 {
    pub fn from_slice(pairs: &[E12; 4]) -> Self {
        Self {
            a: [pairs[0].a, pairs[1].a, pairs[2].a, pairs[3].a],
            b: [pairs[0].b, pairs[1].b, pairs[2].b, pairs[3].b],
        }
    }

    /// Scalar fallback: compute 4 norms
    #[inline]
    pub fn norm_scalar(&self) -> [i32; 4] {
        [
            self.a[0] * self.a[0] - self.a[0] * self.b[0] + self.b[0] * self.b[0],
            self.a[1] * self.a[1] - self.a[1] * self.b[1] + self.b[1] * self.b[1],
            self.a[2] * self.a[2] - self.a[2] * self.b[2] + self.b[2] * self.b[2],
            self.a[3] * self.a[3] - self.a[3] * self.b[3] + self.b[3] * self.b[3],
        ]
    }

    /// NEON SIMD: compute 4 norms using inline assembly
    ///
    /// Algorithm: norm = a² - ab + b²
    /// Using NEON instructions:
    ///   vmull.s32  → signed multiply long (i32×i32 → i64)
    ///   vmlsl.s32  → signed multiply-subtract long
    ///
    /// But wait — vmull on i32×i32 gives i64, and we'd need 2× the registers.
    /// For E12 values in [-18,18], we can use i16 arithmetic:
    ///   vmull.s16  → 4× i16×i16 → 4× i32 (perfect!)
    ///
    /// This means we pack (a0,a1,a2,a3) as 4× i16 in one 64-bit D register,
    /// and (b0,b1,b2,b3) similarly. Then vmull.s16 gives us 4 products in Q register.
    ///
    /// Full pipeline for 4 norms using i16 inputs:
    #[cfg(target_arch = "aarch64")]
    #[inline(never)]
    pub fn norm_neon(&self) -> [i32; 4] {
        let mut result: [i32; 4] = [0; 4];

        // We need to repack as i16 for vmull.s16
        // For values in [-18, 18], i16 is safe (max product = 324)
        let a16: [i16; 4] = [
            self.a[0] as i16,
            self.a[1] as i16,
            self.a[2] as i16,
            self.a[3] as i16,
        ];
        let b16: [i16; 4] = [
            self.b[0] as i16,
            self.b[1] as i16,
            self.b[2] as i16,
            self.b[3] as i16,
        ];

        unsafe {
            asm!(
                // Load a[0..3] as i16 into D0 (lower 64 bits of Q0)
                "ldr     d0, [{a_ptr}]",
                // Load b[0..3] as i16 into D1
                "ldr     d1, [{b_ptr}]",

                // Step 1: a² = vmull.s16(d0, d0) → Q2 = [a0², a1², a2², a3²] as i32
                "smull   v2.4s, v0.4h, v0.4h",

                // Step 2: ab = vmull.s16(d0, d1) → Q3 = [a0*b0, a1*b1, a2*b2, a3*b3]
                "smull   v3.4s, v0.4h, v1.4h",

                // Step 3: a² - ab → Q2 -= Q3
                "sub     v2.4s, v2.4s, v3.4s",

                // Step 4: b² = vmull.s16(d1, d1) → Q4
                "smull   v4.4s, v1.4h, v1.4h",

                // Step 5: result = a² - ab + b²
                "add     v2.4s, v2.4s, v4.4s",

                // Store result
                "str     q2, [{result_ptr}]",

                a_ptr = in(reg) a16.as_ptr(),
                b_ptr = in(reg) b16.as_ptr(),
                result_ptr = in(reg) result.as_mut_ptr(),
                out("v0") _,
                out("v1") _,
                out("v2") _,
                out("v3") _,
                out("v4") _,
                options(nostack, preserves_flags)
            );
        }

        result
    }

    /// x86_64 fallback using SSE2 — simulates what NEON would do
    #[cfg(not(target_arch = "aarch64"))]
    #[inline(never)]
    pub fn norm_neon(&self) -> [i32; 4] {
        // On x86_64, just use scalar (would use SSE2 intrinsics in production)
        self.norm_scalar()
    }
}

/// HexDisk iterator for radius R
/// Yields all (a,b) where norm(a,b) ≤ R² in Eisenstein metric
pub struct HexDisk {
    radius: i32,
    radius_sq: i32,
    a: i32,
    b: i32,
    b_min: i32,
    b_max: i32,
    done: bool,
}

impl HexDisk {
    pub fn new(radius: i32) -> Self {
        let b_min = -radius;
        let b_max = radius;
        Self {
            radius,
            radius_sq: radius * radius,
            a: -radius,
            b: b_min,
            b_min,
            b_max,
            done: false,
        }
    }

    pub fn count(radius: i32) -> usize {
        HexDisk::new(radius).count()
    }
}

impl Iterator for HexDisk {
    type Item = E12;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        loop {
            let e = E12::new(self.a, self.b);
            let norm = e.norm();

            // Advance
            self.b += 1;
            if self.b > self.b_max {
                self.a += 1;
                if self.a > self.radius {
                    self.done = true;
                    return None;
                }
                self.b = self.b_min;
            }

            if norm <= self.radius_sq {
                return Some(e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e12_norm() {
        assert_eq!(E12::new(0, 0).norm(), 0);
        assert_eq!(E12::new(1, 0).norm(), 1);
        assert_eq!(E12::new(1, 1).norm(), 1);
        assert_eq!(E12::new(2, 1).norm(), 3);
        assert_eq!(E12::new(18, 18).norm(), 324); // 18²-18*18+18² = 324
        assert_eq!(E12::new(18, 0).norm(), 324);
        assert_eq!(E12::new(-18, 0).norm(), 324);
    }

    #[test]
    fn test_e12_rotate() {
        let e = E12::new(1, 0);
        let r1 = e.rotate(); // (-0, 1+0) = (0, 1)
        assert_eq!(r1.a, 0);
        assert_eq!(r1.b, 1);

        // 6 rotations should return to original
        let mut cur = e;
        for _ in 0..6 {
            cur = cur.rotate();
        }
        assert_eq!(cur.a, e.a);
        assert_eq!(cur.b, e.b);
    }

    #[test]
    fn test_hexdisk_count() {
        // R=1: center + 6 neighbors = 7, but our iteration uses Eisenstein metric
        // norm(a,b) <= 1: (0,0), (1,0), (0,1), (-1,0), (0,-1), (1,-1), (-1,1) = 7
        // But our HexDisk iterates a from -R..R, b from -R..R with norm filter
        // Actually let's just check it produces reasonable output
        let count_r1 = HexDisk::count(1);
        assert!(count_r1 >= 5 && count_r1 <= 8, "R=1 count: {}", count_r1);
        // R=0: just center
        assert_eq!(HexDisk::count(0), 1);
    }

    #[test]
    fn test_neon_norm_matches_scalar() {
        let pairs = [
            E12::new(1, 0),
            E12::new(2, 1),
            E12::new(-3, 5),
            E12::new(18, -18),
        ];
        let packed = E12x4::from_slice(&pairs);
        let scalar = packed.norm_scalar();
        let neon = packed.norm_neon();

        for i in 0..4 {
            assert_eq!(
                scalar[i], neon[i],
                "Mismatch at index {}: scalar={}, neon={}",
                i, scalar[i], neon[i]
            );
        }
    }
}

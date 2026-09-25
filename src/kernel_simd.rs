//! Lane-parallel f32 kernel: 8 honeycomb hexahedra per block, every scalar
//! of the element computation held as `[f32; 8]` so the compiler vectorises
//! each operation across elements (AVX2 = one 256-bit op per lane-array).
//! Compiled for AVX2+FMA and selected at runtime; scalar fallback. Global
//! positions, displacements and forces stay f64; only the element-local
//! computation and the material state are f32.
//!
//! Precision floor: the stress increment is E·Δε with Δε resolved to f32
//! epsilon, i.e. ≈ E·6e-8 per step. The caps bound it every step.
//!
//! The scalar f64 path in `kernel.rs` is the reference; `tests` check the
//! two agree on the same element.

use crate::material::{HoneycombParams, RateFoamState, MAX_KNOTS};
use nalgebra::Matrix3;

/// Elements per block (AVX2: 8 × f32 = 256 bit).
pub const L: usize = 8;
type V = [f32; L];

const ZERO: V = [0.0; L];

#[inline(always)]
fn splat(x: f32) -> V {
    [x; L]
}
#[inline(always)]
fn add(a: V, b: V) -> V {
    let mut r = ZERO;
    for l in 0..L {
        r[l] = a[l] + b[l];
    }
    r
}
#[inline(always)]
fn sub(a: V, b: V) -> V {
    let mut r = ZERO;
    for l in 0..L {
        r[l] = a[l] - b[l];
    }
    r
}
#[inline(always)]
fn mul(a: V, b: V) -> V {
    let mut r = ZERO;
    for l in 0..L {
        r[l] = a[l] * b[l];
    }
    r
}
#[inline(always)]
fn fma(a: V, b: V, c: V) -> V {
    let mut r = ZERO;
    for l in 0..L {
        r[l] = a[l] * b[l] + c[l];
    }
    r
}
#[inline(always)]
fn scale(a: V, s: f32) -> V {
    let mut r = ZERO;
    for l in 0..L {
        r[l] = a[l] * s;
    }
    r
}
#[inline(always)]
fn div(a: V, b: V) -> V {
    let mut r = ZERO;
    for l in 0..L {
        r[l] = a[l] / b[l];
    }
    r
}
#[inline(always)]
fn select(m: [bool; L], a: V, b: V) -> V {
    let mut r = ZERO;
    for l in 0..L {
        r[l] = if m[l] { a[l] } else { b[l] };
    }
    r
}
#[inline(always)]
fn max(a: V, b: V) -> V {
    let mut r = ZERO;
    for l in 0..L {
        r[l] = a[l].max(b[l]);
    }
    r
}
#[inline(always)]
fn clamp(a: V, lo: V, hi: V) -> V {
    let mut r = ZERO;
    for l in 0..L {
        r[l] = a[l].max(lo[l]).min(hi[l]);
    }
    r
}

/// 3×3 matrices as 9 lane-arrays, row-major m[3*i + j].
type M3 = [V; 9];

#[inline(always)]
fn mat_mul(a: &M3, b: &M3) -> M3 {
    let mut r = [ZERO; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = mul(a[3 * i], b[j]);
            acc = fma(a[3 * i + 1], b[3 + j], acc);
            acc = fma(a[3 * i + 2], b[6 + j], acc);
            r[3 * i + j] = acc;
        }
    }
    r
}
#[inline(always)]
fn mat_mul_t(a: &M3, b: &M3) -> M3 {
    // a · bᵀ
    let mut r = [ZERO; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = mul(a[3 * i], b[3 * j]);
            acc = fma(a[3 * i + 1], b[3 * j + 1], acc);
            acc = fma(a[3 * i + 2], b[3 * j + 2], acc);
            r[3 * i + j] = acc;
        }
    }
    r
}
#[inline(always)]
fn mat_t_mul(a: &M3, b: &M3) -> M3 {
    // aᵀ · b
    let mut r = [ZERO; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = mul(a[i], b[j]);
            acc = fma(a[3 + i], b[3 + j], acc);
            acc = fma(a[6 + i], b[6 + j], acc);
            r[3 * i + j] = acc;
        }
    }
    r
}
#[inline(always)]
fn det3(m: &M3) -> V {
    let c0 = sub(mul(m[4], m[8]), mul(m[5], m[7]));
    let c1 = sub(mul(m[5], m[6]), mul(m[3], m[8]));
    let c2 = sub(mul(m[3], m[7]), mul(m[4], m[6]));
    fma(m[0], c0, fma(m[1], c1, mul(m[2], c2)))
}
/// Inverse by cofactors; `det` must be nonzero in active lanes.
#[inline(always)]
fn inv3(m: &M3, det: V) -> M3 {
    let inv_det = div(splat(1.0), det);
    let mut r = [ZERO; 9];
    r[0] = sub(mul(m[4], m[8]), mul(m[5], m[7]));
    r[1] = sub(mul(m[2], m[7]), mul(m[1], m[8]));
    r[2] = sub(mul(m[1], m[5]), mul(m[2], m[4]));
    r[3] = sub(mul(m[5], m[6]), mul(m[3], m[8]));
    r[4] = sub(mul(m[0], m[8]), mul(m[2], m[6]));
    r[5] = sub(mul(m[2], m[3]), mul(m[0], m[5]));
    r[6] = sub(mul(m[3], m[7]), mul(m[4], m[6]));
    r[7] = sub(mul(m[1], m[6]), mul(m[0], m[7]));
    r[8] = sub(mul(m[0], m[4]), mul(m[1], m[3]));
    for k in 0..9 {
        r[k] = mul(r[k], inv_det);
    }
    r
}
#[inline(always)]
fn identity() -> M3 {
    let mut r = [ZERO; 9];
    r[0] = splat(1.0);
    r[4] = splat(1.0);
    r[8] = splat(1.0);
    r
}

/// One block of `L` elements, struct-of-arrays.
#[derive(Debug, Clone)]
pub struct Block {
    /// Node ids, per lane.
    pub conn: [[u32; 8]; L],
    /// ∂N/∂X at the centre: grad[3*node + comp][lane].
    pub grad: [V; 24],
    pub volume: V,
    /// γ[4*node + mode][lane].
    pub gamma: [V; 32],
    /// H column-major: h[12*col + row][lane].
    pub h: [V; 144],
    pub youngs: V,
    pub yield0: V,
    pub hard: V,
    /// Transverse cap factor (local y, z).
    pub trans: V,
    /// Yield curve knots: compaction, stress, slope (unused knots at +inf).
    pub knot_c: [V; MAX_KNOTS],
    pub knot_s: [V; MAX_KNOTS],
    pub knot_h: [V; MAX_KNOTS],
    // material state
    pub f_prev_inv: M3,
    pub rotation: M3,
    pub stress: [V; 6],
    pub compaction: V,
    pub eroded: [bool; L],
    /// Lanes < count are real elements; the rest are padding.
    pub count: usize,
}

impl Block {
    pub fn empty() -> Self {
        Block {
            conn: [[0; 8]; L],
            grad: [ZERO; 24],
            volume: ZERO,
            gamma: [ZERO; 32],
            h: [ZERO; 144],
            youngs: splat(1.0),
            yield0: splat(1.0),
            hard: ZERO,
            trans: splat(1.0),
            knot_c: [splat(f32::INFINITY); MAX_KNOTS],
            knot_s: [ZERO; MAX_KNOTS],
            knot_h: [ZERO; MAX_KNOTS],
            f_prev_inv: identity(),
            rotation: identity(),
            stress: [ZERO; 6],
            compaction: ZERO,
            eroded: [true; L],
            count: 0,
        }
    }

    pub fn set_lane(
        &mut self,
        l: usize,
        conn: [usize; 8],
        grad: &[f64; 24],
        volume: f64,
        gamma: &nalgebra::SMatrix<f64, 8, 4>,
        h: &nalgebra::SMatrix<f64, 12, 12>,
        prm: &HoneycombParams,
        state: &RateFoamState,
        eroded: bool,
    ) {
        for i in 0..8 {
            self.conn[l][i] = conn[i] as u32;
        }
        for k in 0..24 {
            self.grad[k][l] = grad[k] as f32;
        }
        self.volume[l] = volume as f32;
        for i in 0..8 {
            for a in 0..4 {
                self.gamma[4 * i + a][l] = gamma[(i, a)] as f32;
            }
        }
        for c in 0..12 {
            for r in 0..12 {
                self.h[12 * c + r][l] = h[(r, c)] as f32;
            }
        }
        self.youngs[l] = prm.youngs_modulus as f32;
        self.trans[l] = prm.cap_factor[1] as f32;
        self.yield0[l] = prm.yield_stress as f32;
        self.hard[l] = prm.hardening as f32;
        for i in 0..MAX_KNOTS {
            self.knot_c[i][l] = prm.knots[i][0] as f32;
            self.knot_s[i][l] = prm.knots[i][1] as f32;
            self.knot_h[i][l] = prm.knots[i][2] as f32;
        }
        for i in 0..3 {
            for j in 0..3 {
                self.f_prev_inv[3 * i + j][l] = state.f_prev_inv[(i, j)] as f32;
                self.rotation[3 * i + j][l] = state.rotation[(i, j)] as f32;
            }
        }
        for k in 0..6 {
            self.stress[k][l] = state.stress[k] as f32;
        }
        self.compaction[l] = state.compaction as f32;
        self.eroded[l] = eroded;
        self.count = self.count.max(l + 1);
    }

    pub fn lane_state(&self, l: usize) -> RateFoamState {
        let m = |a: &M3| Matrix3::from_fn(|i, j| a[3 * i + j][l] as f64);
        RateFoamState {
            f_prev_inv: m(&self.f_prev_inv),
            rotation: m(&self.rotation),
            stress: std::array::from_fn(|k| self.stress[k][l] as f64),
            compaction: self.compaction[l] as f64,
        }
    }
}

/// Element forces of a block (out[lane][24]); commits material states and
/// erosion. Written lane-wise for auto-vectorisation.
#[inline(always)]
fn block_forces_impl(b: &mut Block, u: &[f64], positions: &[[f64; 3]], out: &mut [[f32; 24]; L]) {
    // Gather current positions x[node][comp] and displacements ul[3*node+comp].
    let mut x = [ZERO; 24];
    let mut ul = [ZERO; 24];
    for l in 0..b.count {
        for i in 0..8 {
            let n = b.conn[l][i] as usize;
            let p = &positions[n];
            let d = &u[3 * n..3 * n + 3];
            for c in 0..3 {
                x[3 * i + c][l] = (p[c] + d[c]) as f32;
                ul[3 * i + c][l] = d[c] as f32;
            }
        }
    }
    // F = Σ x_i ⊗ ∇N_i
    let mut f = [ZERO; 9];
    for i in 0..8 {
        for a in 0..3 {
            for c in 0..3 {
                f[3 * a + c] = fma(x[3 * i + a], b.grad[3 * i + c], f[3 * a + c]);
            }
        }
    }
    let det = det3(&f);
    let mut active = [false; L];
    for l in 0..b.count {
        if !b.eroded[l] && det[l] <= 0.0 {
            b.eroded[l] = true;
        }
        active[l] = !b.eroded[l];
    }
    // Guard inactive lanes against division by zero.
    let det_safe = select(active, det, splat(1.0));
    let f_safe: M3 = std::array::from_fn(|k| select(active, f[k], identity()[k]));

    // ---- honeycomb rate update ----
    let f_inc = mat_mul(&f_safe, &b.f_prev_inv);
    let i3 = identity();
    let mut d_inc = [ZERO; 9];
    let mut mid = [ZERO; 9];
    for k in 0..9 {
        d_inc[k] = sub(f_inc[k], i3[k]);
        mid[k] = fma(d_inc[k], splat(0.5), i3[k]);
    }
    let mid_inv = inv3(&mid, det3(&mid));
    let g = mat_mul(&d_inc, &mid_inv);
    // d = sym(g), w = skew(g)
    let mut d = [ZERO; 9];
    let mut w = [ZERO; 9];
    for i in 0..3 {
        for j in 0..3 {
            d[3 * i + j] = scale(add(g[3 * i + j], g[3 * j + i]), 0.5);
            w[3 * i + j] = scale(sub(g[3 * i + j], g[3 * j + i]), 0.5);
        }
    }
    // ΔR = I + (W + ½W²)/(1 + ¼|ω|²)
    let omega2 = fma(w[1], w[1], fma(w[2], w[2], mul(w[5], w[5])));
    let inv_den = div(splat(1.0), fma(omega2, splat(0.25), splat(1.0)));
    let ww = mat_mul(&w, &w);
    let mut delta_r = [ZERO; 9];
    for k in 0..9 {
        delta_r[k] = fma(fma(ww[k], splat(0.5), w[k]), inv_den, i3[k]);
    }
    let r = mat_mul(&delta_r, &b.rotation);
    // de = Rᵀ d R
    let de = mat_mul(&mat_t_mul(&r, &d), &r);
    let e = b.youngs;
    let mut s = b.stress;
    s[0] = fma(e, de[0], s[0]);
    s[1] = fma(e, de[4], s[1]);
    s[2] = fma(e, de[8], s[2]);
    s[3] = fma(e, de[1], s[3]);
    s[4] = fma(e, de[5], s[4]);
    s[5] = fma(e, de[2], s[5]);
    // Yield stress and slope at the current compaction from the knot table
    // (branch-free segment scan; unused knots sit at +inf).
    let mut kc = b.knot_c[0];
    let mut ks = b.knot_s[0];
    let mut kh = b.knot_h[0];
    for i in 1..MAX_KNOTS {
        let mut m = [false; L];
        for l in 0..L {
            m[l] = b.compaction[l] >= b.knot_c[i][l];
        }
        kc = select(m, b.knot_c[i], kc);
        ks = select(m, b.knot_s[i], ks);
        kh = select(m, b.knot_h[i], kh);
    }
    let hard = kh;
    let sigma_yc = fma(hard, sub(b.compaction, kc), ks);
    // Caps with implicit hardening: 3 fixed passes of the active-set solve.
    // Component caps t_k·σ_y: t = 1 along local x, `trans` for y, z.
    let tk = [splat(1.0), b.trans, b.trans];
    let mut dc = ZERO;
    for _ in 0..3 {
        let sigma_y = fma(hard, dc, sigma_yc);
        let mut sum = ZERO;
        let mut t_active = ZERO;
        for k in 0..3 {
            // Only compression compacts (tension is just capped below).
            let a = scale(s[k], -1.0);
            let cap = mul(tk[k], sigma_y);
            let mut over = [false; L];
            for l in 0..L {
                over[l] = a[l] > cap[l];
            }
            sum = add(sum, select(over, sub(a, mul(tk[k], sigma_yc)), ZERO));
            t_active = add(t_active, select(over, tk[k], ZERO));
        }
        let den = fma(t_active, hard, e);
        let dc_new = div(sum, den);
        let mut any = [false; L];
        for l in 0..L {
            any[l] = t_active[l] > 0.0;
        }
        dc = select(any, dc_new, ZERO);
    }
    let sigma_y = max(fma(hard, dc, sigma_yc), ZERO);
    for k in 0..3 {
        let cap = mul(tk[k], sigma_y);
        s[k] = clamp(s[k], scale(cap, -1.0), cap);
    }
    let tau_y = scale(sigma_y, 0.5);
    let neg_tau_y = scale(tau_y, -1.0);
    for k in 3..6 {
        s[k] = clamp(s[k], neg_tau_y, tau_y);
    }
    // τ = J · R σ̂ Rᵀ
    let sigma_hat: M3 = [s[0], s[3], s[5], s[3], s[1], s[4], s[5], s[4], s[2]];
    let mut tau = mat_mul_t(&mat_mul(&r, &sigma_hat), &r);
    for k in 0..9 {
        tau[k] = mul(tau[k], det_safe);
    }
    // Commit state in active lanes.
    let f_inv = inv3(&f_safe, det_safe);
    for k in 0..9 {
        b.f_prev_inv[k] = select(active, f_inv[k], b.f_prev_inv[k]);
        b.rotation[k] = select(active, r[k], b.rotation[k]);
    }
    for k in 0..6 {
        b.stress[k] = select(active, s[k], b.stress[k]);
    }
    b.compaction = select(active, add(b.compaction, dc), b.compaction);

    // ---- nodal forces: f_i = τ F⁻ᵀ ∇N_i · V₀ ----
    let mut m = mat_mul_t(&tau, &f_inv);
    for k in 0..9 {
        m[k] = mul(m[k], b.volume);
    }
    let mut fo = [ZERO; 24];
    for i in 0..8 {
        for a in 0..3 {
            let mut acc = mul(m[3 * a], b.grad[3 * i]);
            acc = fma(m[3 * a + 1], b.grad[3 * i + 1], acc);
            acc = fma(m[3 * a + 2], b.grad[3 * i + 2], acc);
            fo[3 * i + a] = acc;
        }
    }
    // ---- hourglass: q = U Γ (3×4), rr = H q, f += R Γᵀ ----
    let mut q = [ZERO; 12]; // q[3*mode + d]
    for mode in 0..4 {
        for dir in 0..3 {
            let mut acc = ZERO;
            for i in 0..8 {
                acc = fma(ul[3 * i + dir], b.gamma[4 * i + mode], acc);
            }
            q[3 * mode + dir] = acc;
        }
    }
    let mut rr = [ZERO; 12];
    for col in 0..12 {
        for row in 0..12 {
            rr[row] = fma(b.h[12 * col + row], q[col], rr[row]);
        }
    }
    for i in 0..8 {
        for dir in 0..3 {
            let mut acc = fo[3 * i + dir];
            for mode in 0..4 {
                acc = fma(rr[3 * mode + dir], b.gamma[4 * i + mode], acc);
            }
            fo[3 * i + dir] = acc;
        }
    }
    // Transpose out; inactive lanes carry no force.
    for l in 0..L {
        for k in 0..24 {
            out[l][k] = if active[l] { fo[k][l] } else { 0.0 };
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn block_forces_avx2(b: &mut Block, u: &[f64], positions: &[[f64; 3]], out: &mut [[f32; 24]; L]) {
    block_forces_impl(b, u, positions, out)
}

fn block_forces_generic(b: &mut Block, u: &[f64], positions: &[[f64; 3]], out: &mut [[f32; 24]; L]) {
    block_forces_impl(b, u, positions, out)
}

/// Compute a block's forces with the best instruction set available.
pub fn block_forces(b: &mut Block, u: &[f64], positions: &[[f64; 3]], out: &mut [[f32; 24]; L]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            // SAFETY: feature presence checked at runtime.
            unsafe { block_forces_avx2(b, u, positions, out) };
            return;
        }
    }
    block_forces_generic(b, u, positions, out)
}

pub fn is_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
    }
    #[allow(unreachable_code)]
    false
}

/// All SIMD blocks of a model plus their output buffers.
#[derive(Debug, Default)]
pub struct Blocks {
    pub blocks: Vec<Block>,
    out: Vec<[[f32; 24]; L]>,
}

impl Blocks {
    /// Build from the honeycomb elements (`simd_ids`) of `el`.
    pub fn build(el: &crate::kernel::Elements) -> Self {
        let mut blocks = Vec::new();
        let mut current = Block::empty();
        for &e in &el.simd_ids {
            if current.count == L {
                blocks.push(current);
                current = Block::empty();
            }
            let l = current.count;
            let state = match el.state[e] {
                crate::kernel::State::Rate(s) => s,
                _ => RateFoamState::default(),
            };
            let conn: [usize; 8] = std::array::from_fn(|i| el.conn[e][i] as usize);
            current.set_lane(l, conn, &el.center_grad[e], el.volume[e], &el.gamma[e], &el.h[e], &HoneycombParams::from_material(&el.material[e]), &state, el.eroded[e]);
        }
        if current.count > 0 {
            blocks.push(current);
        }
        let out = vec![[[0.0; 24]; L]; blocks.len()];
        Blocks { blocks, out }
    }

    /// Forces of all blocks (commits states); returns newly eroded count.
    pub fn compute_forces(&mut self, u: &[f64], positions: &[[f64; 3]], threads: usize) -> usize {
        use rayon::prelude::*;
        let before: usize = self.blocks.iter().map(|b| b.eroded[..b.count].iter().filter(|e| **e).count()).sum();
        if threads > 1 {
            self.blocks.par_iter_mut().zip(self.out.par_iter_mut()).with_min_len(4).for_each(|(b, out)| block_forces(b, u, positions, out));
        } else {
            for (b, out) in self.blocks.iter_mut().zip(self.out.iter_mut()) {
                block_forces(b, u, positions, out);
            }
        }
        let after: usize = self.blocks.iter().map(|b| b.eroded[..b.count].iter().filter(|e| **e).count()).sum();
        after - before
    }

    /// Scatter the last computed forces into the global vector.
    pub fn scatter(&self, global: &mut [f64]) {
        for (b, out) in self.blocks.iter().zip(&self.out) {
            for l in 0..b.count {
                let f = &out[l];
                for i in 0..8 {
                    let n = b.conn[l][i] as usize;
                    global[3 * n] += f[3 * i] as f64;
                    global[3 * n + 1] += f[3 * i + 1] as f64;
                    global[3 * n + 2] += f[3 * i + 2] as f64;
                }
            }
        }
    }

    /// Copy erosion flags into the element array (`ids` = simd element ids).
    pub fn mirror_erosion(&self, ids: &[usize], eroded: &mut [bool]) {
        let mut k = 0;
        for b in &self.blocks {
            for l in 0..b.count {
                eroded[ids[k]] = b.eroded[l];
                k += 1;
            }
        }
    }

    /// State of the k-th SIMD element.
    pub fn state(&self, k: usize) -> RateFoamState {
        self.blocks[k / L].lane_state(k % L)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::honeycomb_rate_update;
    use nalgebra::Vector3;

    /// A crushing element stepped 200 times: the lane kernel must track the
    /// scalar f64 reference to single precision (stress, compaction, forces).
    fn check_lane_vs_scalar(prm: &HoneycombParams) {
        let cube: Vec<[f64; 3]> = crate::element::XI.iter().map(|s| [0.15 * (s[0] + 1.0), 0.15 * (s[1] + 1.0), 0.15 * (s[2] + 1.0)]).collect();
        let mut grad = [0.0; 24];
        for i in 0..8 {
            for c in 0..3 {
                grad[3 * i + c] = crate::element::XI[i][c] / (4.0 * 0.3);
            }
        }
        let gamma = nalgebra::SMatrix::<f64, 8, 4>::zeros();
        let h = nalgebra::SMatrix::<f64, 12, 12>::zeros();
        let conn: [usize; 8] = std::array::from_fn(|i| i);
        let mut block = Block::empty();
        block.set_lane(0, conn, &grad, 0.027, &gamma, &h, prm, &RateFoamState::default(), false);
        let mut state = RateFoamState::default();
        let mut out = [[0.0f32; 24]; L];
        for step in 1..=200 {
            let c = 0.5 * step as f64 / 200.0;
            let mut u = vec![0.0; 24];
            for i in [1usize, 2, 5, 6] {
                u[3 * i] = -c * 0.3;
                u[3 * i + 1] = 0.02 * c * 0.3;
            }
            block_forces(&mut block, &u, &cube, &mut out);
            let mut f = Matrix3::<f64>::zeros();
            for i in 0..8 {
                let xi = Vector3::new(cube[i][0] + u[3 * i], cube[i][1] + u[3 * i + 1], cube[i][2] + u[3 * i + 2]);
                for a in 0..3 {
                    for b in 0..3 {
                        f[(a, b)] += xi[a] * grad[3 * i + b];
                    }
                }
            }
            let (tau, ns) = honeycomb_rate_update(prm, &f, &state);
            state = ns;
            let m = tau * f.try_inverse().unwrap().transpose() * 0.027;
            let lane = block.lane_state(0);
            let floor = 1e-6 * prm.youngs_modulus + 2e-3 * prm.yield_at(state.compaction).0;
            assert!((lane.compaction - state.compaction).abs() < 1e-3 * state.compaction.max(1e-6) + 1e-6, "step {}: compaction {} vs {}", step, lane.compaction, state.compaction);
            for k in 0..6 {
                assert!((lane.stress[k] - state.stress[k]).abs() < floor, "step {} stress[{}]: {} vs {}", step, k, lane.stress[k], state.stress[k]);
            }
            for i in 0..8 {
                let fi = m * Vector3::new(grad[3 * i], grad[3 * i + 1], grad[3 * i + 2]);
                for a in 0..3 {
                    assert!((out[0][3 * i + a] as f64 - fi[a]).abs() < floor * 0.027 / 0.3, "step {} force node {} comp {}: {} vs {}", step, i, a, out[0][3 * i + a], fi[a]);
                }
            }
        }
        assert!(state.compaction > 0.3);
    }

    #[test]
    fn lane_kernel_matches_scalar_reference() {
        check_lane_vs_scalar(&HoneycombParams::new(125.8e6, 1234.0, 800339.0));
    }

    #[test]
    fn lane_kernel_matches_scalar_with_tabulated_curve() {
        // Plateau, dip, steep rise, then flat: exercises every knot branch.
        let m = crate::material::Material::honeycomb_curve(23.1e6, 100.0, vec![[0.0, 7000.0], [0.05, 10000.0], [0.15, 6000.0], [0.3, 30000.0], [0.45, 30000.0]], Some([1.0, 2.0e6]));
        let prm = HoneycombParams::from_material(&m);
        let close = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).abs() < 1e-9 * b.0.abs().max(1.0) && (a.1 - b.1).abs() < 1e-9 * b.1.abs().max(1.0);
        assert!(close(prm.yield_at(0.0), (7000.0, 60000.0)));
        assert!(close(prm.yield_at(0.1), (8000.0, -40000.0)));
        assert!(close(prm.yield_at(0.6), (30000.0, 0.0)));
        assert!(close(prm.yield_at(1.5), (30000.0 + 1.0e6, 2.0e6)));
        assert!(close(m.plasticity.as_ref().unwrap().yield_at(1.5), prm.yield_at(1.5)));
        assert!(close(m.plasticity.as_ref().unwrap().yield_at(0.1), prm.yield_at(0.1)));
        check_lane_vs_scalar(&prm);
        let mut prm_t = prm;
        prm_t.cap_factor = [1.0, 4.0, 4.0];
        check_lane_vs_scalar(&prm_t);
    }
}

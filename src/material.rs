//! Materials and constitutive models.
//!
//! * [`Material`]: linear elastic properties plus optional plasticity.
//! * [`PlasticModel::Honeycomb`] (default for vehicles): crushable
//!   honeycomb in a corotational rate form — no eigen-solve, the model the
//!   SIMD kernel runs.
//! * [`PlasticModel::J2`] / [`PlasticModel::CrushableFoam`]: finite-strain
//!   Hencky models (multiplicative split, principal log strains). Reference
//!   models; they run through the scalar kernel.
//!
//! All stress updates return the Kirchhoff stress τ = J·σ.

use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};

pub type Voigt = nalgebra::SVector<f64, 6>;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum PlasticModel {
    /// von Mises: isochoric flow, hardening on equivalent plastic strain.
    /// For metals.
    J2,
    /// Isotropic crushable foam: each principal Kirchhoff stress capped at
    /// ±σ_y independently, plastic Poisson ratio zero, hardening on the
    /// accumulated compaction. Elastic Poisson ratio treated as 0.
    #[serde(alias = "crushable")]
    CrushableFoam,
    /// Crushable honeycomb in a corotational *rate* form (LS-DYNA MAT_26
    /// style): stress accumulated from the strain increment in a frame that
    /// rotates with the element (Hughes–Winget), normal stresses capped at
    /// ±σ_y in that frame (true stress), shears at σ_y/2, hardening on the
    /// accumulated compaction. No eigenvalue solve.
    #[default]
    Honeycomb,
}

/// Maximum knots of a tabulated yield curve.
pub const MAX_KNOTS: usize = 8;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Plasticity {
    /// Initial yield stress σ_y (Pa). Optional when `curve` is given.
    #[serde(default)]
    pub yield_stress: f64,
    /// Linear hardening modulus H (Pa): σ_y(ε̄ᵖ) = σ_y + H·ε̄ᵖ, ε̄ᵖ the
    /// equivalent plastic strain (J2) or compaction (foam, honeycomb).
    #[serde(default)]
    pub hardening: f64,
    #[serde(default)]
    pub model: PlasticModel,
    /// Tabulated yield stress vs compaction `[[c, σ_y], ...]` (honeycomb
    /// only; up to [`MAX_KNOTS`] knots, c ascending from 0). Piecewise
    /// linear, the last segment's slope continues beyond the table. When
    /// set it replaces `yield_stress` / `hardening`, which are kept equal
    /// to the first knot and first slope for reporting.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub curve: Vec<[f64; 2]>,
    /// Densification (lock-up) for the honeycomb model: `[c_lock, k_lock]`
    /// adds `k_lock·(c − c_lock)` to the yield stress beyond compaction
    /// `c_lock`, so a fully crushed element stiffens instead of inverting.
    /// Uses one of the [`MAX_KNOTS`] knots.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub densification: Option<[f64; 2]>,
    /// Honeycomb only: the caps on the two transverse normal stresses (local
    /// y, z of the corotational frame, which starts aligned with global) are
    /// `transverse_factor · σ_y`. 1 = isotropic (default); a crash rail
    /// structure is far stiffer sideways than along its crush axis.
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub transverse_factor: f64,
}

fn one() -> f64 {
    1.0
}
fn is_one(x: &f64) -> bool {
    *x == 1.0
}

impl Plasticity {
    /// Yield stress and local hardening slope at compaction `c`.
    #[inline]
    pub fn yield_at(&self, c: f64) -> (f64, f64) {
        let (mut sy, mut h) = if self.curve.len() < 2 {
            (self.yield_stress + self.hardening * c, self.hardening)
        } else {
            let k = &self.curve;
            let mut i = 0;
            while i + 2 < k.len() && c >= k[i + 1][0] {
                i += 1;
            }
            let h = (k[i + 1][1] - k[i][1]) / (k[i + 1][0] - k[i][0]);
            (k[i][1] + h * (c - k[i][0]), h)
        };
        if let Some([c_lock, k_lock]) = self.densification {
            if c > c_lock {
                sy += k_lock * (c - c_lock);
                h += k_lock;
            }
        }
        (sy, h)
    }

    /// Validate, and with a tabulated curve set `yield_stress` /
    /// `hardening` to its first knot and slope (for reporting).
    pub fn normalise(&mut self) -> Result<(), String> {
        self.check()?;
        if self.curve.len() >= 2 {
            self.yield_stress = self.curve[0][1];
            self.hardening = (self.curve[1][1] - self.curve[0][1]) / (self.curve[1][0] - self.curve[0][0]);
        }
        Ok(())
    }

    /// Validate a tabulated curve.
    pub fn check(&self) -> Result<(), String> {
        let k = &self.curve;
        let extra = usize::from(self.densification.is_some());
        if let Some([c_lock, k_lock]) = self.densification {
            if self.model != PlasticModel::Honeycomb {
                return Err("densification is only supported by the honeycomb model".into());
            }
            if c_lock <= 0.0 || k_lock < 0.0 {
                return Err("densification needs c_lock > 0 and k_lock >= 0".into());
            }
            if k.last().map_or(false, |p| p[0] >= c_lock) {
                return Err("densification c_lock must lie beyond the last curve knot".into());
            }
        }
        if k.is_empty() {
            return Ok(());
        }
        if self.model != PlasticModel::Honeycomb {
            return Err("tabulated yield curve is only supported by the honeycomb model".into());
        }
        if k.len() < 2 || k.len() + extra > MAX_KNOTS {
            return Err(format!("yield curve needs 2..={} knots (one fewer with densification)", MAX_KNOTS));
        }
        if k[0][0] != 0.0 {
            return Err("yield curve must start at compaction 0".into());
        }
        for w in k.windows(2) {
            if w[1][0] <= w[0][0] {
                return Err("yield curve compaction must be strictly increasing".into());
            }
        }
        if k.iter().any(|p| p[1] <= 0.0) {
            return Err("yield curve stresses must be positive".into());
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Material {
    pub youngs_modulus: f64,
    #[serde(default)]
    pub poisson_ratio: f64,
    pub density: f64,
    #[serde(default, flatten)]
    pub plasticity: Option<Plasticity>,
}

impl Material {
    pub fn elastic(youngs_modulus: f64, poisson_ratio: f64, density: f64) -> Self {
        Material { youngs_modulus, poisson_ratio, density, plasticity: None }
    }

    pub fn j2(youngs_modulus: f64, poisson_ratio: f64, density: f64, yield_stress: f64, hardening: f64) -> Self {
        Material { youngs_modulus, poisson_ratio, density, plasticity: Some(Plasticity { yield_stress, hardening, model: PlasticModel::J2, curve: Vec::new(), densification: None, transverse_factor: 1.0 }) }
    }

    /// Isotropic crushable foam (elastic Poisson ratio treated as 0).
    pub fn crushable_foam(youngs_modulus: f64, density: f64, yield_stress: f64, hardening: f64) -> Self {
        Material { youngs_modulus, poisson_ratio: 0.0, density, plasticity: Some(Plasticity { yield_stress, hardening, model: PlasticModel::CrushableFoam, curve: Vec::new(), densification: None, transverse_factor: 1.0 }) }
    }

    /// Crushable honeycomb, corotational rate form (the SIMD kernel model).
    pub fn honeycomb(youngs_modulus: f64, density: f64, yield_stress: f64, hardening: f64) -> Self {
        Material { youngs_modulus, poisson_ratio: 0.0, density, plasticity: Some(Plasticity { yield_stress, hardening, model: PlasticModel::Honeycomb, curve: Vec::new(), densification: None, transverse_factor: 1.0 }) }
    }

    /// Crushable honeycomb with a tabulated yield stress vs compaction
    /// curve `[[c, σ_y], ...]` (see [`Plasticity::curve`]) and optional
    /// densification `[c_lock, k_lock]` (see [`Plasticity::densification`]).
    pub fn honeycomb_curve(youngs_modulus: f64, density: f64, curve: Vec<[f64; 2]>, densification: Option<[f64; 2]>) -> Self {
        let p = Plasticity { yield_stress: curve[0][1], hardening: (curve[1][1] - curve[0][1]) / (curve[1][0] - curve[0][0]), model: PlasticModel::Honeycomb, curve, densification, transverse_factor: 1.0 };
        p.check().unwrap_or_else(|e| panic!("{}", e));
        Material { youngs_modulus, poisson_ratio: 0.0, density, plasticity: Some(p) }
    }

    pub fn is_plastic(&self) -> bool {
        self.plasticity.is_some()
    }

    /// The same material with every stress-like quantity (modulus, yield
    /// stress, hardening, curve stresses, lock-up slope) and the density
    /// multiplied by `factor`: a region carrying `factor` times the load
    /// per unit area with the same wave speed and strain history.
    pub fn scaled(&self, factor: f64) -> Self {
        let mut m = self.clone();
        m.youngs_modulus *= factor;
        m.density *= factor;
        if let Some(p) = m.plasticity.as_mut() {
            p.yield_stress *= factor;
            p.hardening *= factor;
            for k in &mut p.curve {
                k[1] *= factor;
            }
            if let Some(d) = p.densification.as_mut() {
                d[1] *= factor;
            }
        }
        m
    }

    pub fn model(&self) -> Option<PlasticModel> {
        self.plasticity.as_ref().map(|p| p.model)
    }

    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Dilatational wave speed sqrt((λ+2μ)/ρ), which bounds the explicit step.
    pub fn dilatational_wave_speed(&self) -> f64 {
        let e = self.youngs_modulus;
        let v = self.poisson_ratio;
        let m = e * (1.0 - v) / ((1.0 + v) * (1.0 - 2.0 * v));
        (m / self.density).sqrt()
    }

    /// Small-strain isotropic elastic matrix (Voigt 6×6, engineering shear).
    pub fn elastic_matrix(&self) -> nalgebra::Matrix6<f64> {
        elastic_matrix(self.youngs_modulus, self.poisson_ratio)
    }
}


/// Plastic history at one integration point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlasticPoint {
    /// Plastic strain εᵖ (Voigt, engineering shear).
    pub plastic_strain: Voigt,
    /// Equivalent plastic strain ε̄ᵖ.
    pub eq_plastic_strain: f64,
}

impl Default for PlasticPoint {
    fn default() -> Self {
        PlasticPoint {
            plastic_strain: Voigt::zeros(),
            eq_plastic_strain: 0.0,
        }
    }
}

/// Norm of the deviatoric part of a Voigt *stress* vector:
/// `‖s‖ = sqrt(s₀²+s₁²+s₂² + 2(s₃²+s₄²+s₅²))`.
fn tensor_norm(s: &Voigt) -> f64 {
    (s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
        .sqrt()
}

/// Radial return for one integration point.
///
/// Given the total strain and the committed state, returns the stress and the
/// updated state. If the material has no plasticity, this is plain `C:ε`.
pub fn radial_return(material: &Material, strain: &Voigt, state: &PlasticPoint) -> (Voigt, PlasticPoint) {
    let c = &material.elastic_matrix();
    let trial = c * (strain - state.plastic_strain);
    let Some(p) = material.plasticity.as_ref() else {
        return (trial, *state);
    };

    let mean = (trial[0] + trial[1] + trial[2]) / 3.0;
    let mut dev = trial;
    dev[0] -= mean;
    dev[1] -= mean;
    dev[2] -= mean;
    let norm = tensor_norm(&dev);
    let q = (1.5_f64).sqrt() * norm; // von Mises stress
    let sigma_y = p.yield_stress + p.hardening * state.eq_plastic_strain;
    let f = q - sigma_y;
    if f <= 0.0 || norm < 1e-300 {
        return (trial, *state);
    }

    let g = material.shear_modulus();
    let d_eq = f / (3.0 * g + p.hardening); // Δε̄ᵖ
    let n = dev / norm; // unit flow direction (tensor components)

    // Scale the deviator back onto the yield surface.
    let scale = 1.0 - 3.0 * g * d_eq / q;
    let mut stress = dev * scale;
    stress[0] += mean;
    stress[1] += mean;
    stress[2] += mean;

    // Δεᵖ = sqrt(3/2)·Δε̄ᵖ·n, with engineering shear (×2) on the last three.
    let k = (1.5_f64).sqrt() * d_eq;
    let mut plastic_strain = state.plastic_strain;
    for i in 0..6 {
        let voigt = if i < 3 { 1.0 } else { 2.0 };
        plastic_strain[i] += voigt * k * n[i];
    }

    (
        stress,
        PlasticPoint {
            plastic_strain,
            eq_plastic_strain: state.eq_plastic_strain + d_eq,
        },
    )
}

/// Crushable-foam branch of [`finite_strain_return`], in the principal
/// basis of the trial elastic left Cauchy–Green tensor. Uncoupled principal
/// directions (ν = 0): `τ_a = E·ε_a`, each capped at ±σ_y(ε̄ᵖ), with the
/// excess strain becoming plastic strain in that direction. ε̄ᵖ accumulates
/// the magnitude of the principal plastic increments (compaction under
/// crushing), which drives the hardening.
fn crushable_foam_return(
    e: f64,
    p: &Plasticity,
    f: &Matrix3<f64>,
    eps_trial: &[f64; 3],
    eigenvectors: &Matrix3<f64>,
    state: &FinitePlasticPoint,
) -> (Matrix3<f64>, FinitePlasticPoint) {
    // Implicit hardening: the cap depends on the compaction accumulated in
    // this step, which couples the three directions through ε̄ᵖ. The fixed
    // point converges geometrically with ratio ≤ 3H/(E+H).
    let mut eps_e = *eps_trial;
    let mut d_eq = 0.0;
    for _ in 0..50 {
        let sigma_y = p.yield_stress + p.hardening * (state.eq_plastic_strain + d_eq);
        let eps_y = sigma_y / e;
        let mut d_new = 0.0;
        for a in 0..3 {
            if eps_trial[a] < -eps_y {
                d_new += -eps_y - eps_trial[a];
                eps_e[a] = -eps_y;
            } else if eps_trial[a] > eps_y {
                d_new += eps_trial[a] - eps_y;
                eps_e[a] = eps_y;
            } else {
                eps_e[a] = eps_trial[a];
            }
        }
        let converged = (d_new - d_eq).abs() <= 1e-12 * d_new.max(1e-12);
        d_eq = d_new;
        if converged {
            break;
        }
    }
    let mut tau = Matrix3::<f64>::zeros();
    let mut be_new = Matrix3::<f64>::zeros();
    for a in 0..3 {
        let n = eigenvectors.column(a);
        let nn = n * n.transpose();
        tau += e * eps_e[a] * nn;
        be_new += (2.0 * eps_e[a]).exp() * nn;
    }
    if d_eq > 0.0 {
        let f_inv = f.try_inverse().unwrap_or_else(Matrix3::identity);
        let cp_inv = f_inv * be_new * f_inv.transpose();
        (
            tau,
            FinitePlasticPoint {
                cp_inv: 0.5 * (cp_inv + cp_inv.transpose()),
                eq_plastic_strain: state.eq_plastic_strain + d_eq,
            },
        )
    } else {
        (tau, *state)
    }
}

/// State of the corotational rate-form honeycomb model at one integration
/// point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateFoamState {
    /// Inverse of the deformation gradient at the last committed step.
    pub f_prev_inv: Matrix3<f64>,
    /// Corotational frame (element rotation) R.
    pub rotation: Matrix3<f64>,
    /// Cauchy stress in the corotational frame, Voigt `[xx, yy, zz, xy, yz, xz]`.
    pub stress: [f64; 6],
    /// Accumulated compaction (plastic) strain driving the hardening.
    pub compaction: f64,
}

impl Default for RateFoamState {
    fn default() -> Self {
        RateFoamState { f_prev_inv: Matrix3::identity(), rotation: Matrix3::identity(), stress: [0.0; 6], compaction: 0.0 }
    }
}

/// Corotational rate update of the honeycomb model (see
/// [`PlasticModel::Honeycomb`]).
///
/// From the incremental deformation gradient `F_inc = F·F_prev⁻¹` the
/// midpoint strain increment `Δε = sym(G)` and spin `Ω = skew(G)`,
/// `G = (F_inc − I)(½(F_inc + I))⁻¹`, are formed; the frame is rotated by
/// the Cayley transform `ΔR = (I − ½Ω)⁻¹(I + ½Ω)` (exactly orthogonal);
/// the stress in the frame gets the elastic increment `E·Δε̂` (ν = 0, shear
/// modulus E/2 on engineering shear), the normal components are capped at
/// ±σ_y(c) with the compaction `c` advanced implicitly, and shears at
/// σ_y(c)/2. Returns the Kirchhoff stress τ = J·R σ̂ Rᵀ and the new state.
/// A repeated call at the same `F` is a no-op (F_inc = I).
pub fn honeycomb_rate_return(material: &Material, f: &Matrix3<f64>, state: &RateFoamState) -> (Matrix3<f64>, RateFoamState) {
    honeycomb_rate_update(&HoneycombParams::from_material(material), f, state)
}

/// Compact parameters of the honeycomb model, for hot loops: the yield
/// curve as `knots` (compaction, stress, slope of the following segment),
/// with unused knots at +∞ compaction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoneycombParams {
    pub youngs_modulus: f64,
    pub yield_stress: f64,
    pub hardening: f64,
    pub knots: [[f64; 3]; MAX_KNOTS],
    /// Cap factor per normal component (local x, y, z).
    pub cap_factor: [f64; 3],
}

impl HoneycombParams {
    pub fn from_material(material: &Material) -> Self {
        let p = material.plasticity.as_ref().expect("honeycomb material");
        p.check().unwrap_or_else(|e| panic!("{}", e));
        let mut knots = [[f64::INFINITY, 0.0, 0.0]; MAX_KNOTS];
        let n = if p.curve.len() < 2 {
            knots[0] = [0.0, p.yield_stress, p.hardening];
            1
        } else {
            for (i, k) in p.curve.iter().enumerate() {
                let h = if i + 1 < p.curve.len() { (p.curve[i + 1][1] - k[1]) / (p.curve[i + 1][0] - k[0]) } else { knots[i - 1][2] };
                knots[i] = [k[0], k[1], h];
            }
            p.curve.len()
        };
        if let Some([c_lock, k_lock]) = p.densification {
            let last = knots[n - 1];
            knots[n] = [c_lock, last[1] + last[2] * (c_lock - last[0]), last[2] + k_lock];
        }
        HoneycombParams { youngs_modulus: material.youngs_modulus, yield_stress: p.yield_stress, hardening: p.hardening, knots, cap_factor: [1.0, p.transverse_factor, p.transverse_factor] }
    }

    pub fn new(youngs_modulus: f64, yield_stress: f64, hardening: f64) -> Self {
        let mut knots = [[f64::INFINITY, 0.0, 0.0]; MAX_KNOTS];
        knots[0] = [0.0, yield_stress, hardening];
        HoneycombParams { youngs_modulus, yield_stress, hardening, knots, cap_factor: [1.0; 3] }
    }

    /// Yield stress and local slope at compaction `c` (branch-free scan).
    #[inline]
    pub fn yield_at(&self, c: f64) -> (f64, f64) {
        let mut k = self.knots[0];
        for i in 1..MAX_KNOTS {
            if c >= self.knots[i][0] {
                k = self.knots[i];
            }
        }
        (k[1] + k[2] * (c - k[0]), k[2])
    }
}

/// [`honeycomb_rate_return`] on compact parameters.
#[inline]
pub fn honeycomb_rate_update(prm: &HoneycombParams, f: &Matrix3<f64>, state: &RateFoamState) -> (Matrix3<f64>, RateFoamState) {
    let e = prm.youngs_modulus;
    let p = prm;
    let i3 = Matrix3::<f64>::identity();
    let f_inc = f * state.f_prev_inv;
    let mid_inv = (0.5 * (f_inc + i3)).try_inverse().unwrap_or(i3);
    let g = (f_inc - i3) * mid_inv;
    let d = 0.5 * (g + g.transpose());
    let w = 0.5 * (g - g.transpose());
    // Cayley transform of a skew matrix in closed form:
    // (I − ½W)⁻¹(I + ½W) = I + (W + ½W²) / (1 + ¼|ω|²), ω the axial vector.
    let omega2 = w[(0, 1)] * w[(0, 1)] + w[(0, 2)] * w[(0, 2)] + w[(1, 2)] * w[(1, 2)];
    let delta_r = i3 + (w + 0.5 * w * w) / (1.0 + 0.25 * omega2);
    let r = delta_r * state.rotation;
    // Strain increment in the corotational frame.
    let de = r.transpose() * d * r;
    let mut s = state.stress;
    s[0] += e * de[(0, 0)];
    s[1] += e * de[(1, 1)];
    s[2] += e * de[(2, 2)];
    s[3] += e * de[(0, 1)]; // τ = 2G ε_ij = E ε_ij for ν = 0
    s[4] += e * de[(1, 2)];
    s[5] += e * de[(0, 2)];

    // Cap the normal stresses (component k at t_k·σ_y) with implicit
    // hardening: for the active set A,
    // Δc·E = Σ_A (−σ_k − t_k(σ_yc + H Δc))  ⇒  Δc = Σ_A(−σ_k − t_k σ_yc) / (E + H Σ_A t_k).
    let (sigma_yc, hard) = p.yield_at(state.compaction);
    let t = p.cap_factor;
    let mut dc = 0.0;
    for _ in 0..3 {
        let sigma_y = sigma_yc + hard * dc;
        let (mut sum, mut t_active) = (0.0, 0.0);
        for k in 0..3 {
            // Only compression compacts (and hardens); tension is just
            // capped below.
            if -s[k] > t[k] * sigma_y {
                sum += -s[k] - t[k] * sigma_yc;
                t_active += t[k];
            }
        }
        let dc_new = if t_active == 0.0 { 0.0 } else { sum / (e + t_active * hard) };
        if (dc_new - dc).abs() <= 1e-14 * dc_new.max(1e-300) {
            dc = dc_new;
            break;
        }
        dc = dc_new;
    }
    let sigma_y = (sigma_yc + hard * dc).max(0.0);
    for k in 0..3 {
        s[k] = s[k].clamp(-t[k] * sigma_y, t[k] * sigma_y);
    }
    let tau_y = 0.5 * sigma_y;
    for k in 3..6 {
        s[k] = s[k].clamp(-tau_y, tau_y);
    }

    let sigma_hat = Matrix3::new(s[0], s[3], s[5], s[3], s[1], s[4], s[5], s[4], s[2]);
    let tau = f.determinant() * (r * sigma_hat * r.transpose());
    (
        tau,
        RateFoamState { f_prev_inv: f.try_inverse().unwrap_or(i3), rotation: r, stress: s, compaction: state.compaction + dc },
    )
}

/// Voigt stress → symmetric 3×3 tensor.
pub fn voigt_to_tensor(s: &Voigt) -> Matrix3<f64> {
    Matrix3::new(s[0], s[3], s[5], s[3], s[1], s[4], s[5], s[4], s[2])
}

/// Symmetric 3×3 stress tensor → Voigt.
pub fn tensor_to_voigt(t: &Matrix3<f64>) -> Voigt {
    Voigt::from([t[(0, 0)], t[(1, 1)], t[(2, 2)], t[(0, 1)], t[(1, 2)], t[(0, 2)]])
}

/// Rotation part of the polar decomposition F = R·U (proper rotation, det = +1).
pub fn polar_rotation(f: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = f.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    let mut r = u * v_t;
    if r.determinant() < 0.0 {
        let mut u_fixed = u;
        u_fixed.column_mut(2).neg_mut();
        r = u_fixed * v_t;
    }
    r
}

/// Eigen-decomposition of a symmetric 3×3 matrix: (eigenvalues ascending,
/// orthonormal eigenvectors as columns).
///
/// Closed-form, non-iterative: trigonometric eigenvalues (as in D. Eberly,
/// "A Robust Eigensolver for 3×3 Symmetric Matrices", 2014), the eigenvector
/// of the well-separated eigenvalue from row cross products, the other two
/// from an exact Jacobi rotation in the orthogonal plane, and eigenvalues
/// refined as Rayleigh quotients. Accurate for repeated or nearly repeated
/// eigenvalues; about 2× faster than `nalgebra`'s iterative `symmetric_eigen`.
pub fn symmetric_eigen3(a: &Matrix3<f64>) -> ([f64; 3], Matrix3<f64>) {
    use nalgebra::Vector3;
    let (a00, a01, a02, a11, a12, a22) = (a[(0, 0)], a[(0, 1)], a[(0, 2)], a[(1, 1)], a[(1, 2)], a[(2, 2)]);
    let max_abs = a00.abs().max(a01.abs()).max(a02.abs()).max(a11.abs()).max(a12.abs()).max(a22.abs());
    if max_abs == 0.0 {
        return ([0.0; 3], Matrix3::identity());
    }
    let inv = 1.0 / max_abs;
    let (a00, a01, a02, a11, a12, a22) = (a00 * inv, a01 * inv, a02 * inv, a11 * inv, a12 * inv, a22 * inv);
    let norm = a01 * a01 + a02 * a02 + a12 * a12;
    if norm == 0.0 {
        // Already diagonal: sort ascending.
        let mut pairs = [(a00, 0usize), (a11, 1), (a22, 2)];
        pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        let mut v = Matrix3::zeros();
        for (c, (_, axis)) in pairs.iter().enumerate() {
            v[(*axis, c)] = 1.0;
        }
        return ([pairs[0].0 * max_abs, pairs[1].0 * max_abs, pairs[2].0 * max_abs], v);
    }

    let q = (a00 + a11 + a22) / 3.0;
    let (b00, b11, b22) = (a00 - q, a11 - q, a22 - q);
    let p = ((b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * norm) / 6.0).sqrt();
    let c00 = b11 * b22 - a12 * a12;
    let c01 = a01 * b22 - a12 * a02;
    let c02 = a01 * a12 - b11 * a02;
    let det = (b00 * c00 - a01 * c01 + a02 * c02) / (p * p * p);
    let half_det = (0.5 * det).clamp(-1.0, 1.0);
    let angle = half_det.acos() / 3.0;
    const TWO_THIRDS_PI: f64 = 2.094_395_102_393_195_5;
    let beta2 = 2.0 * angle.cos();
    let beta0 = 2.0 * (angle + TWO_THIRDS_PI).cos();
    let beta1 = -(beta0 + beta2);
    let eval = [q + p * beta0, q + p * beta1, q + p * beta2];

    let rows = |e: f64| {
        (
            Vector3::new(a00 - e, a01, a02),
            Vector3::new(a01, a11 - e, a12),
            Vector3::new(a02, a12, a22 - e),
        )
    };
    // Eigenvector of the well-separated eigenvalue via the largest row cross product.
    let vector0 = |e: f64| {
        let (r0, r1, r2) = rows(e);
        let (c01, c02, c12) = (r0.cross(&r1), r0.cross(&r2), r1.cross(&r2));
        let (d0, d1, d2) = (c01.norm_squared(), c02.norm_squared(), c12.norm_squared());
        if d0 >= d1 && d0 >= d2 {
            c01 / d0.sqrt()
        } else if d1 >= d2 {
            c02 / d1.sqrt()
        } else {
            c12 / d2.sqrt()
        }
    };
    let am = Matrix3::new(a00, a01, a02, a01, a11, a12, a02, a12, a22);
    // The other two eigenvectors span the plane orthogonal to `w`: diagonalise
    // A restricted to that plane with one exact Jacobi rotation. This does not
    // use the (less accurate) trigonometric eigenvalues, so it stays stable
    // however close the remaining two eigenvalues are.
    let plane_pair = |w: &Vector3<f64>| {
        let u = if w.x.abs() > w.y.abs() {
            Vector3::new(-w.z, 0.0, w.x) / (w.x * w.x + w.z * w.z).sqrt()
        } else {
            Vector3::new(0.0, w.z, -w.y) / (w.y * w.y + w.z * w.z).sqrt()
        };
        let v = w.cross(&u);
        let (au, av) = (am * u, am * v);
        let (m00, m01, m11) = (u.dot(&au), u.dot(&av), v.dot(&av));
        let theta = 0.5 * (2.0 * m01).atan2(m00 - m11);
        let (sn, cs) = theta.sin_cos();
        (u * cs + v * sn, v * cs - u * sn)
    };

    // Start from the eigenvalue that is well separated from the other two.
    let w = if half_det >= 0.0 { vector0(eval[2]) } else { vector0(eval[0]) };
    let (p1, p2) = plane_pair(&w);
    let (v0, v1, v2) = (w, p1, p2);

    // Near a repeated root the trigonometric eigenvalues lose about half their
    // digits while the eigenvectors stay accurate, so recompute each
    // eigenvalue as a Rayleigh quotient vᵀAv (error quadratic in the vector error).
    let mut pairs = [(v0.dot(&(am * v0)), v0), (v1.dot(&(am * v1)), v1), (v2.dot(&(am * v2)), v2)];
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    (
        [pairs[0].0 * max_abs, pairs[1].0 * max_abs, pairs[2].0 * max_abs],
        Matrix3::from_columns(&[pairs[0].1, pairs[1].1, pairs[2].1]),
    )
}

/// Isotropic small-strain elastic matrix (Voigt 6×6, engineering shear).
pub fn elastic_matrix(e: f64, nu: f64) -> nalgebra::Matrix6<f64> {
    let lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    let mut c = nalgebra::Matrix6::<f64>::zeros();
    for i in 0..3 {
        for j in 0..3 {
            c[(i, j)] = if i == j { lam + 2.0 * mu } else { lam };
        }
        c[(i + 3, i + 3)] = mu;
    }
    c
}


/// von Mises equivalent stress of a Voigt stress.
pub fn von_mises(s: &Voigt) -> f64 {
    (0.5 * ((s[0] - s[1]).powi(2) + (s[1] - s[2]).powi(2) + (s[2] - s[0]).powi(2)) + 3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5])).sqrt()
}

/// Plastic history for the finite-strain model at one integration point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FinitePlasticPoint {
    /// Inverse plastic right Cauchy–Green tensor C_p⁻¹ (identity when virgin).
    pub cp_inv: Matrix3<f64>,
    /// Equivalent plastic strain ε̄ᵖ.
    pub eq_plastic_strain: f64,
}

impl Default for FinitePlasticPoint {
    fn default() -> Self {
        FinitePlasticPoint {
            cp_inv: Matrix3::identity(),
            eq_plastic_strain: 0.0,
        }
    }
}

/// Finite-strain J2 plasticity: multiplicative split F = Fᵉ·Fᵖ, Hencky
/// (logarithmic) elasticity and radial return in principal log-strain space,
/// with the exponential-map update of Simo (1992) / Simo & Hughes (1998,
/// Box 9.1). Exactly objective (no corotational frame needed), exact for
/// large rotations, and the pressure K·ln J grows without bound as the volume
/// goes to zero, so elements cannot collapse under crushing.
///
/// Returns the Kirchhoff stress τ = J·σ and the updated state. With no
/// plasticity on the material it is hyperelastic Hencky.
pub fn finite_strain_return(
    material: &Material,
    f: &Matrix3<f64>,
    state: &FinitePlasticPoint,
) -> (Matrix3<f64>, FinitePlasticPoint) {
    let e = material.youngs_modulus;
    let nu = material.poisson_ratio;
    let g = e / (2.0 * (1.0 + nu));
    let k = e / (3.0 * (1.0 - 2.0 * nu));

    // Trial elastic left Cauchy–Green tensor and its spectral decomposition.
    let be_trial = f * state.cp_inv * f.transpose();
    let be_trial = 0.5 * (be_trial + be_trial.transpose());
    let (eigenvalues, eigenvectors) = symmetric_eigen3(&be_trial);
    let mut eps = [0.0; 3]; // principal elastic log strains
    for a in 0..3 {
        eps[a] = 0.5 * eigenvalues[a].max(1e-300).ln();
    }

    if let Some(p) = material.plasticity.as_ref().filter(|p| p.model == PlasticModel::CrushableFoam) {
        return crushable_foam_return(e, p, f, &eps, &eigenvectors, state);
    }

    let vol = eps[0] + eps[1] + eps[2];
    let mut dev = [eps[0] - vol / 3.0, eps[1] - vol / 3.0, eps[2] - vol / 3.0];
    let pressure = k * vol;
    let mut eq = state.eq_plastic_strain;

    if let Some(p) = material.plasticity.as_ref() {
        let dev_norm = (dev[0] * dev[0] + dev[1] * dev[1] + dev[2] * dev[2]).sqrt();
        let q = (1.5_f64).sqrt() * 2.0 * g * dev_norm; // von Mises of trial τ
        let fy = q - (p.yield_stress + p.hardening * eq);
        if fy > 0.0 && dev_norm > 0.0 {
            let d_eq = fy / (3.0 * g + p.hardening);
            let scale = 1.0 - 3.0 * g * d_eq / q;
            for a in 0..3 {
                dev[a] *= scale;
            }
            eq += d_eq;
        }
    }

    // Kirchhoff stress and updated elastic strain, in the trial eigenbasis.
    let mut tau = Matrix3::<f64>::zeros();
    let mut be_new = Matrix3::<f64>::zeros();
    for a in 0..3 {
        let n = eigenvectors.column(a);
        let nn = n * n.transpose();
        tau += (pressure + 2.0 * g * dev[a]) * nn;
        be_new += (2.0 * (dev[a] + vol / 3.0)).exp() * nn;
    }

    let new_state = if eq > state.eq_plastic_strain {
        let f_inv = f.try_inverse().unwrap_or_else(Matrix3::identity);
        let cp_inv = f_inv * be_new * f_inv.transpose();
        FinitePlasticPoint {
            cp_inv: 0.5 * (cp_inv + cp_inv.transpose()),
            eq_plastic_strain: eq,
        }
    } else {
        *state
    };
    (tau, new_state)
}

#[cfg(test)]
mod tests {
    use super::*;
    

    fn steel(h: f64) -> Material {
        Material::j2(200e9, 0.3, 7800.0, 250e6, h)
    }


    #[test]
    fn elastic_below_yield() {
        let m = steel(0.0);
        let e = Voigt::from([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let (s, st) = radial_return(&m, &e, &PlasticPoint::default());
        assert_eq!(st.eq_plastic_strain, 0.0);
        assert!((s - &m.elastic_matrix() * e).norm() < 1e-6);
    }

    #[test]
    fn returns_onto_yield_surface() {
        for h in [0.0, 2e9] {
            let m = steel(h);
            let e = Voigt::from([5e-3, -1e-3, -2e-3, 3e-3, 1e-3, -2e-3]);
            let (s, st) = radial_return(&m, &e, &PlasticPoint::default());
            let radius = 250e6 + h * st.eq_plastic_strain;
            assert!(st.eq_plastic_strain > 0.0);
            assert!((von_mises(&s) - radius).abs() / radius < 1e-9, "vm {} vs {}", von_mises(&s), radius);
            // Plastic flow is isochoric.
            let tr = st.plastic_strain[0] + st.plastic_strain[1] + st.plastic_strain[2];
            assert!(tr.abs() < 1e-15);
            // Stress is consistent with the updated plastic strain.
            let s2 = &m.elastic_matrix() * (e - st.plastic_strain);
            assert!((s2 - s).norm() / s.norm() < 1e-9);
        }
    }

    #[test]
    fn repeated_call_is_idempotent() {
        // The explicit solver may evaluate forces twice at one configuration.
        let m = steel(1e9);
        let e = Voigt::from([4e-3, -2e-3, -2e-3, 0.0, 0.0, 0.0]);
        let (s1, st1) = radial_return(&m, &e, &PlasticPoint::default());
        let (s2, st2) = radial_return(&m, &e, &st1);
        assert!((s1 - s2).norm() / s1.norm() < 1e-9);
        assert!((st1.eq_plastic_strain - st2.eq_plastic_strain).abs() < 1e-15);
    }

    #[test]
    fn unloading_is_elastic() {
        let m = steel(0.0);
        let e = Voigt::from([4e-3, -2e-3, -2e-3, 0.0, 0.0, 0.0]);
        let (_, st) = radial_return(&m, &e, &PlasticPoint::default());
        let e2 = e * 0.9;
        let (_, st2) = radial_return(&m, &e2, &st);
        assert_eq!(st.eq_plastic_strain, st2.eq_plastic_strain);
    }

    #[test]
    fn polar_rotation_recovers_rotation() {
        let r = nalgebra::Rotation3::from_euler_angles(0.3, -0.7, 1.1).into_inner();
        let u = Matrix3::new(1.2, 0.1, 0.0, 0.1, 0.9, 0.05, 0.0, 0.05, 1.1);
        let rr = polar_rotation(&(r * u));
        assert!((rr - r).norm() < 1e-10);
    }

    #[test]
    fn finite_strain_matches_small_strain_for_small_strain() {
        let m = steel(1e9);
        let h = Matrix3::new(2e-3, 3e-4, 0.0, 3e-4, -1e-3, 1e-4, 0.0, 1e-4, -5e-4);
        let (tau, st) = finite_strain_return(&m, &(Matrix3::identity() + h), &FinitePlasticPoint::default());
        let eps = Voigt::from([h[(0, 0)], h[(1, 1)], h[(2, 2)], 2.0 * h[(0, 1)], 2.0 * h[(1, 2)], 0.0]);
        let (s_small, st_small) = radial_return(&m, &eps, &PlasticPoint::default());
        assert!(st.eq_plastic_strain > 0.0 && st_small.eq_plastic_strain > 0.0);
        let rel = (tensor_to_voigt(&tau) - s_small).norm() / s_small.norm();
        assert!(rel < 5e-3, "relative difference {}", rel);
    }

    #[test]
    fn finite_strain_is_objective() {
        let m = steel(1e9);
        let f = Matrix3::new(0.7, 0.2, 0.0, 0.1, 1.2, 0.05, 0.0, 0.0, 1.15);
        let r = nalgebra::Rotation3::from_euler_angles(0.4, -1.1, 2.0).into_inner();
        let (t1, s1) = finite_strain_return(&m, &f, &FinitePlasticPoint::default());
        let (t2, s2) = finite_strain_return(&m, &(r * f), &FinitePlasticPoint::default());
        assert!((r * t1 * r.transpose() - t2).norm() / t1.norm() < 1e-9);
        assert!((s1.eq_plastic_strain - s2.eq_plastic_strain).abs() < 1e-12);
    }

    #[test]
    fn finite_strain_stays_on_yield_surface_and_is_idempotent() {
        let m = steel(2e9);
        let f = Matrix3::new(0.5, 0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 1.4); // 50% crush
        let (tau, st) = finite_strain_return(&m, &f, &FinitePlasticPoint::default());
        let vm = von_mises(&tensor_to_voigt(&tau));
        let radius = 250e6 + 2e9 * st.eq_plastic_strain;
        assert!((vm - radius).abs() / radius < 1e-9);
        let (tau2, st2) = finite_strain_return(&m, &f, &st);
        assert!((tau - tau2).norm() / tau.norm() < 1e-9);
        assert!((st.eq_plastic_strain - st2.eq_plastic_strain).abs() < 1e-12);
        // Plastic flow is isochoric: det(C_p⁻¹) stays 1.
        assert!((st.cp_inv.determinant() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pressure_resists_volume_collapse() {
        let m = Material::j2(20e6, 0.3, 200.0, 0.2e6, 0.0);
        let k = 20e6 / (3.0 * 0.4);
        let (tau, _) = finite_strain_return(&m, &(Matrix3::identity() * 0.5_f64.cbrt()), &FinitePlasticPoint::default());
        // Pure volume change: p = K ln J, unbounded as J → 0.
        assert!((tau.trace() / 3.0 - k * 0.5_f64.ln()).abs() < 1e-6 * k);
    }

    fn check_eigen(a: &Matrix3<f64>) {
        let (vals, vecs) = symmetric_eigen3(a);
        let scale = a.norm().max(1e-300);
        let recon = vecs * Matrix3::from_diagonal(&nalgebra::Vector3::from(vals)) * vecs.transpose();
        assert!((recon - a).norm() <= 1e-12 * scale, "reconstruction error {} for {}", (recon - a).norm(), a);
        assert!((vecs.transpose() * vecs - Matrix3::identity()).norm() < 1e-12, "not orthonormal: {}", vecs);
        assert!(vals[0] <= vals[1] && vals[1] <= vals[2]);
        let mut reference: Vec<f64> = a.symmetric_eigen().eigenvalues.iter().copied().collect();
        reference.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for k in 0..3 {
            assert!((vals[k] - reference[k]).abs() <= 1e-12 * scale);
        }
    }

    #[test]
    fn closed_form_eigen_matches_reference() {
        // Deterministic pseudo-random symmetric matrices over many scales.
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        for k in 0..20000 {
            let s = 10f64.powi((k % 13) as i32 - 6);
            let m = Matrix3::new(rnd(), rnd(), rnd(), rnd(), rnd(), rnd(), rnd(), rnd(), rnd());
            check_eigen(&((m + m.transpose()) * s));
        }
        // Near-identity (virgin elastic state) and repeated eigenvalues.
        for k in 0..2000 {
            let e = 10f64.powi(-(k % 12) as i32);
            let m = Matrix3::new(rnd(), rnd(), rnd(), rnd(), rnd(), rnd(), rnd(), rnd(), rnd());
            check_eigen(&(Matrix3::identity() + (m + m.transpose()) * e));
            let r = nalgebra::Rotation3::from_euler_angles(rnd(), rnd(), rnd()).into_inner();
            check_eigen(&(r * Matrix3::from_diagonal(&nalgebra::Vector3::new(2.0, 2.0 + e * rnd(), 0.5)) * r.transpose()));
            check_eigen(&(r * Matrix3::from_diagonal(&nalgebra::Vector3::new(0.3, 1.0, 1.0)) * r.transpose()));
        }
        check_eigen(&Matrix3::identity());
        check_eigen(&Matrix3::zeros());
        check_eigen(&Matrix3::from_diagonal(&nalgebra::Vector3::new(3.0, -1.0, 2.0)));
    }

    #[test]
    fn crushable_foam_caps_stress_without_bulging() {
        let m = Material::crushable_foam(20e6, 200.0, 0.2e6, 0.5e6);
        // Uniaxial crush to 60 % of length, no lateral strain imposed.
        let f = Matrix3::new(0.4, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let (tau, st) = finite_strain_return(&m, &f, &FinitePlasticPoint::default());
        // Axial stress on the (hardened) cap, lateral stresses zero.
        let sigma_y = 0.2e6 + 0.5e6 * st.eq_plastic_strain;
        assert!((tau[(0, 0)] + sigma_y).abs() < 1e-6 * sigma_y, "{} vs {}", tau[(0, 0)], -sigma_y);
        assert!(tau[(1, 1)].abs() < 1e-6 * sigma_y && tau[(2, 2)].abs() < 1e-6 * sigma_y);
        // Compaction strain ≈ |ln 0.4| − σ_y/E.
        let expected = -(0.4_f64.ln()) - sigma_y / 20e6;
        assert!((st.eq_plastic_strain - expected).abs() < 1e-9);
        // Unloading is elastic and idempotent.
        let (tau2, st2) = finite_strain_return(&m, &f, &st);
        assert!((tau2 - tau).norm() < 1e-6 * sigma_y);
        assert_eq!(st2.eq_plastic_strain, st.eq_plastic_strain);
        let (tau3, st3) = finite_strain_return(&m, &(f * 1.01), &st);
        assert_eq!(st3.eq_plastic_strain, st.eq_plastic_strain);
        assert!(tau3[(0, 0)] > tau[(0, 0)]);
    }

    #[test]
    fn crushable_foam_is_objective() {
        let m = Material::crushable_foam(20e6, 200.0, 0.2e6, 0.5e6);
        let f = Matrix3::new(0.5, 0.1, 0.0, 0.0, 1.1, 0.05, 0.0, 0.0, 0.9);
        let r = nalgebra::Rotation3::from_euler_angles(0.4, -1.1, 2.0).into_inner();
        let (t1, s1) = finite_strain_return(&m, &f, &FinitePlasticPoint::default());
        let (t2, s2) = finite_strain_return(&m, &(r * f), &FinitePlasticPoint::default());
        assert!((r * t1 * r.transpose() - t2).norm() / t1.norm() < 1e-9);
        assert!((s1.eq_plastic_strain - s2.eq_plastic_strain).abs() < 1e-12);
    }

    #[test]
    fn honeycomb_matches_foam_in_uniaxial_crush() {
        // Stepped uniaxial crush to 60 %: the rate model must reproduce the
        // Hencky foam's capped axial stress (same hardening on compaction)
        // and stay at zero lateral stress.
        let hc = Material::honeycomb(20e6, 200.0, 0.2e6, 0.5e6);
        let foam = Material::crushable_foam(20e6, 200.0, 0.2e6, 0.5e6);
        let mut st = RateFoamState::default();
        let mut tau = Matrix3::zeros();
        let mut f = Matrix3::identity();
        for k in 1..=200 {
            f[(0, 0)] = 1.0 - 0.6 * k as f64 / 200.0;
            let (t, s) = honeycomb_rate_return(&hc, &f, &st);
            tau = t;
            st = s;
        }
        let (_, sf) = finite_strain_return(&foam, &f, &FinitePlasticPoint::default());
        // Same compaction as the Hencky foam (log strain), and the *Cauchy*
        // stress on the hardened cap: σ = τ/J = −σ_y(c). (The Hencky foam
        // caps the Kirchhoff stress instead, so its true stress is 1/J higher.)
        assert!((st.compaction - sf.eq_plastic_strain).abs() < 2e-2 * sf.eq_plastic_strain);
        let j = f.determinant();
        let sigma_y = 0.2e6 + 0.5e6 * st.compaction;
        assert!((tau[(0, 0)] / j + sigma_y).abs() < 1e-6 * sigma_y, "{} vs {}", tau[(0, 0)] / j, -sigma_y);
        assert!(tau[(1, 1)].abs() < 1e-6 * tau[(0, 0)].abs());
        // Repeating the last configuration changes nothing.
        let (tau2, st2) = honeycomb_rate_return(&hc, &f, &st);
        assert!((tau2 - tau).norm() < 1e-9 * tau.norm());
        assert_eq!(st2.compaction, st.compaction);
        // Unloading a little is elastic (stress drops, compaction fixed).
        let mut f2 = f;
        f2[(0, 0)] += 0.01;
        let (tau3, st3) = honeycomb_rate_return(&hc, &f2, &st);
        assert!(tau3[(0, 0)] > tau[(0, 0)]);
        assert_eq!(st3.compaction, st.compaction);
    }

    #[test]
    fn honeycomb_is_objective_under_superposed_rotation() {
        // Crush, then rotate rigidly in small steps: the Cauchy stress must
        // rotate with the body and the compaction must not change.
        let hc = Material::honeycomb(20e6, 200.0, 0.2e6, 0.5e6);
        let mut st = RateFoamState::default();
        let f0 = Matrix3::new(0.7, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let mut f = Matrix3::identity();
        for k in 1..=100 {
            f = Matrix3::identity() + (f0 - Matrix3::identity()) * (k as f64 / 100.0);
            st = honeycomb_rate_return(&hc, &f, &st).1;
        }
        let (tau0, _) = honeycomb_rate_return(&hc, &f, &st);
        let c0 = st.compaction;
        let axis = nalgebra::Unit::new_normalize(nalgebra::Vector3::new(0.3, -0.5, 0.8));
        let mut tau = tau0;
        let total = 1.2;
        for k in 1..=200 {
            let r = nalgebra::Rotation3::from_axis_angle(&axis, total * k as f64 / 200.0).into_inner();
            let (t, s) = honeycomb_rate_return(&hc, &(r * f), &st);
            tau = t;
            st = s;
        }
        let r = nalgebra::Rotation3::from_axis_angle(&axis, total).into_inner();
        assert!((tau - r * tau0 * r.transpose()).norm() < 1e-4 * tau0.norm(), "rotated stress error {}", (tau - r * tau0 * r.transpose()).norm() / tau0.norm());
        assert!((st.compaction - c0).abs() < 1e-6);
    }
}

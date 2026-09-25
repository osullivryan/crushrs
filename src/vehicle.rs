//! Homogenised crushable-vehicle models for impact simulation, calibrated to
//! NHTSA NCAP rigid-barrier test measures.
//!
//! A vehicle is a block of C3D8 elements: a *crush zone* of length `L_c` at
//! the front made of finite-strain elastic–plastic material, and an elastic
//! body behind it. By default the crush zone is the whole vehicle (a uniform
//! crushable bar): yielding still concentrates at the loaded end because the
//! stress decays towards the rear, but the nominal strain at full crush stays
//! small (≈ 12 %), which keeps the isochoric plastic bulging and the
//! logarithmic-strain hardening in their near-linear range. A short crush
//! zone bottoms out at ~50 % strain and locks up. The crush zone reproduces
//! a bilinear barrier force–crush curve
//!
//! ```text
//! F(C) = F_y + k·C        (C = dynamic crush)
//! ```
//!
//! via `σ_y = F_y / A_front` and `H = k·L_c / A_front` (crush zone strain
//! ε ≈ C / L_c, so dF/dC = A·H/L_c).
//!
//! `F_y` and `k` are set from two NHTSA measures:
//!
//! * **KW400** – NCAP energy-equivalent stiffness over 25–400 mm of dynamic
//!   crush, `KW400 = 2·(E(400 mm) − E(25 mm)) / (0.4² − 0.025²)` with
//!   `E(d) = ∫₀ᵈ F dC` (Wiacek et al., NHTSA). For the bilinear curve this is
//!   `KW400 = k + 2·F_y·(0.4 − 0.025)/(0.4² − 0.025²) = k + 4.706·F_y`.
//! * **CRASH3 A/B ratio** – `F/w = A + B·C_res`, so `F_y/k = A/B`, taken from
//!   the class averages of the NHTSA crash test database
//!   (sedans A = 75.65 kg/cm, B = 11.07 kg/cm²; pickups A = 86.36, B = 11.72).
//!
//! The elastic modulus is chosen so the elastic energy stored at peak force
//! gives a coefficient of restitution near the measured 0.03–0.2 range.

use crate::material::Material;
use crate::mesh::{BlockFace, Mesh};
use crate::model::Model;
use crate::solver::Results;

/// Bilinear frontal force–crush model `F = F_y + k·C` (N, N/m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrushCurve {
    pub yield_force: f64,
    pub stiffness: f64,
}

impl CrushCurve {
    /// From NCAP KW400 (N/m) and the CRASH3 ratio A/B (m).
    pub fn from_kw400(kw400: f64, a_over_b: f64) -> Self {
        let c = 2.0 * (0.4 - 0.025) / (0.4_f64.powi(2) - 0.025_f64.powi(2)); // 4.706 /m
        let stiffness = kw400 / (1.0 + c * a_over_b);
        CrushCurve { yield_force: stiffness * a_over_b, stiffness }
    }

    /// NCAP KW400 of this curve (N/m).
    pub fn kw400(&self) -> f64 {
        self.kw_window((0.025, 0.4))
    }

    /// Energy-equivalent stiffness of this curve over a crush window (m).
    pub fn kw_window(&self, (w0, w1): (f64, f64)) -> f64 {
        let c = 2.0 * (w1 - w0) / (w1 * w1 - w0 * w0);
        self.stiffness + c * self.yield_force
    }

    /// Dynamic crush that absorbs `energy` joules.
    pub fn crush_for_energy(&self, energy: f64) -> f64 {
        // ½ k C² + F_y C − E = 0
        let (a, b) = (0.5 * self.stiffness, self.yield_force);
        (-b + (b * b + 4.0 * a * energy).sqrt()) / (2.0 * a)
    }
}

/// CRASH3 A/B ratio (m) from coefficients in kg/cm and kg/cm².
pub fn crash3_ratio(a_kg_per_cm: f64, b_kg_per_cm2: f64) -> f64 {
    a_kg_per_cm / b_kg_per_cm2 * 0.01
}

/// A homogenised vehicle.
#[derive(Clone, Debug)]
pub struct Vehicle {
    pub name: String,
    /// Test mass (kg).
    pub mass: f64,
    /// Length (x), width (y), height (z) of the block (m).
    pub size: [f64; 3],
    /// Length of the elastic–plastic crush zone at the front (m).
    pub crush_zone_length: f64,
    pub curve: CrushCurve,
    /// Young's modulus of the crush zone (Pa).
    pub modulus: f64,
    /// Young's modulus of the body behind the crush zone (Pa).
    pub body_modulus: f64,
    /// Side force–crush curve (whole-side, per the full vehicle length), for
    /// T-bone / side impacts. `None` = use the frontal curve.
    pub side_curve: Option<CrushCurve>,
    /// Modulus of the side profile (Pa); `None` = `modulus`.
    pub side_modulus: Option<f64>,
}

impl Vehicle {
    /// Build a vehicle from NCAP data. `restitution` sets the crush-zone
    /// modulus so the elastic energy at peak force is `e²` of the kinetic
    /// energy at `test_speed`.
    pub fn from_ncap(
        name: &str,
        mass: f64,
        size: [f64; 3],
        crush_zone_length: f64,
        kw400: f64,
        a_over_b: f64,
        restitution: f64,
        test_speed: f64,
    ) -> Self {
        let curve = CrushCurve::from_kw400(kw400, a_over_b);
        let kinetic = 0.5 * mass * test_speed * test_speed;
        let c_max = curve.crush_for_energy(kinetic);
        let f_max = curve.yield_force + curve.stiffness * c_max;
        let area = size[1] * size[2];
        // Elastic energy F²·L/(2·E·A) over the whole length = e²·KE.
        let modulus = f_max * f_max * size[0] / (2.0 * area * restitution * restitution * kinetic);
        Vehicle {
            name: name.to_string(),
            mass,
            size,
            crush_zone_length,
            curve,
            modulus,
            body_modulus: modulus,
            side_curve: None,
            side_modulus: None,
        }
    }

    /// Set the side curve from CRASH3-style side coefficients `A` (N/m of
    /// contact length) and `B` (N/m²): `F = L·(A + B·C)` for a hit spanning
    /// length `L` of the side. Stored for the full vehicle length.
    pub fn with_side_coefficients(mut self, a: f64, b: f64) -> Self {
        self.side_curve = Some(CrushCurve { yield_force: a * self.size[0], stiffness: b * self.size[0] });
        self
    }

    /// The vehicle turned 90°: a block whose x-axis is the lateral direction,
    /// carrying the side crush curve. Calibrate and impact it exactly like a
    /// frontal profile (`run_barrier_test`, `calibrate`, `add_vehicle`).
    pub fn side_profile(&self) -> Vehicle {
        let curve = self.side_curve.unwrap_or(self.curve);
        Vehicle {
            name: format!("{}_side", self.name),
            mass: self.mass,
            size: [self.size[1], self.size[0], self.size[2]],
            crush_zone_length: self.size[1],
            curve,
            modulus: self.side_modulus.unwrap_or(self.modulus),
            body_modulus: self.side_modulus.unwrap_or(self.modulus),
            side_curve: None,
            side_modulus: None,
        }
    }

    pub fn frontal_area(&self) -> f64 {
        self.size[1] * self.size[2]
    }

    pub fn density(&self) -> f64 {
        self.mass / (self.size[0] * self.size[1] * self.size[2])
    }

    /// Crush-zone material: honeycomb (rate form) with `σ_y = F_y/A`, `H = k·L_c/A`.
    pub fn crush_material(&self) -> Material {
        let a = self.frontal_area();
        Material::honeycomb(self.modulus, self.density(), self.curve.yield_force / a, self.curve.stiffness * self.crush_zone_length / a)
    }

    pub fn body_material(&self) -> Material {
        Material::elastic(self.body_modulus, 0.3, self.density())
    }

    /// Expected dynamic crush and peak force in a rigid-barrier test at `speed`.
    pub fn barrier_prediction(&self, speed: f64) -> (f64, f64) {
        let c = self.curve.crush_for_energy(0.5 * self.mass * speed * speed);
        (c, self.curve.yield_force + self.curve.stiffness * c)
    }

    /// 1996 Dodge Neon: NCAP test weight 1354 kg, KW400 = 1251 N/mm
    /// (NHTSA DOT HS 811 293), sedan-class CRASH3 A/B.
    /// Side: sedan side coefficients from FMVSS 214 MDB tests, A = 230 lb/in,
    /// B = 187 psi (2013 Ford Fusion; no Neon-specific side data is
    /// published), i.e. A = 40.3 kN/m, B = 1.29 MN/m².
    pub fn dodge_neon_1996() -> Self {
        Vehicle::from_ncap("neon", 1354.0, [4.36, 1.71, 1.35], 4.36, 1.251e6, crash3_ratio(75.65, 11.07), 0.12, 35.0 * 0.44704)
            .with_side_coefficients(230.0 * 4.44822 / 0.0254, 187.0 * 6894.76)
    }

    /// 2007 Chevrolet Silverado 1500 crew cab: NCAP test weight 2622 kg,
    /// KW400 = 2550 N/mm (NHTSA DOT HS 811 293), pickup-class CRASH3 A/B.
    pub fn chevrolet_silverado_2007() -> Self {
        Vehicle::from_ncap("silverado", 2622.0, [5.85, 2.03, 1.87], 5.85, 2.550e6, crash3_ratio(86.36, 11.72), 0.12, 35.0 * 0.44704)
    }
}

impl Vehicle {
    /// 1996 Dodge Neon, tuned with [`calibrate`] at 0.3 m elements
    /// (honeycomb model, one-point integration) so the simulated 35 mph
    /// barrier test matches NHTSA: KW400 1269 vs 1251 N/mm, max crush 538
    /// vs 527 mm, restitution 0.112.
    pub fn dodge_neon_1996_tuned() -> Self {
        let mut v = Vehicle::dodge_neon_1996();
        v.curve = CrushCurve { yield_force: 16.73e3, stiffness: 2.930e5 };
        v.modulus = 23.1e6;
        v.body_modulus = v.modulus;
        // Side profile tuned at 0.3 m elements (whole side into a wall at
        // 45 mph): KW150 7622 vs 7628 N/mm, restitution 0.084. Crush
        // saturates at ~200 mm (target 283 mm from the unbounded bilinear
        // curve); published side saturation crush is 180–240 mm.
        v.side_curve = Some(CrushCurve { yield_force: 2.75e3, stiffness: 4.657e6 });
        v.side_modulus = Some(125.8e6);
        v
    }

    /// 2007 Chevrolet Silverado, tuned with [`calibrate`] at 0.3 m elements
    /// (honeycomb, one-point): KW400 2563 vs 2550 N/mm, max crush 540 vs
    /// 513 mm, restitution 0.119.
    pub fn chevrolet_silverado_2007_tuned() -> Self {
        let mut v = Vehicle::chevrolet_silverado_2007();
        v.curve = CrushCurve { yield_force: 51.21e3, stiffness: 5.592e5 };
        v.modulus = 27.8e6;
        v.body_modulus = v.modulus;
        v
    }

    /// Element size the `_tuned` constructors were calibrated at (m).
    pub const TUNED_ELEMENT_SIZE: f64 = 0.3;
}

/// Which way the vehicle faces along x.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Heading {
    /// Front face at x_max, travelling towards +x.
    PlusX,
    /// Front face at x_min, travelling towards −x.
    MinusX,
}

/// Add a vehicle block to the mesh with its front face at `front_x`,
/// centred on `y_center`, standing on `z_bottom`. Creates part `<name>`
/// (also a node set) and face set `<name>_front`.
pub fn add_vehicle(mesh: &mut Mesh, v: &Vehicle, front_x: f64, y_center: f64, z_bottom: f64, heading: Heading, element_size: f64) -> usize {
    let (x0, face) = match heading {
        Heading::PlusX => (front_x - v.size[0], BlockFace::XMax),
        Heading::MinusX => (front_x, BlockFace::XMin),
    };
    let front = format!("{}_front", v.name);
    mesh.add_hex_block(&v.name, [x0, y_center - v.size[1] / 2.0, z_bottom], v.size, element_size, &[(face, &front)])
}

/// Assign the vehicle's crush material and initial velocity in the model.
/// The crush zone is the whole block unless `crush_zone_length` is shorter
/// (then the rear elements get the elastic body material — only meaningful
/// with separate parts, so here the whole part gets the crush material).
pub fn assign_vehicle(model: &mut Model, v: &Vehicle, heading: Heading, speed: f64) {
    model.set_material(&v.name, v.crush_material());
    let vx = match heading {
        Heading::PlusX => speed,
        Heading::MinusX => -speed,
    };
    model.set_initial_velocity(&v.name, [vx, 0.0, 0.0]);
}

/// Add a rear-seat-style accelerometer to a vehicle part: at the quarter
/// point from the rear (opposite the `front_face` set), lateral centre,
/// 35% of the height, with a body-fixed frame. Named `<part>_rear`.
pub fn add_vehicle_accelerometer(model: &mut Model, part: &str, front_face: &str) -> String {
    let p = model.mesh.part_index(part).unwrap_or_else(|| panic!("no part '{}'", part));
    let nodes = model.mesh.part_nodes(p);
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for &n in &nodes {
        for d in 0..3 {
            lo[d] = lo[d].min(model.mesh.nodes[n][d]);
            hi[d] = hi[d].max(model.mesh.nodes[n][d]);
        }
    }
    let front = model.mesh.face_set_nodes(front_face).unwrap_or_else(|| panic!("no face set '{}'", front_face));
    let mut c = [0.0; 3];
    for &n in &front {
        for d in 0..3 {
            c[d] += model.mesh.nodes[n][d] / front.len() as f64;
        }
    }
    let mut at = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2] + 0.35 * (hi[2] - lo[2])];
    for d in 0..2 {
        let len = hi[d] - lo[d];
        if (c[d] - lo[d]).abs() < 1e-6 * len {
            at[d] = hi[d] - 0.25 * len;
        } else if (c[d] - hi[d]).abs() < 1e-6 * len {
            at[d] = lo[d] + 0.25 * len;
        }
    }
    let name = format!("{}_rear", part);
    model.add_accelerometer_at(&name, part, at);
    name
}

/// Rigid barrier: a fixed stiff block with its contact face at `x`. Creates
/// part `<name>` and face set `<name>_face`.
pub fn add_rigid_wall(mesh: &mut Mesh, name: &str, x: f64, y: [f64; 2], z: [f64; 2], facing: Heading, thickness: f64) -> usize {
    let (x0, face) = match facing {
        Heading::PlusX => (x - thickness, BlockFace::XMax),
        Heading::MinusX => (x, BlockFace::XMin),
    };
    let face_name = format!("{}_face", name);
    let h = ((y[1] - y[0]).min(z[1] - z[0]) / 4.0).max(thickness);
    mesh.add_hex_block(name, [x0, y[0], z[0]], [thickness, y[1] - y[0], z[1] - z[0]], h, &[(face, &face_name)])
}

/// Barrier-test measures for one body, from `Results::history`
/// (record with `history_steps`). Force is `−m·dv/dt` of the body's
/// mean velocity, which is what an accelerometer-based NCAP analysis uses;
/// crush is the body's mean displacement since first contact.
#[derive(Clone, Debug)]
pub struct BarrierMetrics {
    /// Energy-equivalent stiffness over the crush window (KW400 for the
    /// default 25–400 mm window).
    pub kw400: f64,
    pub max_crush: f64,
    pub peak_force: f64,
    /// Mean force over the loading phase (contact to max crush).
    pub mean_force: f64,
    pub restitution: f64,
    pub delta_v: f64,
    /// (crush, force) samples over the loading phase.
    pub curve: Vec<(f64, f64)>,
}


pub fn barrier_metrics(model: &Model, results: &Results, part: &str, axis: usize) -> BarrierMetrics {
    barrier_metrics_window(model, results, part, axis, (0.025, 0.4))
}

/// [`barrier_metrics`] with an energy-equivalent stiffness window other than
/// NCAP's 25–400 mm (e.g. for side profiles that crush less).
pub fn barrier_metrics_window(model: &Model, results: &Results, part: &str, axis: usize, window: (f64, f64)) -> BarrierMetrics {
    let p = model.mesh.part_index(part).unwrap_or_else(|| panic!("no part '{}'", part));
    let series: Vec<(f64, f64, f64, f64)> = results
        .history
        .iter()
        .map(|(t, states)| {
            let s = &states[p];
            (*t, s.mass, s.velocity[axis], s.displacement[axis])
        })
        .collect();
    assert!(series.len() > 3, "no history for '{}' (set history_steps)", part);
    let v0 = series[0].2;
    let sign = v0.signum();
    let n = series.len();
    let mut force = vec![0.0; n];
    for i in 1..n - 1 {
        let (t0, m, v_prev, _) = series[i - 1];
        let (t2, _, v_next, _) = series[i + 1];
        force[i] = -sign * m * (v_next - v_prev) / (t2 - t0);
    }
    let force: Vec<f64> = (0..n).map(|i| if i == 0 || i == n - 1 { force[i] } else { (force[i - 1] + 2.0 * force[i] + force[i + 1]) / 4.0 }).collect();
    let peak_force = force.iter().cloned().fold(0.0, f64::max);
    let contact = force.iter().position(|f| *f > 0.02 * peak_force).unwrap_or(0);
    let x_contact = series[contact].3;
    let crush: Vec<f64> = series.iter().map(|s| sign * (s.3 - x_contact)).collect();
    let i_max = crush.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
    let max_crush = crush[i_max];
    let curve: Vec<(f64, f64)> = (contact..=i_max).map(|i| (crush[i], force[i])).collect();
    let energy_at = |d: f64| -> f64 {
        let mut e = 0.0;
        for w in curve.windows(2) {
            let (c0, f0) = w[0];
            let (c1, f1) = w[1];
            if c1 <= c0 {
                continue;
            }
            if c1 <= d {
                e += 0.5 * (f0 + f1) * (c1 - c0);
            } else if c0 < d {
                let f_d = f0 + (f1 - f0) * (d - c0) / (c1 - c0);
                e += 0.5 * (f0 + f_d) * (d - c0);
                break;
            } else {
                break;
            }
        }
        e
    };
    let (w0, w1) = window;
    let kw400 = 2.0 * (energy_at(w1) - energy_at(w0)) / (w1 * w1 - w0 * w0);
    let mean_force = if max_crush > 0.0 { energy_at(max_crush) / max_crush } else { 0.0 };
    let v_end = series[n - 1].2;
    BarrierMetrics { kw400, max_crush, peak_force, mean_force, restitution: -v_end / v0, delta_v: v_end - v0, curve }
}

/// Build the NCAP-style rigid-barrier model of `v` at `speed`.
pub fn barrier_model(v: &Vehicle, element_size: f64, speed: f64, end_time: f64, frame_steps: usize) -> Model {
    let gap = 0.005;
    let mut mesh = Mesh::new();
    add_vehicle(&mut mesh, v, -gap, 0.0, 0.0, Heading::PlusX, element_size);
    let (w, z) = (v.size[1], v.size[2]);
    add_rigid_wall(&mut mesh, "wall", 0.0, [-w * 0.75, w * 0.75], [-0.2 * z, 1.2 * z], Heading::MinusX, 0.2);
    let mut model = Model::new(mesh);
    assign_vehicle(&mut model, v, Heading::PlusX, speed);
    model.set_material("wall", Material::elastic(1e9, 0.3, 1000.0)).fix("wall");
    // Contact penalty per node: much stiffer than the crush stiffness shared
    // over the front-face nodes, so the penalty springs store ≲1 % of the
    // energy (otherwise they return it as spurious restitution).
    let n_face = model.mesh.face_set_nodes(&format!("{}_front", v.name)).unwrap().len() as f64;
    let front = format!("{}_front", v.name);
    model.add_contact_pair(&front, "wall_face", 400.0 * v.curve.stiffness / n_face, 0.3);
    model.settings.end_time = end_time;
    model.settings.history_steps = 2;
    model.settings.frame_steps = frame_steps;
    model
}

/// Run an NCAP-style rigid-barrier test and return the model, results and metrics.
pub fn run_barrier_test(v: &Vehicle, element_size: f64, speed: f64, end_time: f64, frame_steps: usize) -> (Model, Results, BarrierMetrics) {
    let model = barrier_model(v, element_size, speed, end_time, frame_steps);
    let results = crate::solver::run(&model);
    let metrics = barrier_metrics(&model, &results, &v.name, 0);
    (model, results, metrics)
}

/// Calibration targets for [`calibrate`].
#[derive(Clone, Copy, Debug)]
pub struct BarrierTargets {
    pub kw400: f64,
    pub max_crush: f64,
    pub restitution: f64,
    pub speed: f64,
    /// Crush window for the stiffness metric (m).
    pub window: (f64, f64),
}

impl BarrierTargets {
    /// Targets implied by a vehicle's NCAP curve at 35 mph.
    pub fn from_vehicle(v: &Vehicle, restitution: f64) -> Self {
        BarrierTargets::from_curve(v, 35.0 * 0.44704, (0.025, 0.4), restitution)
    }

    /// Targets implied by the vehicle's curve at any speed and window.
    pub fn from_curve(v: &Vehicle, speed: f64, window: (f64, f64), restitution: f64) -> Self {
        BarrierTargets { kw400: v.curve.kw_window(window), max_crush: v.barrier_prediction(speed).0, restitution, speed, window }
    }
}

/// Tune the crush-zone material so the *simulated* barrier test reproduces
/// the targets at the given element size. The analytic mapping in
/// [`Vehicle::crush_material`] over-predicts stiffness because plastic flow is
/// isochoric (the crushed zone bulges and its load-bearing area grows) and
/// hardening acts on logarithmic strain, so the effective `F_y`, `k` and
/// modulus are corrected iteratively (log-space secant on the three ratios).
/// Returns the tuned vehicle and the final metrics.
pub fn calibrate(v: &Vehicle, targets: &BarrierTargets, element_size: f64, iterations: usize, verbose: bool) -> (Vehicle, BarrierMetrics) {
    let mut tuned = v.clone();
    let mut last = None;
    for it in 0..iterations {
        let model = barrier_model(&tuned, element_size, targets.speed, 0.14, 0);
        let results = crate::solver::run(&model);
        let m = barrier_metrics_window(&model, &results, &tuned.name, 0, targets.window);
        if verbose {
            log::info!(
                "calibrate {} it {}: KW {:.0}/{:.0} N/mm, crush {:.0}/{:.0} mm, e {:.3}/{:.3}",
                tuned.name, it, m.kw400 / 1e3, targets.kw400 / 1e3, m.max_crush * 1e3, targets.max_crush * 1e3, m.restitution, targets.restitution
            );
        }
        let r_c = targets.max_crush / m.max_crush;
        let r_k = if m.max_crush >= targets.window.1 { targets.kw400 / m.kw400 } else { 1.0 / r_c.max(1.0) };
        let r_e = targets.restitution / m.restitution.max(1e-3);
        last = Some(m);
        if (r_k - 1.0).abs() < 0.02 && (r_c - 1.0).abs() < 0.02 && (r_e - 1.0).abs() < 0.1 {
            break;
        }
        let damp = 0.7;
        tuned.curve.stiffness *= r_k.powf(damp);
        tuned.curve.yield_force *= (1.0 / r_c).powf(damp) * r_k.powf(0.3 * damp);
        if (r_k - 1.0).abs() < 0.15 && (r_c - 1.0).abs() < 0.15 {
            tuned.modulus *= (1.0 / r_e).powf(2.0 * damp).clamp(0.5, 2.0);
            tuned.body_modulus = tuned.modulus;
        }
    }
    (tuned, last.unwrap())
}

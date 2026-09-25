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
use serde::{Deserialize, Serialize};

/// Bilinear frontal force–crush model `F = F_y + k·C` (N, N/m).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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

/// Rail box of a crush zone: the frame rails and the engine between them, the box that carries
/// most of the barrier force, inside a soft skin (fenders, hood, radiator
/// support, everything above and outside the rails).
///
/// The rail box is the crush-zone elements whose centroid lies within
/// `width` of the vehicle's lateral centreline and between `z_range`
/// (heights above the ground). It carries `force_fraction` of the
/// vehicle's force–crush curve over its own area, the skin the rest over
/// the remaining area; modulus and density are split the same way, so
/// rail box and skin have the same wave speed and the same strain history
/// against a flat rigid barrier: the barrier calibration is untouched. A
/// partner that overlaps only part of the face (a narrower, lower car)
/// sees the concentrated load instead of a uniformly spread one.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RailBox {
    /// Lateral extent of the rail box (m), centred on the vehicle.
    pub width: f64,
    /// Bottom and top of the rail box above the ground (m).
    pub z_range: [f64; 2],
    /// Fraction of the barrier force carried by the rail box (0..1).
    pub force_fraction: f64,
}

/// A homogenised vehicle. Serialises to TOML (`Vehicle::save` /
/// `Vehicle::load`), which is how a vehicle tuned by `crushrs tune` is
/// kept and passed to the impact commands.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default = "Vehicle::blank")]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub side_curve: Option<CrushCurve>,
    /// Modulus of the side profile (Pa); `None` = `modulus`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub side_modulus: Option<f64>,
    /// Rear force–crush curve (whole rear face), for rear-end impacts.
    /// `None` = no rear structure known (rear impacts refuse to run).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rear_curve: Option<CrushCurve>,
    /// Modulus of the rear profile (Pa); `None` = `modulus`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rear_modulus: Option<f64>,
    /// Tabulated frontal force–crush curve `[[crush m, force N], ...]`
    /// (from a measured pulse, see [`calibrate_pulse`]). When set it
    /// replaces `curve` for the crush material.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub force_table: Option<Vec<[f64; 2]>>,
    /// Element length along the crush direction inside the crush zone
    /// (`None` = the block element size). Only used when the crush zone is
    /// shorter than the vehicle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crush_element_size: Option<f64>,
    /// Compaction at each `force_table` knot (from [`calibrate_pulse`]);
    /// `None` = uniform crush of the crush zone, `c = −ln(1 − x/L_c)`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compaction_map: Option<Vec<f64>>,
    /// Transverse cap factor of the crush material (see
    /// `Plasticity::transverse_factor`): sideways the front is this many
    /// times stronger than along the crush axis.
    pub transverse_factor: f64,
    /// Stiff elastic bumper layer at the front of the crush zone
    /// `[thickness m, mass kg, modulus Pa]` (part `<name>_bumper`): spreads
    /// nodal contact loads into the honeycomb like a bumper beam, so two
    /// soft fronts meet as two stiff faces. Only with a crush zone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bumper: Option<[f64; 3]>,
    /// Mass of the crush-zone material (kg); the rest of the vehicle mass
    /// sits in the body block. `None` = uniform density. A light crush
    /// zone carries the plastic wave faster (√(H/ρ)), like real rails
    /// ahead of the engine and cabin mass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crush_zone_mass: Option<f64>,
    /// Rail box (frame rails and engine) of the crush zone, see [`RailBox`];
    /// part `<name>_rails`. Only with a crush zone. `None` = uniform face.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rail_box: Option<RailBox>,
}

impl Vehicle {
    /// An empty vehicle: the serde default for missing TOML fields.
    fn blank() -> Self {
        Vehicle {
            name: String::new(),
            mass: 0.0,
            size: [0.0; 3],
            crush_zone_length: 0.0,
            curve: CrushCurve { yield_force: 0.0, stiffness: 0.0 },
            modulus: 0.0,
            body_modulus: 0.0,
            side_curve: None,
            side_modulus: None,
            rear_curve: None,
            rear_modulus: None,
            force_table: None,
            crush_element_size: None,
            compaction_map: None,
            crush_zone_mass: None,
            bumper: None,
            rail_box: None,
            transverse_factor: Vehicle::TRANSVERSE_FACTOR,
        }
    }

    /// A vehicle with only its name, mass and dimensions: the starting
    /// point of `crushrs tune`, which fits everything else to the tests.
    /// The bilinear `curve` is a placeholder until a pulse fit sets it
    /// (from the fitted table's first knot and overall slope).
    pub fn new(name: &str, mass: f64, size: [f64; 3]) -> Self {
        let mut v = Vehicle::blank();
        v.name = name.to_string();
        v.mass = mass;
        v.size = size;
        v.crush_zone_length = size[0];
        v.modulus = 20e6;
        v.body_modulus = 20e6;
        v
    }

    /// Read a vehicle from a TOML file written by [`Vehicle::save`] or
    /// by `crushrs tune`.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let v: Vehicle = toml::from_str(&text).map_err(|e| format!("{}: {}", path.display(), e))?;
        if v.name.is_empty() || v.mass <= 0.0 || v.size.iter().any(|s| *s <= 0.0) || v.modulus <= 0.0 {
            return Err(format!("{}: a vehicle needs name, mass, size and modulus", path.display()));
        }
        Ok(v)
    }

    /// Write the vehicle as TOML.
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let text = toml::to_string(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("{}: {}", path.display(), e))
    }

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
            side_modulus: None, rear_curve: None, rear_modulus: None, force_table: None, crush_element_size: None, compaction_map: None, crush_zone_mass: None, bumper: None, rail_box: None, transverse_factor: Vehicle::TRANSVERSE_FACTOR,
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
            side_modulus: None, rear_curve: None, rear_modulus: None, force_table: None, crush_element_size: None, compaction_map: None, crush_zone_mass: None, bumper: None, rail_box: None, transverse_factor: Vehicle::TRANSVERSE_FACTOR,
        }
    }

    /// Set the rear curve from CRASH3-style rear coefficients `A` (N/m of
    /// width) and `B` (N/m²): `F = w·(A + B·C)` over the full width.
    pub fn with_rear_coefficients(mut self, a: f64, b: f64) -> Self {
        self.rear_curve = Some(CrushCurve { yield_force: a * self.size[1], stiffness: b * self.size[1] });
        self
    }

    /// The vehicle seen from behind: a uniform crushable block carrying the
    /// rear curve, whose "front" face is the rear of the car. Calibrate and
    /// impact it like a frontal profile; in a rear-end the struck vehicle is
    /// this profile heading *away* (a negative speed in `headon::HeadOn`).
    pub fn rear_profile(&self) -> Option<Vehicle> {
        let curve = self.rear_curve?;
        let mut v = self.clone();
        v.name = format!("{}_rear", self.name);
        v.crush_zone_length = self.size[0];
        v.curve = curve;
        v.modulus = self.rear_modulus.unwrap_or(self.modulus);
        v.body_modulus = v.modulus;
        v.side_curve = None;
        v.side_modulus = None;
        v.rear_curve = None;
        v.rear_modulus = None;
        v.force_table = None;
        v.crush_element_size = None;
        v.compaction_map = None;
        v.crush_zone_mass = None;
        v.bumper = None;
        v.rail_box = None;
        Some(v)
    }

    pub fn frontal_area(&self) -> f64 {
        self.size[1] * self.size[2]
    }

    /// Whether the crush zone is shorter than the block (two-part vehicle).
    pub fn has_crush_zone(&self) -> bool {
        self.crush_zone_length < self.size[0] - 1e-9
    }

    /// Part name of the crush zone when [`has_crush_zone`](Self::has_crush_zone).
    pub fn crush_part_name(&self) -> String {
        format!("{}_crush", self.name)
    }

    /// Part name of the bumper layer when `bumper` is set.
    pub fn bumper_part_name(&self) -> String {
        format!("{}_bumper", self.name)
    }

    /// Part name of the rail box when `rail_box` is set.
    pub fn rail_box_part_name(&self) -> String {
        format!("{}_rails", self.name)
    }

    /// Part name of the bumper layer over the rail box (the bumper beam on
    /// the rail ends) when both `bumper` and `rail_box` are set.
    pub fn bumper_rails_part_name(&self) -> String {
        format!("{}_bumper_rails", self.name)
    }

    /// Whether the vehicle is built with a rail box.
    pub fn has_rail_box(&self) -> bool {
        self.rail_box.is_some() && self.has_crush_zone()
    }

    /// Whether a crush-zone element with cross-section centroid `(y, z)`
    /// (`y` from the vehicle's centreline, `z` above the ground) belongs to
    /// the rail box.
    pub fn in_rail_box(&self, y: f64, z: f64) -> bool {
        match self.rail_box {
            Some(c) => y.abs() < 0.5 * c.width && z > c.z_range[0] && z < c.z_range[1],
            None => false,
        }
    }

    /// Stress / modulus / density scale factors of the rail box and of the
    /// skin relative to the uniform crush material, for a rail box occupying
    /// `area_fraction` of the crush-zone cross-section: the rail box carries
    /// `force_fraction` of the force over its area, the skin the rest.
    pub fn rail_box_factors(&self, area_fraction: f64) -> (f64, f64) {
        let f = self.rail_box.map_or(0.0, |c| c.force_fraction).clamp(0.0, 1.0);
        let a = area_fraction.clamp(1e-6, 1.0 - 1e-6);
        (f / a, (1.0 - f) / (1.0 - a))
    }

    /// Length of the honeycomb crush zone proper (crush zone minus bumper).
    pub fn honeycomb_length(&self) -> f64 {
        self.crush_zone_length - self.bumper.map_or(0.0, |b| b[0])
    }

    /// Contact penalty stiffness per front-face node (N/m): the axial
    /// stiffness of the material behind the node, `E·A_node/h`, so the
    /// penalty is neither softer than the structure (spurious restitution
    /// from energy parked in the springs) nor so stiff that nodal contact
    /// loads crush single elements flat. `n_face` is the number of nodes on
    /// the front face, `h` the block element size.
    pub fn contact_stiffness_per_node(&self, n_face: usize, h: f64) -> f64 {
        // With a bumper the penalty is still that of the honeycomb behind it:
        // the plate only spreads the load, it must not make the contact
        // itself stiff enough to fold the plate on impact.
        let h_front = if self.has_crush_zone() { self.crush_element_size.unwrap_or(h) } else { h };
        self.modulus * self.frontal_area() / (n_face as f64 * h_front)
    }

    /// Per-node factor on the front-face contact penalty (in the order of
    /// `Mesh::face_set_nodes` of `<name>_front`): the rail-box factor for
    /// nodes over the rail box, the skin factor elsewhere, averaged over
    /// the crush-zone columns a node borders. Empty (uniform) without a
    /// rail box.
    pub fn front_contact_scales(&self, mesh: &Mesh, element_size: f64) -> Vec<f64> {
        if !self.has_rail_box() {
            return Vec::new();
        }
        let front = format!("{}_front", self.name);
        let Some(nodes) = mesh.face_set_nodes(&front) else { return Vec::new() };
        let ny = (self.size[1] / element_size).round().max(1.0) as usize;
        let nz = (self.size[2] / element_size).round().max(1.0) as usize;
        let (dy, dz) = (self.size[1] / ny as f64, self.size[2] / nz as f64);
        // Face extent from the nodes themselves (any heading, offset, height).
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for &n in &nodes {
            for d in 0..3 {
                lo[d] = lo[d].min(mesh.nodes[n][d]);
                hi[d] = hi[d].max(mesh.nodes[n][d]);
            }
        }
        let (y_center, z_bottom) = (0.5 * (lo[1] + hi[1]), lo[2]);
        let cells = ny * nz;
        let n_rails = (0..cells).filter(|c| self.in_rail_box((c % ny) as f64 * dy + 0.5 * dy - 0.5 * self.size[1], (c / ny) as f64 * dz + 0.5 * dz)).count();
        if n_rails == 0 || n_rails == cells {
            return Vec::new();
        }
        let (f_rails, f_skin) = self.rail_box_factors(n_rails as f64 / cells as f64);
        nodes
            .iter()
            .map(|&n| {
                let p = mesh.nodes[n];
                let (mut sum, mut count) = (0.0, 0);
                for (sy, sz) in [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)] {
                    let (yc, zc) = (p[1] - y_center + sy * dy, p[2] - z_bottom + sz * dz);
                    if yc.abs() < 0.5 * self.size[1] && zc > 0.0 && zc < self.size[2] {
                        sum += if self.in_rail_box(yc, zc) { f_rails } else { f_skin };
                        count += 1;
                    }
                }
                if count > 0 { sum / count as f64 } else { 1.0 }
            })
            .collect()
    }

    /// Stiffness scale for contact penalties (N/m).
    pub fn contact_stiffness_reference(&self) -> f64 {
        match &self.force_table {
            Some(t) => {
                let last = t.last().unwrap();
                (last[1] / last[0].max(1e-3)).max(self.curve.stiffness)
            }
            None => self.curve.stiffness,
        }
    }

    pub fn density(&self) -> f64 {
        self.mass / (self.size[0] * self.size[1] * self.size[2])
    }

    /// Density of the crush-zone material.
    pub fn crush_density(&self) -> f64 {
        match self.crush_zone_mass {
            Some(m) if self.has_crush_zone() => (m - self.bumper.map_or(0.0, |b| b[1])) / (self.honeycomb_length() * self.frontal_area()),
            _ => self.density(),
        }
    }

    /// Density of the body block behind the crush zone.
    pub fn body_density(&self) -> f64 {
        match self.crush_zone_mass {
            Some(m) if self.has_crush_zone() => (self.mass - m) / ((self.size[0] - self.crush_zone_length) * self.frontal_area()),
            _ => self.density(),
        }
    }

    /// Crush-zone material: honeycomb (rate form) with `σ_y = F_y/A`, `H = k·L_c/A`.
    pub fn crush_material(&self) -> Material {
        let mut m = self.crush_material_axial();
        if let Some(p) = m.plasticity.as_mut() {
            p.transverse_factor = self.transverse_factor;
        }
        m
    }

    fn crush_material_axial(&self) -> Material {
        let a = self.frontal_area();
        match &self.force_table {
            Some(table) => {
                let curve = match &self.compaction_map {
                    Some(cm) if cm.len() == table.len() => table.iter().zip(cm).map(|([_, f], c)| [*c, f / a]).collect(),
                    _ => Self::yield_curve(table, self.honeycomb_length(), a),
                };
                Material::honeycomb_curve(self.modulus, self.crush_density(), curve, Some([Self::LOCK_COMPACTION, Self::LOCK_SLOPE_FACTOR * self.modulus]))
            }
            None => Material::honeycomb(self.modulus, self.crush_density(), self.curve.yield_force / a, self.curve.stiffness * self.crush_zone_length / a),
        }
    }

    /// Default transverse cap factor of the crush material.
    pub const TRANSVERSE_FACTOR: f64 = 4.0;

    /// Ground clearance (m) of the road under every vehicle model: the
    /// underbody bottoms out on a rigid road this far below it, which is
    /// what keeps a nose from diving under a partner or pitching freely
    /// when the load box sits below the centre of gravity.
    pub const GROUND_CLEARANCE: f64 = 0.15;


    /// Compaction at which a tabulated crush material locks up (≈ 75 %
    /// crushed); beyond it the yield stress rises with `LOCK_SLOPE_FACTOR·E`.
    pub const LOCK_COMPACTION: f64 = 1.4;
    pub const LOCK_SLOPE_FACTOR: f64 = 1.0;

    /// Force–crush table → yield stress vs compaction, assuming the crush
    /// zone (length `l`) compacts uniformly: `c = −ln(1 − x/l)`, `σ = F/A`.
    pub fn yield_curve(table: &[[f64; 2]], l: f64, area: f64) -> Vec<[f64; 2]> {
        table.iter().map(|[x, f]| [-(1.0 - (x / l).min(0.95)).ln(), f / area]).collect()
    }

    pub fn body_material(&self) -> Material {
        Material::elastic(self.body_modulus, 0.3, self.body_density())
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
    /// 1996 Dodge Neon fitted to the left-rear-seat X pulse of NHTSA test
    /// 2320 (NCAP, 56.5 km/h rigid barrier, 1354 kg; curve 78, CFC 60) with
    /// [`calibrate_pulse`]: a 1.5 m crush zone of 0.1 m elements and 190 kg
    /// ahead of an elastic body. Simulated vs measured: delta-v 17.64 vs
    /// 17.65 m/s at 150 ms, peak −32.9 vs −35.2 g, max crush 800 vs 736 mm
    /// at 83 vs 78 ms, restitution 0.13 vs 0.17, CFC 60 RMS error 6.2 g.
    /// `crushrs barrier neon-pulse --pulse data/nhtsa/v02320tsv.078`.
    pub fn dodge_neon_1996_pulse() -> Self {
        let mut v = Vehicle::dodge_neon_1996_tuned();
        v.crush_zone_length = 1.5;
        v.crush_element_size = Some(0.1);
        v.crush_zone_mass = Some(190.0);
        v.bumper = Some([0.1, 25.0, 3.0e8]);
        // Rail box fitted with `calibrate_rail_boxes` to the car-to-car test
        // 4429 together with the Navigator's (`headon navigator neon-pulse
        // 30.14 30.14 --mass-b 1378 --pulse-a ... --pulse-b ...
        // --calibrate-rail-boxes 12`, J 2.92 vs 3.43 for uniform faces):
        // the front rails and subframe, 0.45 m wide at 0.06–0.41 m,
        // carrying 80 % of the force. At 0.3 m elements that is the
        // middle 2 of 6 columns and rows 1–2 of 5 (0–0.54 m). Invisible
        // to the barrier test above, which stays as fitted.
        v.rail_box = Some(RailBox { width: 0.45, z_range: [0.06, 0.41], force_fraction: 0.8 });
        v.modulus = 4e6;
        v.body_modulus = 2.7e8;
        let knots = [0.0, 0.1165, 0.2330, 0.3496, 0.4661, 0.5826, 0.6991];
        let force_kn = [13.0, 135.0, 135.0, 146.0, 361.0, 364.0, 364.0];
        v.force_table = Some(knots.iter().zip(force_kn).map(|(x, f)| [*x, f * 1e3]).collect());
        v
    }

    /// 1999 Ford Expedition (body-on-frame full-size SUV; the Lincoln
    /// Navigator is the same platform), for pulse calibration against NHTSA
    /// test 3124 (flat rigid barrier at 48.5 km/h, 2460 kg, rear frame
    /// crossmember X): a 1.7 m crush zone with 320 kg and a bumper.
    pub fn ford_expedition_1999() -> Self {
        let mut v = Vehicle::from_ncap("expedition", 2460.0, [5.2, 2.0, 1.9], 5.2, 2.5e6, crash3_ratio(86.36, 11.72), 0.12, 30.0 * 0.44704);
        v.crush_zone_length = 1.7;
        v.crush_element_size = Some(0.1);
        v.crush_zone_mass = Some(320.0);
        v.bumper = Some([0.1, 35.0, 3.0e8]);
        // Rail box fitted to the car-to-car test 4429 (see
        // `dodge_neon_1996_pulse`): the frame rails and bumper beam,
        // 1.25 m wide at 0.15–0.72 m, carrying 75 % of the force. At 0.3 m
        // elements it is the middle 5 of 7 columns and row 2 of 6
        // (0.32–0.63 m).
        v.rail_box = Some(RailBox { width: 1.25, z_range: [0.15, 0.72], force_fraction: 0.75 });
        v.modulus = 7.3e6;
        v.body_modulus = 3.93e8;
        // Fitted to test 3124 (`barrier expedition --pulse
        // data/nhtsa/v03124tsv.031,data/nhtsa/v03124tsv.032 --mph 30.14
        // --calibrate 8`): crush 677 vs 672 mm, e 0.118 vs 0.130, peak −21
        // vs −28 g, CFC 60 RMS 3.5 g.
        let knots = [0.0, 0.1064, 0.2128, 0.3192, 0.4256, 0.5320, 0.6384];
        let force_kn = [29.0, 183.0, 259.0, 417.0, 457.0, 457.0, 457.0];
        v.force_table = Some(knots.iter().zip(force_kn).map(|(x, f)| [*x, f * 1e3]).collect());
        v
    }

    /// 1999 Lincoln Navigator as tested in NHTSA test 4429 (2873 kg):
    /// the Expedition structure with the Navigator's mass and dimensions
    /// (length 5.175 m, width 2.049 m).
    pub fn lincoln_navigator_1999() -> Self {
        let mut v = Vehicle::ford_expedition_1999();
        v.name = "navigator".into();
        v.mass = 2873.0;
        v.size = [5.175, 2.049, 1.9];
        v
    }

    /// Same vehicle with another test mass (structure unchanged: the extra
    /// mass goes to the body block).
    pub fn with_mass(mut self, mass: f64) -> Self {
        self.mass = mass;
        self
    }

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
/// (also a node set), face set `<name>_front` and face set `<name>_bottom`
/// (the underbody of every part, for a ground contact).
pub fn add_vehicle(mesh: &mut Mesh, v: &Vehicle, front_x: f64, y_center: f64, z_bottom: f64, heading: Heading, element_size: f64) -> usize {
    let (x0, face) = match heading {
        Heading::PlusX => (front_x - v.size[0], BlockFace::XMax),
        Heading::MinusX => (front_x, BlockFace::XMin),
    };
    let front = format!("{}_front", v.name);
    let bottom = format!("{}_bottom", v.name);
    let origin = [x0, y_center - v.size[1] / 2.0, z_bottom];
    if !v.has_crush_zone() {
        return mesh.add_hex_block(&v.name, origin, v.size, element_size, &[(face, &front), (BlockFace::ZMin, &bottom)]);
    }
    // Body block (part `<name>`) and crush-zone block (part `<name>_crush`)
    // with matching y/z grids, merged at their shared face.
    let t_b = v.bumper.map_or(0.0, |b| b[0]);
    let l_c = v.crush_zone_length - t_b;
    let l_b = v.size[0] - v.crush_zone_length;
    let ny = (v.size[1] / element_size).round().max(1.0) as usize;
    let nz = (v.size[2] / element_size).round().max(1.0) as usize;
    let nxb = (l_b / element_size).round().max(1.0) as usize;
    let nxc = (l_c / v.crush_element_size.unwrap_or(element_size)).round().max(1.0) as usize;
    // Blocks from rear to front along the heading.
    let (body_x0, crush_x0, bumper_x0) = match heading {
        Heading::PlusX => (x0, x0 + l_b, x0 + l_b + l_c),
        Heading::MinusX => (x0 + l_c + t_b, x0 + t_b, x0),
    };
    // Each block's underbody faces go to a temporary set, merged into
    // `<name>_bottom` below.
    let sub = |n: &str| format!("{}_bottom_{}", v.name, n);
    let part = mesh.add_hex_block_n(&v.name, [body_x0, origin[1], origin[2]], [l_b, v.size[1], v.size[2]], [nxb, ny, nz], &[(BlockFace::ZMin, &sub("body"))]);
    let crush_name = v.crush_part_name();
    let mut names = vec![crush_name.clone()];
    let mut bottoms = vec![sub("body"), sub("crush")];
    let crush_part;
    if t_b > 0.0 {
        crush_part = mesh.add_hex_block_n(&crush_name, [crush_x0, origin[1], origin[2]], [l_c, v.size[1], v.size[2]], [nxc, ny, nz], &[(BlockFace::ZMin, &sub("crush"))]);
        let bumper_name = v.bumper_part_name();
        mesh.add_hex_block_n(&bumper_name, [bumper_x0, origin[1], origin[2]], [t_b, v.size[1], v.size[2]], [1, ny, nz], &[(face, &front), (BlockFace::ZMin, &sub("bumper"))]);
        names.push(bumper_name);
        bottoms.push(sub("bumper"));
    } else {
        crush_part = mesh.add_hex_block_n(&crush_name, [crush_x0, origin[1], origin[2]], [l_c, v.size[1], v.size[2]], [nxc, ny, nz], &[(face, &front), (BlockFace::ZMin, &sub("crush"))]);
    }
    let mut bottom_faces = Vec::new();
    for b in &bottoms {
        bottom_faces.extend(mesh.face_sets.remove(b).unwrap_or_default());
    }
    mesh.face_sets.insert(bottom, bottom_faces);
    if v.has_rail_box() {
        // Move the crush-zone elements inside the rail box to their own
        // part, and the bumper elements over it to theirs (the bumper beam
        // on the rail ends); the node sets still span the whole blocks.
        let bumper_part = mesh.part_index(&v.bumper_part_name());
        let mut carve = vec![(crush_part, v.rail_box_part_name())];
        if let Some(b) = bumper_part {
            carve.push((b, v.bumper_rails_part_name()));
        }
        for (from, to) in carve {
            let to = mesh.add_part(&to);
            for e in 0..mesh.hexes.len() {
                if mesh.hex_part[e] != from {
                    continue;
                }
                let h = mesh.hexes[e];
                let yc = h.iter().map(|n| mesh.nodes[*n][1]).sum::<f64>() / 8.0 - y_center;
                let zc = h.iter().map(|n| mesh.nodes[*n][2]).sum::<f64>() / 8.0 - z_bottom;
                if v.in_rail_box(yc, zc) {
                    mesh.hex_part[e] = to;
                }
            }
        }
    }
    mesh.merge_coincident_nodes(1e-6 * element_size);
    // Node set `<name>` spans all parts (initial velocity, delta-v).
    let mut all = mesh.node_sets[&v.name].clone();
    for n in &names {
        all.extend(mesh.node_sets[n].iter().copied());
    }
    all.sort_unstable();
    all.dedup();
    mesh.node_sets.insert(v.name.clone(), all);
    part
}

/// Assign the vehicle's crush material and initial velocity in the model.
/// The crush zone is the whole block unless `crush_zone_length` is shorter
/// (then the rear elements get the elastic body material — only meaningful
/// with separate parts, so here the whole part gets the crush material).
pub fn assign_vehicle(model: &mut Model, v: &Vehicle, heading: Heading, speed: f64) {
    if v.has_crush_zone() && model.mesh.part_index(&v.crush_part_name()).is_some() {
        model.set_material(&v.name, v.body_material());
        let crush = v.crush_material();
        let plate = v.bumper.map(|b| Material::elastic(b[2], 0.3, b[1] / (b[0] * v.frontal_area())));
        let rails = if v.has_rail_box() { model.mesh.part_index(&v.rail_box_part_name()) } else { None };
        let (f_rails, f_skin) = match rails {
            Some(rails) => {
                let skin = model.mesh.part_index(&v.crush_part_name()).unwrap();
                let (n_rails, n_skin) = (model.mesh.part_hexes(rails).len() as f64, model.mesh.part_hexes(skin).len() as f64);
                // Every x-layer of the crush zone has the same cross-section,
                // so the element count ratio is the area ratio. A box that
                // holds all of the face, or none of it, is a uniform face.
                if n_rails > 0.0 && n_skin > 0.0 { v.rail_box_factors(n_rails / (n_rails + n_skin)) } else { (1.0, 1.0) }
            }
            None => (1.0, 1.0),
        };
        if rails.is_some() {
            model.set_material(&v.rail_box_part_name(), crush.scaled(f_rails));
        }
        model.set_material(&v.crush_part_name(), crush.scaled(f_skin));
        if let Some(plate) = plate {
            if model.mesh.part_index(&v.bumper_part_name()).is_some() {
                model.set_material(&v.bumper_part_name(), plate.scaled(f_skin));
            }
            if model.mesh.part_index(&v.bumper_rails_part_name()).is_some() {
                model.set_material(&v.bumper_rails_part_name(), plate.scaled(f_rails));
            }
        }
    } else {
        model.set_material(&v.name, v.crush_material());
    }
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
        if c[d] <= lo[d] + 1e-6 * len {
            at[d] = hi[d] - 0.25 * len;
        } else if c[d] >= hi[d] - 1e-6 * len {
            at[d] = lo[d] + 0.25 * len;
        }
    }
    let name = format!("{}_rear", part);
    model.add_accelerometer_at(&name, part, at);
    name
}

/// Rigid ground: a fixed stiff slab whose top face is at height `z`,
/// spanning `x` and `y`. Creates part `<name>` and face set `<name>_face`.
/// Pair it with the vehicles' `<name>_bottom` sets ([`add_ground_contact`])
/// so a nose pushed down (underride) bottoms out on the road after its
/// ground clearance instead of diving freely.
pub fn add_ground(mesh: &mut Mesh, name: &str, z: f64, x: [f64; 2], y: [f64; 2], thickness: f64) -> usize {
    let face_name = format!("{}_face", name);
    let h = ((x[1] - x[0]).min(y[1] - y[0]) / 8.0).max(thickness);
    mesh.add_hex_block(name, [x[0], y[0], z - thickness], [x[1] - x[0], y[1] - y[0], thickness], h, &[(BlockFace::ZMax, &face_name)])
}

/// Fix the ground and let vehicle `v` (its `<name>_bottom` set) rest on it,
/// `clearance` metres above it: one-way penalty contact at the vehicle's
/// front-contact stiffness per node.
pub fn add_ground_contact(model: &mut Model, v: &Vehicle, ground: &str, clearance: f64, element_size: f64) {
    if model.mesh.part_index(ground).is_none() {
        return;
    }
    let bottom = format!("{}_bottom", v.name);
    let n_bottom = model.mesh.face_set_nodes(&bottom).map_or(1, |f| f.len());
    let n_face = model.mesh.face_set_nodes(&format!("{}_front", v.name)).map_or(n_bottom, |f| f.len());
    model.set_material(ground, Material::elastic(1e9, 0.3, 1000.0)).fix(ground);
    model.add_contact_one_way(&bottom, &format!("{}_face", ground), v.contact_stiffness_per_node(n_face, element_size), clearance + 0.3);
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
    model.mesh.part_index(part).unwrap_or_else(|| panic!("no part '{}'", part));
    // A vehicle built with a crush zone is several parts: combine them.
    let parts: Vec<usize> = ["", "_crush", "_rails", "_bumper", "_bumper_rails"].iter().filter_map(|s| model.mesh.part_index(&format!("{}{}", part, s))).collect();
    let series: Vec<(f64, f64, f64, f64)> = results
        .history
        .iter()
        .map(|(t, states)| {
            let (mut m, mut mv, mut mu) = (0.0, 0.0, 0.0);
            for &q in &parts {
                let s = &states[q];
                m += s.mass;
                mv += s.mass * s.velocity[axis];
                mu += s.mass * s.displacement[axis];
            }
            (*t, m, mv / m, mu / m)
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
    add_ground(&mut mesh, "ground", -Vehicle::GROUND_CLEARANCE, [-v.size[0] - 1.0, -0.3], [-w, w], 0.3);
    let mut model = Model::new(mesh);
    assign_vehicle(&mut model, v, Heading::PlusX, speed);
    model.set_material("wall", Material::elastic(1e9, 0.3, 1000.0)).fix("wall");
    add_ground_contact(&mut model, v, "ground", Vehicle::GROUND_CLEARANCE, element_size);
    // Contact penalty per node: much stiffer than the crush stiffness shared
    // over the front-face nodes, so the penalty springs store ≲1 % of the
    // energy (otherwise they return it as spurious restitution).
    let n_face = model.mesh.face_set_nodes(&format!("{}_front", v.name)).unwrap().len();
    let front = format!("{}_front", v.name);
    let scales = v.front_contact_scales(&model.mesh, element_size);
    model.add_contact_pair_scaled(&front, "wall_face", v.contact_stiffness_per_node(n_face, element_size), 0.3, scales, Vec::new());
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

/// Measured-vs-simulated rigid-barrier pulse on a common time grid
/// (CFC 60, m/s², m/s, m).
#[derive(Debug, Clone)]
pub struct PulseComparison {
    pub time: Vec<f64>,
    pub a_meas: Vec<f64>,
    pub a_sim: Vec<f64>,
    pub v_meas: Vec<f64>,
    pub v_sim: Vec<f64>,
    pub x_meas: Vec<f64>,
    pub x_sim: Vec<f64>,
    pub meas: crate::signal::PulseMetrics,
    pub sim: crate::signal::PulseMetrics,
}

impl PulseComparison {
    /// Both pulses are put on the measured pulse's time grid.
    pub fn new(meas: &crate::signal::Pulse, sim: &crate::signal::Pulse, mass: f64, speed: f64) -> Self {
        let t_end = meas.time.last().unwrap().min(*sim.time.last().unwrap());
        let m = meas.truncated(t_end);
        let s = if (sim.dt() - m.dt()).abs() > 1e-9 * m.dt() { sim.resampled(m.dt()).truncated(t_end) } else { sim.truncated(t_end) };
        let n = m.time.len().min(s.time.len());
        PulseComparison {
            time: m.time[..n].to_vec(),
            a_meas: m.accel[..n].to_vec(),
            a_sim: s.accel[..n].to_vec(),
            v_meas: m.velocity(speed)[..n].to_vec(),
            v_sim: s.velocity(speed)[..n].to_vec(),
            x_meas: m.displacement(speed)[..n].to_vec(),
            x_sim: s.displacement(speed)[..n].to_vec(),
            meas: crate::signal::PulseMetrics::new(&m, mass, speed),
            sim: crate::signal::PulseMetrics::new(&s, mass, speed),
        }
    }

    /// RMS difference of the filtered accelerations over the record (m/s²).
    pub fn rms_accel_error(&self) -> f64 {
        (self.a_meas.iter().zip(&self.a_sim).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / self.a_meas.len() as f64).sqrt()
    }

    /// Velocity error at a time (m/s).
    pub fn velocity_error_at(&self, t: f64) -> f64 {
        let i = ((t / (self.time[1] - self.time[0])).round() as usize).min(self.time.len() - 1);
        self.v_sim[i] - self.v_meas[i]
    }

    pub fn write_parquet(&self, path: &std::path::Path) -> Result<(), String> {
        use crate::output::parquet::{write_table, Column};
        let f = |v: &Vec<f64>| Column::F64(v.clone());
        write_table(
            path,
            "pulse",
            &[("time", f(&self.time)), ("a_meas", f(&self.a_meas)), ("a_sim", f(&self.a_sim)), ("v_meas", f(&self.v_meas)), ("v_sim", f(&self.v_sim)), ("x_meas", f(&self.x_meas)), ("x_sim", f(&self.x_sim))],
        )
    }
}

/// Sampling interval of the simulated pulse used for comparisons (s).
pub const PULSE_DT: f64 = 1e-4;
/// Record length compared (s).
pub const PULSE_END: f64 = 0.15;

/// Rigid-barrier run of `v` with a rear-seat accelerometer, returning the
/// model, results and the simulated pulse (CFC 60).
pub fn run_barrier_pulse(v: &Vehicle, element_size: f64, speed: f64, frame_steps: usize) -> (Model, Results, crate::signal::Pulse) {
    let mut model = barrier_model(v, element_size, speed, PULSE_END, frame_steps);
    let acc = add_vehicle_accelerometer(&mut model, &v.name, &format!("{}_front", v.name));
    model.settings.history_steps = 1;
    let results = crate::solver::run(&model);
    let pulse = crate::signal::Pulse::from_history(&model, &results, &acc, PULSE_DT).expect("pulse").filtered(60.0);
    (model, results, pulse)
}

/// Pulse-matching objective: velocity-history RMS error (m/s) over the
/// record, plus the end-velocity error (restitution / delta-v) and a small
/// weight on the CFC 60 acceleration RMS error (in g).
pub fn pulse_objective(cmp: &PulseComparison) -> f64 {
    let n = cmp.time.len() as f64;
    let v_rms = (cmp.v_meas.iter().zip(&cmp.v_sim).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n).sqrt();
    let v_end = (cmp.v_sim.last().unwrap() - cmp.v_meas.last().unwrap()).abs();
    v_rms + 0.5 * v_end + 0.1 * cmp.rms_accel_error() / 9.81
}

/// Fit the vehicle's tabulated force–crush curve (and crush-zone modulus)
/// so the simulated rear-seat pulse in a rigid-barrier test reproduces a
/// measured one (e.g. an NHTSA NCAP rear-seat X accelerometer; it is
/// CFC 60 filtered here).
///
/// The curve has 7 knots spread over the measured dynamic crush (the 8th
/// is densification), initialised from the smoothed measured force–crush
/// curve and mapped onto compaction as if the crush zone compacts
/// uniformly (`c = −ln(1 − x/L_c)`, which is what the block does until
/// the curve flattens; a flat plateau then localises into a lock-up front
/// that works through the zone element by element at the plateau force).
/// Then `rounds` of a pattern search on the log of each knot force, of
/// the crush-zone modulus and of the body modulus minimise
/// [`pulse_objective`], keeping the table non-decreasing from the second
/// knot on (a softening segment would localise into a shock). The 18
/// probes of a round run in parallel and the best is taken; the step
/// halves when none improves.
pub fn calibrate_pulse(v: &Vehicle, measured: &crate::signal::Pulse, speed: f64, element_size: f64, rounds: usize, verbose: bool) -> (Vehicle, PulseComparison) {
    use crate::signal::PulseMetrics;
    let meas = measured.filtered(60.0).resampled(PULSE_DT).truncated(PULSE_END);
    let mm = PulseMetrics::new(&meas, v.mass, speed);
    let n_knots = crate::material::MAX_KNOTS - 1; // one knot is the densification
    let x_max = mm.max_crush;
    let spacing = x_max * 0.95 / (n_knots - 1) as f64;
    let knots: Vec<f64> = (0..n_knots).map(|i| i as f64 * spacing).collect();
    let w = 0.5 * spacing;
    let mut tuned = v.clone();
    if tuned.force_table.as_ref().map_or(true, |t| t.len() != n_knots) {
        let mut targets: Vec<f64> = knots.iter().map(|&x| mm.mean_force((x - w).max(0.0), (x + w).min(x_max)).unwrap_or(0.0).max(1e3)).collect();
        for i in 2..targets.len() {
            targets[i] = targets[i].max(targets[i - 1]);
        }
        tuned.force_table = Some(knots.iter().zip(&targets).map(|(x, f)| [*x, *f]).collect());
        tuned.compaction_map = None;
    }

    let evaluate = |cand: &Vehicle| -> PulseComparison {
        let (_, _, sim) = run_barrier_pulse(cand, element_size, speed, 0);
        PulseComparison::new(&meas, &sim, v.mass, speed)
    };
    let objective = pulse_objective;
    let report = |tag: &str, cand: &Vehicle, cmp: &PulseComparison| {
        let sm = &cmp.sim;
        log::info!(
            "pulse calibrate {} {}: J {:.3}, peak {:.1}/{:.1} g, crush {:.0}/{:.0} mm at {:.0}/{:.0} ms, e {:.3}/{:.3}, rms Δa {:.2} g, v(end) err {:+.2} m/s, E {:.1} MPa, E_body {:.0} MPa, table kN {}",
            cand.name,
            tag,
            objective(cmp),
            sm.peak_accel / 9.81,
            mm.peak_accel / 9.81,
            sm.max_crush * 1e3,
            mm.max_crush * 1e3,
            sm.t_max_crush * 1e3,
            mm.t_max_crush * 1e3,
            sm.restitution,
            mm.restitution,
            cmp.rms_accel_error() / 9.81,
            cmp.v_sim.last().unwrap() - cmp.v_meas.last().unwrap(),
            cand.modulus / 1e6,
            cand.body_modulus / 1e6,
            cand.force_table.as_ref().unwrap().iter().map(|[_, f]| format!("{:.0}", f / 1e3)).collect::<Vec<_>>().join(" ")
        );
    };

    let mut best_cmp = evaluate(&tuned);
    let mut best_j = objective(&best_cmp);
    if verbose {
        report("start", &tuned, &best_cmp);
    }
    let mut step = 0.3_f64; // in ln(force) / ln(E)
    // Candidate `param` moved by `factor`, or None when the move is a no-op.
    let candidate = |base: &Vehicle, param: usize, factor: f64| -> Option<Vehicle> {
        let mut cand = base.clone();
        if param < n_knots {
            let t = cand.force_table.as_mut().unwrap();
            t[param][1] *= factor;
            for i in 2..t.len() {
                t[i][1] = t[i][1].max(t[i - 1][1]);
            }
            if t == base.force_table.as_ref().unwrap() {
                return None;
            }
        } else if param == n_knots {
            cand.modulus = (cand.modulus * factor).clamp(2e6, 500e6);
        } else {
            cand.body_modulus = (cand.body_modulus * factor).clamp(2e7, 5e9);
        }
        Some(cand)
    };
    for round in 0..rounds {
        // Every probe of the round (each parameter up and down) is
        // independent: run them side by side and take the best.
        let mut probes: Vec<(usize, f64, Vehicle)> = Vec::new();
        for param in 0..=n_knots + 1 {
            for dir in [1.0, -1.0] {
                if let Some(c) = candidate(&tuned, param, (dir * step).exp()) {
                    probes.push((param, dir, c));
                }
            }
        }
        let scored: Vec<(f64, PulseComparison)> = {
            use rayon::prelude::*;
            probes.par_iter().map(|(_, _, c)| {
                let cmp = evaluate(c);
                (objective(&cmp), cmp)
            }).collect()
        };
        let mut improved = false;
        if let Some((i, (j, _))) = scored.iter().enumerate().min_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap()) {
            if *j < best_j {
                let (param, dir, cand) = &probes[i];
                best_j = *j;
                tuned = cand.clone();
                best_cmp = scored[i].1.clone();
                improved = true;
                if verbose {
                    report(&format!("round {} p{}{}", round, param, if *dir > 0.0 { "+" } else { "-" }), &tuned, &best_cmp);
                }
            }
        }
        if !improved {
            step *= 0.5;
            if verbose {
                log::info!("pulse calibrate {}: round {} no improvement, step -> {:.3}", tuned.name, round, step);
            }
            if step < 0.04 {
                break;
            }
        }
    }
    (tuned, best_cmp)
}

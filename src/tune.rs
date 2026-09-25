//! `crushrs tune`: from one spec that names a vehicle's dimensions and its
//! NHTSA tests, run every calibration the data allows and write a tuned
//! [`Vehicle`] TOML that the impact commands take in place of a preset.
//!
//! ```toml
//! name = "neon"
//! mass = 1354.0                          # NCAP test weight (kg)
//! size = [4.36, 1.71, 1.35]              # length, width, height (m)
//!
//! [frontal]                              # the NCAP rigid-barrier test
//! test = 2320                            # NHTSA test number (for the record)
//! speed_mph = 35.1
//! curves = ["data/nhtsa/v02320tsv.078"]  # rear-seat / sill X channels as NHTSA ASCII exports, averaged
//! rounds = 8                             # pulse-calibration rounds (18 parallel barrier runs each)
//! crush_zone_length = 1.5                # two-part vehicle for the pulse fit (m)
//! crush_element_size = 0.1
//! crush_zone_mass = 190.0
//! bumper = [0.1, 25.0, 3.0e8]
//! modulus = 4.0e6                        # starting crush-zone / body moduli for the pulse fit
//! body_modulus = 2.7e8
//!
//! [side]                                 # FMVSS 214 MDB-test coefficients: F = L·(A + B·C)
//! a = 40.3e3                             # N per m of contact length
//! b = 1.29e6                             # N/m²
//! iterations = 8
//!
//! [rear]                                 # CRASH3 rear coefficients: F/w = A + B·C
//! a = 60.0                               # kg/cm
//! b = 8.0                                # kg/cm²
//! iterations = 8
//!
//! [rail_box]                             # where the force goes on the face, one of three ways:
//! # (a) the frontal test's load-cell wall, one force channel per cell, listed
//! #     row by row from the bottom, left to right (NHTSA NCAP: 4 × 9 cells)
//! rows = 4
//! columns = 9
//! cell_size = [0.234, 0.246]             # cell width, height (m)
//! wall_bottom = 0.08                     # height of the wall's lower edge above ground (m)
//! cells = ["data/nhtsa/v12345tsv.101", "..."]
//! coverage = 0.8                         # the box spans the central 80 % of the force laterally and vertically
//! # (b) a published average height of force
//! # ahof = 0.45                          # m; box = ±0.2 m about it, 60 % of the width, 75 % of the force
//! # (c) explicit, e.g. from `headon --calibrate-rail-boxes` against a car-to-car test
//! # width = 0.45
//! # z_range = [0.06, 0.41]
//! # force_fraction = 0.8
//! ```
//!
//! NHTSA's vehicle crash test database exports each instrumentation
//! channel as an ASCII file (`vNNNNNtsv.CCC`: time [s], value); the
//! rear-seat or sill longitudinal accelerometers (g) are the pulse
//! channels, the barrier load cells (N) the wall channels. Nothing here
//! needs a second vehicle: a pair's rail boxes can be cross-checked
//! against a car-to-car test with `headon --calibrate-rail-boxes`.

use crate::signal::Pulse;
use crate::vehicle::{calibrate, calibrate_pulse, BarrierTargets, RailBox, Vehicle};
use crate::MPH;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct TuneSpec {
    pub name: String,
    pub mass: f64,
    pub size: [f64; 3],
    pub frontal: Frontal,
    /// Restitution target of the side / rear stiffness calibrations.
    #[serde(default = "default_restitution")]
    pub restitution: f64,
    #[serde(default)]
    pub side: Option<Side>,
    #[serde(default)]
    pub rear: Option<Rear>,
    #[serde(default)]
    pub rail_box: Option<RailBoxSpec>,
}

#[derive(Debug, Deserialize)]
pub struct Frontal {
    #[serde(default)]
    pub test: Option<u32>,
    pub speed_mph: f64,
    /// Test weight if it differs from `mass`.
    #[serde(default)]
    pub mass: Option<f64>,
    pub curves: Vec<String>,
    #[serde(default = "default_rounds")]
    pub rounds: usize,
    #[serde(default)]
    pub crush_zone_length: Option<f64>,
    #[serde(default)]
    pub crush_element_size: Option<f64>,
    #[serde(default)]
    pub crush_zone_mass: Option<f64>,
    #[serde(default)]
    pub bumper: Option<[f64; 3]>,
    #[serde(default)]
    pub modulus: Option<f64>,
    #[serde(default)]
    pub body_modulus: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct Side {
    /// N per m of contact length.
    pub a: f64,
    /// N/m².
    pub b: f64,
    #[serde(default = "default_iterations")]
    pub iterations: usize,
    #[serde(default = "default_side_speed")]
    pub speed_mph: f64,
}

#[derive(Debug, Deserialize)]
pub struct Rear {
    /// CRASH3 rear A (kg/cm).
    pub a: f64,
    /// CRASH3 rear B (kg/cm²).
    pub b: f64,
    #[serde(default = "default_iterations")]
    pub iterations: usize,
    #[serde(default = "default_rear_speed")]
    pub speed_mph: f64,
}

#[derive(Debug, Deserialize, Default)]
pub struct RailBoxSpec {
    // (a) load-cell wall
    #[serde(default)]
    pub rows: usize,
    #[serde(default)]
    pub columns: usize,
    #[serde(default)]
    pub cell_size: Option<[f64; 2]>,
    #[serde(default)]
    pub wall_bottom: f64,
    #[serde(default)]
    pub cells: Vec<String>,
    #[serde(default = "default_coverage")]
    pub coverage: f64,
    // (b) average height of force
    #[serde(default)]
    pub ahof: Option<f64>,
    // (c) explicit
    #[serde(default)]
    pub width: Option<f64>,
    #[serde(default)]
    pub z_range: Option<[f64; 2]>,
    #[serde(default)]
    pub force_fraction: Option<f64>,
}

fn default_coverage() -> f64 {
    0.8
}

fn default_iterations() -> usize {
    10
}
fn default_rounds() -> usize {
    8
}
fn default_restitution() -> f64 {
    0.12
}
fn default_side_speed() -> f64 {
    45.0
}
fn default_rear_speed() -> f64 {
    35.0
}

impl TuneSpec {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        toml::from_str(&text).map_err(|e| format!("{}: {}", path.display(), e))
    }

}

/// Rail box from a load-cell wall: the mean force of each cell (row-major
/// from the bottom-left) gives the force distribution over the face; the
/// box is the central lateral span and the vertical band that each hold
/// `coverage` of the force, and carries the fraction of the force that
/// falls inside it. Cells are located on the wall from the geometry and
/// mapped onto the vehicle assuming the wall is centred on it.
pub fn rail_box_from_wall(v: &Vehicle, rows: usize, columns: usize, cell: [f64; 2], wall_bottom: f64, mean_force: &[f64], coverage: f64) -> RailBox {
    let total: f64 = mean_force.iter().sum::<f64>().max(1e-9);
    let wall_width = columns as f64 * cell[0];
    // Marginal force along each axis, then the central `coverage` span.
    let span = |n: usize, size: f64, origin: f64, weight: &dyn Fn(usize) -> f64| -> (f64, f64) {
        let w: Vec<f64> = (0..n).map(weight).collect();
        let tail = 0.5 * (1.0 - coverage) * total;
        let (mut lo, mut acc) = (0, 0.0);
        while lo + 1 < n && acc + w[lo] <= tail {
            acc += w[lo];
            lo += 1;
        }
        let (mut hi, mut acc) = (n - 1, 0.0);
        while hi > lo && acc + w[hi] <= tail {
            acc += w[hi];
            hi -= 1;
        }
        (origin + lo as f64 * size, origin + (hi + 1) as f64 * size)
    };
    let (y0, y1) = span(columns, cell[0], -0.5 * wall_width, &|c| (0..rows).map(|r| mean_force[r * columns + c]).sum());
    let (z0, z1) = span(rows, cell[1], wall_bottom, &|r| (0..columns).map(|c| mean_force[r * columns + c]).sum());
    // Symmetric about the centreline (the mesh box is), as wide as needed
    // to cover the span found.
    let width = (2.0 * y0.abs().max(y1.abs())).min(v.size[1]);
    let inside: f64 = (0..rows)
        .flat_map(|r| (0..columns).map(move |c| (r, c)))
        .filter(|(r, c)| {
            let yc = -0.5 * wall_width + (*c as f64 + 0.5) * cell[0];
            let zc = wall_bottom + (*r as f64 + 0.5) * cell[1];
            yc.abs() < 0.5 * width && zc > z0 && zc < z1
        })
        .map(|(r, c)| mean_force[r * columns + c])
        .sum();
    RailBox { width, z_range: [z0.max(0.0), z1.min(v.size[2])], force_fraction: (inside / total).clamp(0.05, 0.95) }
}

/// One line per calibration stage, for the report.
#[derive(Debug, Clone, Default)]
pub struct TuneReport {
    pub lines: Vec<String>,
}

fn read_curves(files: &[String], base: &Path) -> Result<Pulse, String> {
    if files.is_empty() {
        return Err("no curves".into());
    }
    let list = files.iter().map(|f| {
        let p = Path::new(f);
        if p.is_absolute() || p.exists() { f.clone() } else { base.join(f).to_string_lossy().into_owned() }
    }).collect::<Vec<_>>().join(",");
    Pulse::read_nhtsa_list(&list, true)
}

/// Run the pipeline. `base` resolves relative file names in the spec;
/// `element_size` is the transverse element size everything is tuned at.
pub fn tune(spec: &TuneSpec, base: &Path, element_size: f64, verbose: bool) -> Result<(Vehicle, TuneReport), String> {
    let mut report = TuneReport::default();
    let mut say = |line: String| {
        if verbose {
            log::info!("{}", line);
        }
        report.lines.push(line);
    };
    let f = &spec.frontal;
    let restitution = spec.restitution;
    if f.curves.is_empty() {
        return Err("frontal.curves is empty: name the NCAP accelerometer channel(s)".into());
    }

    // 1. Frontal: the measured barrier pulse → crush zone + body with a
    //    tabulated force–crush curve, fitted to the velocity history.
    let measured = read_curves(&f.curves, base)?;
    let speed = f.speed_mph * MPH;
    let mut p = Vehicle::new(&spec.name, f.mass.unwrap_or(spec.mass), spec.size);
    p.crush_zone_length = f.crush_zone_length.unwrap_or(0.35 * spec.size[0]);
    p.crush_element_size = f.crush_element_size.or(Some(0.1));
    p.crush_zone_mass = f.crush_zone_mass.or(Some(0.14 * spec.mass));
    p.bumper = f.bumper.or(Some([0.1, 0.02 * spec.mass, 3.0e8]));
    p.modulus = f.modulus.unwrap_or(4e6);
    p.body_modulus = f.body_modulus.unwrap_or(2.7e8);
    let (mut v, cmp) = calibrate_pulse(&p, &measured, speed, element_size, f.rounds, verbose);
    v.mass = spec.mass;
    // Bilinear summary of the fitted table (first knot, overall slope) for
    // the commands that still read `curve`.
    if let Some(t) = &v.force_table {
        let (first, last) = (t[0], t[t.len() - 1]);
        v.curve = crate::vehicle::CrushCurve { yield_force: first[1], stiffness: ((last[1] - first[1]) / last[0].max(1e-3)).max(0.0) };
    }
    say(format!(
        "frontal pulse{}: peak {:.1} vs {:.1} g, crush {:.0} vs {:.0} mm at {:.0} vs {:.0} ms, e {:.3} vs {:.3}, rms {:.2} g, Δv {:.2} vs {:.2} m/s (simulated vs measured)",
        f.test.map_or(String::new(), |t| format!(" (test {})", t)),
        cmp.sim.peak_accel / 9.81, cmp.meas.peak_accel / 9.81, cmp.sim.max_crush * 1e3, cmp.meas.max_crush * 1e3, cmp.sim.t_max_crush * 1e3, cmp.meas.t_max_crush * 1e3,
        cmp.sim.restitution, cmp.meas.restitution, cmp.rms_accel_error() / 9.81, speed - cmp.v_sim.last().unwrap(), speed - cmp.v_meas.last().unwrap()
    ));

    // 2. Side: the block turned 90°, tuned to the side curve's KW150 at 45 mph.
    if let Some(side) = &spec.side {
        v = v.with_side_coefficients(side.a, side.b);
        let profile = v.side_profile();
        let speed = side.speed_mph * MPH;
        let t = BarrierTargets::from_curve(&profile, speed, (0.025, 0.15), restitution);
        let (tuned, m) = calibrate(&profile, &t, element_size, side.iterations, verbose);
        v.side_curve = Some(tuned.curve);
        v.side_modulus = Some(tuned.modulus);
        say(format!("side: KW150 {:.0} vs {:.0} N/mm, crush {:.0} vs {:.0} mm, e {:.3}", m.kw400 / 1e3, t.kw400 / 1e3, m.max_crush * 1e3, t.max_crush * 1e3, m.restitution));
    }

    // 3. Rear: the block seen from behind, tuned to the rear curve's KW400.
    if let Some(rear) = &spec.rear {
        let (a, b) = (rear.a * 9.80665 * 100.0, rear.b * 9.80665 * 1e4); // kg/cm → N/m, kg/cm² → N/m²
        v = v.with_rear_coefficients(a, b);
        let profile = v.rear_profile().unwrap();
        let speed = rear.speed_mph * MPH;
        let t = BarrierTargets::from_curve(&profile, speed, (0.025, 0.4), restitution);
        let (tuned, m) = calibrate(&profile, &t, element_size, rear.iterations, verbose);
        v.rear_curve = Some(tuned.curve);
        v.rear_modulus = Some(tuned.modulus);
        say(format!("rear: KW400 {:.0} vs {:.0} N/mm, crush {:.0} vs {:.0} mm, e {:.3}", m.kw400 / 1e3, t.kw400 / 1e3, m.max_crush * 1e3, t.max_crush * 1e3, m.restitution));
    }

    // 4. Rail box: where the force goes on the face.
    if let Some(r) = &spec.rail_box {
        if !v.has_crush_zone() {
            return Err("rail_box needs a frontal pulse fit (a crush zone) to carry it".into());
        }
        let (rb, how) = if !r.cells.is_empty() {
            let cell = r.cell_size.ok_or("rail_box.cell_size is needed with cells")?;
            if r.rows * r.columns != r.cells.len() {
                return Err(format!("rail_box: {} cells for {} rows × {} columns", r.cells.len(), r.rows, r.columns));
            }
            let mut mean = Vec::with_capacity(r.cells.len());
            for file in &r.cells {
                let p = read_curves(std::slice::from_ref(file), base)?;
                // Mean force over the loading phase (the record up to peak crush
                // is what the pulse-calibrated block reproduces); channels in N.
                let n = p.accel.len().max(1);
                mean.push(p.accel.iter().map(|f| f.abs()).sum::<f64>() / n as f64);
            }
            (rail_box_from_wall(&v, r.rows, r.columns, cell, r.wall_bottom, &mean, r.coverage), format!("from {} load cells", r.cells.len()))
        } else if let Some(ahof) = r.ahof {
            (RailBox { width: r.width.unwrap_or(0.6 * v.size[1]), z_range: [(ahof - 0.2).max(0.0), (ahof + 0.2).min(v.size[2])], force_fraction: r.force_fraction.unwrap_or(0.75) }, format!("about AHOF {:.2} m", ahof))
        } else {
            match (r.width, r.z_range, r.force_fraction) {
                (Some(w), Some(z), Some(f)) => (RailBox { width: w, z_range: z, force_fraction: f }, "as given".to_string()),
                _ => return Err("rail_box needs cells (+ rows, columns, cell_size, wall_bottom), ahof, or width + z_range + force_fraction".into()),
            }
        };
        v.rail_box = Some(rb);
        say(format!("rail box {}: {:.2} m wide, z {:.2}–{:.2} m, {:.0} % of the force", how, rb.width, rb.z_range[0], rb.z_range[1], 100.0 * rb.force_fraction));
    }
    Ok((v, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wall_reduction_finds_the_rails() {
        // A 4 × 9 NCAP wall, cells 0.234 × 0.246 m from 0.08 m up, with all
        // the force in row 2 (0.57–0.82 m) on columns 2 and 6 (the rails)
        // plus a little everywhere.
        let v = Vehicle::dodge_neon_1996_pulse();
        let (rows, cols) = (4, 9);
        let mut f = vec![0.25; rows * cols];
        f[1 * cols + 2] = 40.0;
        f[1 * cols + 6] = 40.0;
        let total = 80.0 + 0.25 * 34.0;
        let rb = rail_box_from_wall(&v, rows, cols, [0.234, 0.246], 0.08, &f, 0.8);
        // Row 1 spans 0.326–0.572 m; columns 2..=6 span ±0.585 m.
        assert!((rb.z_range[0] - 0.326).abs() < 1e-6 && (rb.z_range[1] - 0.572).abs() < 1e-6, "{:?}", rb);
        assert!((rb.width - 1.17).abs() < 1e-6, "{:?}", rb);
        // The rails and the 3 background cells between them fall inside.
        assert!((rb.force_fraction - (80.0 + 0.75) / total).abs() < 1e-6, "{:?}", rb);
    }
}

//! Vehicle-to-vehicle frontal impacts: the model builder shared by the
//! `headon` command and the rail-box calibration, and the calibration itself.
//!
//! A car-to-car test is what a rigid-barrier calibration cannot see: how
//! the force is distributed over the face. Both vehicles keep their barrier
//! calibration whatever their [`RailBox`] is (the rail box only redistributes the
//! same force–crush curve over the face), so the rail boxes can be fitted to a
//! measured car-to-car test without touching the barrier fits.

use crate::mesh::Mesh;
use crate::model::Model;
use crate::signal::Pulse;
use crate::solver::Results;
use crate::vehicle::{add_ground, add_ground_contact, add_vehicle, add_vehicle_accelerometer, assign_vehicle, pulse_objective, RailBox, Heading, PulseComparison, Vehicle, PULSE_DT, PULSE_END};

/// Setup of a frontal vehicle-to-vehicle impact: A heads +x, B heads −x,
/// noses `2·gap` apart at x = 0.
#[derive(Clone, Debug)]
pub struct HeadOn {
    /// Speeds (m/s) of A and B.
    pub speed_a: f64,
    pub speed_b: f64,
    /// Lateral offset of B (m); 0 = full overlap.
    pub offset: f64,
    /// Rigid road this far below the underbodies (m); `None` = no road.
    pub clearance: Option<f64>,
    /// Transverse element size (m).
    pub element_size: f64,
    pub end_time: f64,
}

impl HeadOn {
    pub fn new(speed_a: f64, speed_b: f64) -> Self {
        HeadOn { speed_a, speed_b, offset: 0.0, clearance: Some(Vehicle::GROUND_CLEARANCE), element_size: Vehicle::TUNED_ELEMENT_SIZE, end_time: PULSE_END }
    }

    /// Build the model: both vehicles (with `_a` / `_b` name suffixes
    /// already applied by the caller), the road, the front-to-front contact
    /// and a rear-seat accelerometer per vehicle (names returned).
    pub fn build(&self, a: &Vehicle, b: &Vehicle) -> (Model, String, String) {
        let h = self.element_size;
        let gap = 0.005;
        let mut mesh = Mesh::new();
        add_vehicle(&mut mesh, a, -gap, 0.0, 0.0, Heading::PlusX, h);
        add_vehicle(&mut mesh, b, gap, self.offset, 0.0, Heading::MinusX, h);
        if let Some(c) = self.clearance {
            let w = a.size[1].max(b.size[1]) + 2.0 * self.offset.abs();
            add_ground(&mut mesh, "ground", -c, [-a.size[0] - 3.0, b.size[0] + 3.0], [-w, w], 0.3);
        }
        let mut model = Model::new(mesh);
        assign_vehicle(&mut model, a, Heading::PlusX, self.speed_a);
        assign_vehicle(&mut model, b, Heading::MinusX, self.speed_b);
        if let Some(c) = self.clearance {
            add_ground_contact(&mut model, a, "ground", c, h);
            add_ground_contact(&mut model, b, "ground", c, h);
        }
        let (fa, fb) = (format!("{}_front", a.name), format!("{}_front", b.name));
        let n_face = model.mesh.face_set_nodes(&fa).unwrap().len().min(model.mesh.face_set_nodes(&fb).unwrap().len());
        let (sa, sb) = (a.front_contact_scales(&model.mesh, h), b.front_contact_scales(&model.mesh, h));
        model.add_contact_pair_scaled(&fa, &fb, a.contact_stiffness_per_node(n_face, h).min(b.contact_stiffness_per_node(n_face, h)), 0.3, sa, sb);
        model.settings.end_time = self.end_time;
        let acc_a = add_vehicle_accelerometer(&mut model, &a.name, &fa);
        let acc_b = add_vehicle_accelerometer(&mut model, &b.name, &fb);
        model.settings.history_steps = 1;
        (model, acc_a, acc_b)
    }

    /// Run the impact and return the two rear-seat pulses (CFC 60, each
    /// vehicle's own forward direction positive) with the model and results.
    pub fn run(&self, a: &Vehicle, b: &Vehicle) -> (Model, Results, Pulse, Pulse) {
        let (model, acc_a, acc_b) = self.build(a, b);
        let results = crate::solver::run(&model);
        let pa = Pulse::from_history(&model, &results, &acc_a, PULSE_DT).expect("pulse").filtered(60.0);
        let mut pb = Pulse::from_history(&model, &results, &acc_b, PULSE_DT).expect("pulse");
        pb.accel.iter_mut().for_each(|x| *x = -*x);
        let pb = pb.filtered(60.0);
        (model, results, pa, pb)
    }
}

/// Rail-box parameters as a point in the unit cube, for the optimizer:
/// `[width / vehicle width, z-centre / height, z-extent / height, force
/// fraction]`. The bounds keep the box a stiff structure low in the face:
/// 25–95 % of the width, centred between 15 % and 50 % of the height
/// (every measured average height of force lies below the centre of
/// gravity), 15–60 % of the height tall, carrying 50–95 % of the force.
fn rail_box_from_unit(v: &Vehicle, u: &[f64]) -> RailBox {
    let (w, hgt) = (v.size[1], v.size[2]);
    let width = (0.25 + 0.7 * u[0]) * w;
    let zc = (0.15 + 0.35 * u[1]) * hgt;
    let dz = (0.15 + 0.45 * u[2]) * hgt;
    let z0 = (zc - 0.5 * dz).max(0.0);
    let z1 = (zc + 0.5 * dz).min(hgt);
    RailBox { width, z_range: [z0, z1], force_fraction: 0.5 + 0.45 * u[3] }
}

fn unit_from_rail_box(v: &Vehicle, c: &RailBox) -> [f64; 4] {
    let (w, hgt) = (v.size[1], v.size[2]);
    let zc = 0.5 * (c.z_range[0] + c.z_range[1]) / hgt;
    let dz = (c.z_range[1] - c.z_range[0]) / hgt;
    [((c.width / w - 0.25) / 0.7).clamp(0.0, 1.0), ((zc - 0.15) / 0.35).clamp(0.0, 1.0), ((dz - 0.15) / 0.45).clamp(0.0, 1.0), ((c.force_fraction - 0.5) / 0.45).clamp(0.0, 1.0)]
}

/// xorshift64* — a small deterministic RNG so calibrations are repeatable.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.uniform() * n as f64) as usize % n
    }
}

/// Result of [`calibrate_rail_boxes`].
#[derive(Clone, Debug)]
pub struct RailBoxFit {
    pub a: Vehicle,
    pub b: Vehicle,
    pub objective: f64,
    pub cmp_a: PulseComparison,
    pub cmp_b: PulseComparison,
    pub evaluations: usize,
}

/// Fit the rail boxes of both vehicles to a measured car-to-car
/// test: the two rear-seat pulses of `setup` are compared with
/// `measured_a` / `measured_b` (each vehicle's forward direction positive)
/// through [`pulse_objective`], summed, and the eight rail box parameters
/// (width, height-band centre and extent, force fraction, per vehicle)
/// are searched with differential evolution (rand/1/bin, population
/// `population`, `generations` generations; about `population ×
/// (generations + 1)` impact runs) in a unit cube, seeded with the
/// vehicles' current rail boxes. A global method because the objective is
/// piecewise constant in the geometry (the box snaps to the element grid)
/// and noisy (the lock-up front is chaotic). The barrier calibration of
/// both vehicles is unchanged by construction.
pub fn calibrate_rail_boxes(a: &Vehicle, b: &Vehicle, setup: &HeadOn, measured_a: &Pulse, measured_b: &Pulse, population: usize, generations: usize, verbose: bool) -> RailBoxFit {
    let (ma, mb) = (a.mass, b.mass);
    let meas_a = measured_a.filtered(60.0).resampled(PULSE_DT).truncated(setup.end_time);
    let meas_b = measured_b.filtered(60.0).resampled(PULSE_DT).truncated(setup.end_time);
    let base = |v: &Vehicle| -> Vehicle {
        let mut v = v.clone();
        if v.rail_box.is_none() {
            v.rail_box = Some(RailBox { width: 0.5 * v.size[1], z_range: [0.2 * v.size[2], 0.6 * v.size[2]], force_fraction: 0.7 });
        }
        v
    };
    let (a0, b0) = (base(a), base(b));
    let dims = 8;
    let with = |u: &[f64]| -> (Vehicle, Vehicle) {
        let mut va = a0.clone();
        let mut vb = b0.clone();
        va.rail_box = Some(rail_box_from_unit(&va, &u[..4]));
        vb.rail_box = Some(rail_box_from_unit(&vb, &u[4..]));
        (va, vb)
    };
    let evaluations = std::cell::Cell::new(0usize);
    let evaluate = |u: &[f64]| -> (f64, PulseComparison, PulseComparison) {
        let (va, vb) = with(u);
        let (_, _, pa, pb) = setup.run(&va, &vb);
        let ca = PulseComparison::new(&meas_a, &pa, ma, setup.speed_a);
        let cb = PulseComparison::new(&meas_b, &pb, mb, setup.speed_b);
        evaluations.set(evaluations.get() + 1);
        (pulse_objective(&ca) + pulse_objective(&cb), ca, cb)
    };
    let describe = |u: &[f64]| -> String {
        let (va, vb) = with(u);
        let f = |v: &Vehicle| {
            let c = v.rail_box.unwrap();
            format!("{} rail box {:.2} m wide, z {:.2}–{:.2} m, {:.0} % of the force", v.name, c.width, c.z_range[0], c.z_range[1], 100.0 * c.force_fraction)
        };
        format!("{}; {}", f(&va), f(&vb))
    };

    let mut rng = Rng(0x9E3779B97F4A7C15);
    let np = population.max(4);
    let mut pop: Vec<Vec<f64>> = Vec::with_capacity(np);
    let mut seed = unit_from_rail_box(&a0, &a0.rail_box.unwrap()).to_vec();
    seed.extend(unit_from_rail_box(&b0, &b0.rail_box.unwrap()));
    pop.push(seed);
    while pop.len() < np {
        pop.push((0..dims).map(|_| rng.uniform()).collect());
    }
    let mut scored: Vec<(f64, PulseComparison, PulseComparison)> = pop.iter().map(|u| evaluate(u)).collect();
    let best_index = |s: &[(f64, PulseComparison, PulseComparison)]| s.iter().enumerate().min_by(|x, y| x.1 .0.partial_cmp(&y.1 .0).unwrap()).map(|(i, _)| i).unwrap();
    let mut best = best_index(&scored);
    if verbose {
        log::info!("rail-box calibrate: initial population of {}: best J {:.3} ({})", np, scored[best].0, describe(&pop[best]));
    }
    let (f_weight, cr) = (0.7, 0.9);
    for gen in 0..generations {
        for i in 0..np {
            let (r1, r2, r3) = loop {
                let r = [rng.below(np), rng.below(np), rng.below(np)];
                if r[0] != i && r[1] != i && r[2] != i && r[0] != r[1] && r[0] != r[2] && r[1] != r[2] {
                    break (r[0], r[1], r[2]);
                }
            };
            let j_rand = rng.below(dims);
            let trial: Vec<f64> = (0..dims)
                .map(|j| {
                    if j == j_rand || rng.uniform() < cr {
                        let v = pop[r1][j] + f_weight * (pop[r2][j] - pop[r3][j]);
                        // Reflect into the unit cube.
                        let v = if v < 0.0 { -v } else if v > 1.0 { 2.0 - v } else { v };
                        v.clamp(0.0, 1.0)
                    } else {
                        pop[i][j]
                    }
                })
                .collect();
            let s = evaluate(&trial);
            if s.0 <= scored[i].0 {
                pop[i] = trial;
                scored[i] = s;
            }
        }
        best = best_index(&scored);
        if verbose {
            let mean = scored.iter().map(|s| s.0).sum::<f64>() / np as f64;
            log::info!("rail-box calibrate: generation {} ({} runs): best J {:.3}, mean {:.3} — {}", gen, evaluations.get(), scored[best].0, mean, describe(&pop[best]));
        }
    }
    let (va, vb) = with(&pop[best]);
    let (objective, cmp_a, cmp_b) = scored.swap_remove(best);
    RailBoxFit { a: va, b: vb, objective, cmp_a, cmp_b, evaluations: evaluations.get() }
}

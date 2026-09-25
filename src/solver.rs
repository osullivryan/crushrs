//! Explicit central-difference solver.

use crate::contact::ContactRuntime;
use crate::kernel::Elements;
use crate::model::Model;
use log::{debug, info, warn};

/// Deformed-mesh snapshot.
#[derive(Debug, Clone, Default)]
pub struct Frame {
    pub time: f64,
    /// Nodal displacement, 3 per node.
    pub displacement: Vec<f32>,
    /// Equivalent plastic strain / compaction per element.
    pub plastic_strain: Vec<f32>,
    /// Cauchy stress (Voigt) per element.
    pub stress: Vec<[f32; 6]>,
    pub eroded: Vec<bool>,
}

/// Mass-weighted state of one part at one time.
#[derive(Debug, Clone)]
pub struct PartState {
    pub part: usize,
    pub mass: f64,
    pub velocity: [f64; 3],
    pub displacement: [f64; 3],
}

#[derive(Debug, Clone, Default)]
pub struct Results {
    pub time: f64,
    pub steps: usize,
    pub displacement: Vec<[f64; 3]>,
    pub velocity: Vec<[f64; 3]>,
    pub masses: Vec<f64>,
    pub frames: Vec<Frame>,
    /// (time, per-part states) every `history_steps`.
    pub history: Vec<(f64, Vec<PartState>)>,
    pub eroded_elements: usize,
    pub wall_time: std::time::Duration,
    pub force_time: std::time::Duration,
    pub contact_time: std::time::Duration,
}

impl Results {
    /// Mass and mass-weighted mean velocity of a set of nodes.
    pub fn mean_velocity(&self, nodes: &[usize]) -> (f64, [f64; 3]) {
        Model::mean_velocity(&self.masses, &self.velocity, nodes)
    }

    /// Rigid-body angular velocity fit of a set of nodes (rad/s).
    pub fn mean_angular_velocity(&self, positions: &[[f64; 3]], nodes: &[usize]) -> [f64; 3] {
        use nalgebra::{Matrix3, Vector3};
        let (m, v_mean) = self.mean_velocity(nodes);
        if m <= 0.0 {
            return [0.0; 3];
        }
        let v_mean = Vector3::from(v_mean);
        let x = |n: usize| Vector3::from(positions[n]) + Vector3::from(self.displacement[n]);
        let mut c = Vector3::zeros();
        for &n in nodes {
            c += self.masses[n] * x(n);
        }
        c /= m;
        let mut l = Vector3::zeros();
        let mut inertia = Matrix3::zeros();
        for &n in nodes {
            let r = x(n) - c;
            l += self.masses[n] * r.cross(&(Vector3::from(self.velocity[n]) - v_mean));
            inertia += self.masses[n] * (Matrix3::identity() * r.norm_squared() - r * r.transpose());
        }
        let w = inertia.try_inverse().map(|inv| inv * l).unwrap_or_else(Vector3::zeros);
        [w.x, w.y, w.z]
    }
}

fn part_states(model: &Model, masses: &[f64], u: &[f64], v: &[f64]) -> Vec<PartState> {
    (0..model.mesh.parts.len())
        .map(|p| {
            let nodes = model.mesh.part_nodes(p);
            let mut m = 0.0;
            let mut mv = [0.0; 3];
            let mut mu = [0.0; 3];
            for n in nodes {
                m += masses[n];
                for d in 0..3 {
                    mv[d] += masses[n] * v[3 * n + d];
                    mu[d] += masses[n] * u[3 * n + d];
                }
            }
            let inv = if m > 0.0 { 1.0 / m } else { 0.0 };
            PartState { part: p, mass: m, velocity: [mv[0] * inv, mv[1] * inv, mv[2] * inv], displacement: [mu[0] * inv, mu[1] * inv, mu[2] * inv] }
        })
        .collect()
}

/// Run the explicit solve.
pub fn run(model: &Model) -> Results {
    let start = std::time::Instant::now();
    let s = &model.settings;
    let n_nodes = model.mesh.nodes.len();
    let n = 3 * n_nodes;
    let threads = if s.threads == 0 { rayon::current_num_threads() } else { s.threads };

    let mut elements = Elements::build(model, s.simd);
    let masses = model.lumped_masses(s.hourglass);
    let mut mass_dof = vec![0.0; n];
    let mut inv_mass = vec![0.0; n];
    for i in 0..n_nodes {
        for d in 0..3 {
            mass_dof[3 * i + d] = masses[i];
            inv_mass[3 * i + d] = if masses[i] > 0.0 { 1.0 / masses[i] } else { 0.0 };
        }
    }
    let _ = &mass_dof;
    let mut contacts: Vec<ContactRuntime> = model.contacts.iter().map(|c| ContactRuntime::new(c, &model.mesh.faces, &model.mesh.nodes)).collect();
    let fixed: Vec<usize> = model.fixed_nodes.iter().flat_map(|n| [3 * n, 3 * n + 1, 3 * n + 2]).collect();

    // Time step.
    let mut u = vec![0.0; n];
    let stable = elements.stable_time_step(&u).unwrap_or_else(|| model.mesh.min_edge_length() / (3.0_f64.sqrt() * elements.material.iter().map(|m| m.dilatational_wave_speed()).fold(0.0, f64::max)));
    let mut dt = s.time_step.unwrap_or(stable * s.dt_scale);
    let adaptive = s.time_step.is_none() && s.adaptive_check_steps > 0;
    debug!("stable dt {:.3e} s, using {:.3e} s (adaptive: {}, simd: {}, threads: {})", stable, dt, adaptive, elements.uses_simd(), threads);
    if elements.uses_simd() {
        debug!("f32 lane kernel, avx2 = {}", crate::kernel_simd::is_avx2());
    }

    let mut v: Vec<f64> = model.initial_velocity.iter().flat_map(|w| [w[0], w[1], w[2]]).collect();
    let mut v_half = vec![0.0; n];
    let mut f_int = vec![0.0; n];
    let mut f_ext = vec![0.0; n];
    let mut results = Results { masses: masses.clone(), ..Default::default() };
    let mut time_force = std::time::Duration::ZERO;
    let mut time_contact = std::time::Duration::ZERO;

    let t0 = std::time::Instant::now();
    elements.compute_forces(&u, &mut f_int, threads);
    time_force += t0.elapsed();

    let mut t = 0.0;
    let mut step = 0usize;
    let mut eroded_total = 0usize;
    let capture = |elements: &mut Elements, u: &[f64], t: f64, results: &mut Results| {
        elements.sync_states();
        results.frames.push(Frame {
            time: t,
            displacement: u.iter().map(|x| *x as f32).collect(),
            plastic_strain: elements.plastic_strain().iter().map(|x| *x as f32).collect(),
            stress: elements.cauchy_stress(u).iter().map(|s| std::array::from_fn(|k| s[k] as f32)).collect(),
            eroded: elements.eroded.clone(),
        });
    };
    if s.frame_steps > 0 {
        capture(&mut elements, &u, t, &mut results);
    }
    if s.history_steps > 0 {
        results.history.push((t, part_states(model, &masses, &u, &v)));
    }
    let mut dt_cfl = dt;
    while t < s.end_time {
        if adaptive && step % s.adaptive_check_steps == 0 {
            if let Some(dt_stable) = elements.stable_time_step(&u) {
                dt_cfl = s.dt_scale * dt_stable;
            }
        }
        dt = dt_cfl.min(s.end_time - t).max(1e-15);
        if step % 20 == 0 && u.iter().any(|x| !x.is_finite()) {
            panic!("displacement became non-finite at step {} (t = {:.3e})", step, t);
        }

        // External forces: contact.
        let tc = std::time::Instant::now();
        f_ext.fill(0.0);
        for c in &mut contacts {
            c.apply(&model.mesh.nodes, &u, &mut f_ext);
        }
        time_contact += tc.elapsed();

        // Half-step velocity and position update, in place.
        for k in 0..n {
            let a = (f_ext[k] - f_int[k]) * inv_mass[k];
            v_half[k] = v[k] + 0.5 * dt * a;
            u[k] += dt * v_half[k];
        }
        for &k in &fixed {
            u[k] = 0.0;
            v[k] = 0.0;
            v_half[k] = 0.0;
        }

        // Internal forces at the new configuration, velocity to full step.
        let tf = std::time::Instant::now();
        f_int.fill(0.0);
        let eroded = elements.compute_forces(&u, &mut f_int, threads);
        time_force += tf.elapsed();
        if eroded > 0 {
            eroded_total += eroded;
            warn!("{} element(s) inverted and were eroded at t = {:.4} s", eroded, t);
        }
        for k in 0..n {
            let a = (f_ext[k] - f_int[k]) * inv_mass[k];
            v[k] = (v_half[k] + 0.5 * dt * a) * s.velocity_damping;
        }
        for &k in &fixed {
            v[k] = 0.0;
        }

        step += 1;
        t += dt;
        if s.frame_steps > 0 && step % s.frame_steps == 0 {
            capture(&mut elements, &u, t, &mut results);
        }
        if s.history_steps > 0 && step % s.history_steps == 0 {
            results.history.push((t, part_states(model, &masses, &u, &v)));
        }
    }
    if s.frame_steps > 0 && step % s.frame_steps != 0 {
        capture(&mut elements, &u, t, &mut results);
    }
    if s.history_steps > 0 && step % s.history_steps != 0 {
        results.history.push((t, part_states(model, &masses, &u, &v)));
    }

    results.time = t;
    results.steps = step;
    results.displacement = (0..n_nodes).map(|i| [u[3 * i], u[3 * i + 1], u[3 * i + 2]]).collect();
    results.velocity = (0..n_nodes).map(|i| [v[3 * i], v[3 * i + 1], v[3 * i + 2]]).collect();
    results.eroded_elements = eroded_total;
    results.wall_time = start.elapsed();
    results.force_time = time_force;
    results.contact_time = time_contact;
    info!("{} steps in {:.2?} (forces {:.2?}, contact {:.2?}); {} elements eroded", step, results.wall_time, time_force, time_contact, eroded_total);
    for (p, part) in model.mesh.parts.iter().enumerate() {
        let nodes = model.mesh.part_nodes(p);
        let (m, v1) = results.mean_velocity(&nodes);
        let (_, v0) = Model::mean_velocity(&masses, &model.initial_velocity, &nodes);
        let dv = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        info!("part '{}': mass {:.1} kg, delta-v [{:+.3}, {:+.3}, {:+.3}] m/s (|dv| = {:.3})", part.name, m, dv[0], dv[1], dv[2], (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt());
    }
    results
}

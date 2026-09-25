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
    /// Net force / mass.
    pub acceleration: [f64; 3],
    pub kinetic_energy: f64,
}

/// One accelerometer sample: a node's motion in global axes and, if the
/// accelerometer has a frame, in that body-fixed frame (otherwise the local
/// columns repeat the global ones).
#[derive(Debug, Clone, Default)]
pub struct NodeHistory {
    pub time: Vec<f64>,
    pub step: Vec<usize>,
    /// Index into `Model::accelerometers`.
    pub accelerometer: Vec<usize>,
    pub node: Vec<usize>,
    pub position: Vec<[f64; 3]>,
    pub displacement: Vec<[f64; 3]>,
    pub velocity: Vec<[f64; 3]>,
    pub acceleration: Vec<[f64; 3]>,
    pub local_displacement: Vec<[f64; 3]>,
    pub local_velocity: Vec<[f64; 3]>,
    pub local_acceleration: Vec<[f64; 3]>,
}

/// Body-fixed frame of an accelerometer at one time.
#[derive(Debug, Clone, Default)]
pub struct FrameHistory {
    pub time: Vec<f64>,
    pub step: Vec<usize>,
    pub accelerometer: Vec<usize>,
    pub origin: Vec<[f64; 3]>,
    /// Local unit axes (rows) in global components.
    pub axes: Vec<[[f64; 3]; 3]>,
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
    /// Accelerometer samples every `history_steps`.
    pub node_history: NodeHistory,
    pub frame_history: FrameHistory,
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

fn part_states(part_nodes: &[Vec<usize>], masses: &[f64], u: &[f64], v: &[f64], a: &[f64]) -> Vec<PartState> {
    part_nodes
        .iter()
        .enumerate()
        .map(|(p, nodes)| {
            let mut m = 0.0;
            let mut mv = [0.0; 3];
            let mut mu = [0.0; 3];
            let mut ma = [0.0; 3];
            let mut ke = 0.0;
            for &n in nodes {
                m += masses[n];
                for d in 0..3 {
                    mv[d] += masses[n] * v[3 * n + d];
                    mu[d] += masses[n] * u[3 * n + d];
                    ma[d] += masses[n] * a[3 * n + d];
                    ke += 0.5 * masses[n] * v[3 * n + d] * v[3 * n + d];
                }
            }
            let inv = if m > 0.0 { 1.0 / m } else { 0.0 };
            PartState {
                part: p,
                mass: m,
                velocity: [mv[0] * inv, mv[1] * inv, mv[2] * inv],
                displacement: [mu[0] * inv, mu[1] * inv, mu[2] * inv],
                acceleration: [ma[0] * inv, ma[1] * inv, ma[2] * inv],
                kinetic_energy: ke,
            }
        })
        .collect()
}

fn rotate(axes: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| axes[i][0] * v[0] + axes[i][1] * v[1] + axes[i][2] * v[2])
}

fn sample_accelerometers(model: &Model, t: f64, step: usize, u: &[f64], v: &[f64], a: &[f64], nodes: &mut NodeHistory, frames: &mut FrameHistory) {
    let x = |n: usize| -> [f64; 3] { std::array::from_fn(|d| model.mesh.nodes[n][d] + u[3 * n + d]) };
    let vec3 = |w: &[f64], n: usize| -> [f64; 3] { [w[3 * n], w[3 * n + 1], w[3 * n + 2]] };
    for (k, acc) in model.accelerometers.iter().enumerate() {
        let axes = acc.frame.map(|f| {
            let axes = f.axes(x);
            frames.time.push(t);
            frames.step.push(step);
            frames.accelerometer.push(k);
            frames.origin.push(x(f.origin));
            frames.axes.push(axes);
            axes
        });
        for &n in &acc.nodes {
            let (un, vn, an) = (vec3(u, n), vec3(v, n), vec3(a, n));
            nodes.time.push(t);
            nodes.step.push(step);
            nodes.accelerometer.push(k);
            nodes.node.push(n);
            nodes.position.push(x(n));
            nodes.displacement.push(un);
            nodes.velocity.push(vn);
            nodes.acceleration.push(an);
            match &axes {
                Some(r) => {
                    nodes.local_displacement.push(rotate(r, un));
                    nodes.local_velocity.push(rotate(r, vn));
                    nodes.local_acceleration.push(rotate(r, an));
                }
                None => {
                    nodes.local_displacement.push(un);
                    nodes.local_velocity.push(vn);
                    nodes.local_acceleration.push(an);
                }
            }
        }
    }
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
    let element_mass: Vec<f64> = (0..model.mesh.hexes.len()).map(|e| elements.volume[e] * model.materials[model.mesh.hex_part[e]].density).collect();
    let mut added_mass = vec![0.0; n_nodes];
    let mut inv_mass = vec![0.0; n];
    let set_inv_mass = |inv_mass: &mut [f64], added: &[f64]| {
        for i in 0..n_nodes {
            let m = masses[i] + added[i];
            for d in 0..3 {
                inv_mass[3 * i + d] = if m > 0.0 { 1.0 / m } else { 0.0 };
            }
        }
    };
    set_inv_mass(&mut inv_mass, &added_mass);
    let mut contacts: Vec<ContactRuntime> = model.contacts.iter().map(|c| ContactRuntime::new(c, &model.mesh.faces, &model.mesh.nodes)).collect();
    let fixed: Vec<usize> = model.fixed_nodes.iter().flat_map(|n| [3 * n, 3 * n + 1, 3 * n + 2]).collect();

    // Time step: element stability, and the penalty springs' own limit
    // 2·√(m/k) for the lightest contacting node (both scaled by dt_scale).
    let mut u = vec![0.0; n];
    let contact_dt = model
        .contacts
        .iter()
        .flat_map(|c| c.nodes.iter().map(move |&nd| (nd, c.stiffness)))
        .filter(|(nd, k)| masses[*nd] > 0.0 && *k > 0.0)
        .map(|(nd, k)| 2.0 * (masses[nd] / k).sqrt())
        .fold(f64::INFINITY, f64::min);
    let stable = elements.stable_time_step(&u).unwrap_or_else(|| model.mesh.min_edge_length() / (3.0_f64.sqrt() * elements.material.iter().map(|m| m.dilatational_wave_speed()).fold(0.0, f64::max)));
    let stable = stable.min(contact_dt);
    let dt_floor = if s.mass_scaling > 0.0 { s.mass_scaling * stable } else { 0.0 };
    let mut added_mass_total = 0.0_f64;
    let mut dt = s.time_step.unwrap_or(stable * s.dt_scale);
    if contact_dt < stable * 1.0001 {
        debug!("time step limited by contact penalty stiffness: {:.3e} s", contact_dt);
    }
    let adaptive = s.time_step.is_none() && s.adaptive_check_steps > 0;
    debug!("stable dt {:.3e} s, using {:.3e} s (adaptive: {}, simd: {}, threads: {})", stable, dt, adaptive, elements.uses_simd(), threads);
    if elements.uses_simd() {
        debug!("f32 lane kernel, avx2 = {}", crate::kernel_simd::is_avx2());
    }

    let mut v: Vec<f64> = model.initial_velocity.iter().flat_map(|w| [w[0], w[1], w[2]]).collect();
    let mut v_half = vec![0.0; n];
    let mut f_int = vec![0.0; n];
    let mut f_ext = vec![0.0; n];
    let mut acc = vec![0.0; n];
    let part_nodes: Vec<Vec<usize>> = (0..model.mesh.parts.len()).map(|p| model.mesh.part_nodes(p)).collect();
    for (k, a) in model.accelerometers.iter().enumerate() {
        for m in a.nodes.iter().copied().chain(a.frame.iter().flat_map(|f| [f.origin, f.x_axis, f.plane])) {
            assert!(m < n_nodes, "accelerometer {} '{}': node {} out of range", k, a.name, m);
        }
    }
    let mut results = Results { masses: masses.clone(), ..Default::default() };
    let mut time_force = std::time::Duration::ZERO;
    let mut time_contact = std::time::Duration::ZERO;

    let t0 = std::time::Instant::now();
    elements.compute_forces(&u, &mut f_int, threads);
    time_force += t0.elapsed();
    for c in &mut contacts {
        c.apply(&model.mesh.nodes, &u, &v, &masses, dt, &mut f_ext);
    }
    for k in 0..n {
        acc[k] = (f_ext[k] - f_int[k]) * inv_mass[k];
    }

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
    let record = |t: f64, step: usize, u: &[f64], v: &[f64], acc: &[f64], results: &mut Results| {
        results.history.push((t, part_states(&part_nodes, &masses, u, v, acc)));
        sample_accelerometers(model, t, step, u, v, acc, &mut results.node_history, &mut results.frame_history);
    };
    if s.history_steps > 0 {
        record(t, 0, &u, &v, &acc, &mut results);
    }
    let mut dt_cfl = dt;
    while t < s.end_time {
        if adaptive && step % s.adaptive_check_steps == 0 {
            if let Some(dt_stable) = elements.stable_time_step(&u) {
                let mut dt_use = dt_stable;
                if dt_floor > 0.0 && dt_stable < dt_floor {
                    let total = elements.mass_scaling(&u, dt_floor, &element_mass, &mut added_mass);
                    set_inv_mass(&mut inv_mass, &added_mass);
                    added_mass_total = added_mass_total.max(total);
                    dt_use = dt_floor;
                }
                dt_cfl = s.dt_scale * dt_use.min(contact_dt);
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
            c.apply(&model.mesh.nodes, &u, &v, &masses, dt, &mut f_ext);
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
            acc[k] = a;
        }
        for &k in &fixed {
            v[k] = 0.0;
            acc[k] = 0.0;
        }

        step += 1;
        t += dt;
        if s.frame_steps > 0 && step % s.frame_steps == 0 {
            capture(&mut elements, &u, t, &mut results);
        }
        if s.history_steps > 0 && step % s.history_steps == 0 {
            record(t, step, &u, &v, &acc, &mut results);
        }
    }
    if s.frame_steps > 0 && step % s.frame_steps != 0 {
        capture(&mut elements, &u, t, &mut results);
    }
    if s.history_steps > 0 && step % s.history_steps != 0 {
        record(t, step, &u, &v, &acc, &mut results);
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
    if added_mass_total > 0.0 {
        let total_mass: f64 = masses.iter().sum();
        info!("mass scaling added up to {:.3} kg ({:.3} % of {:.0} kg) to keep dt >= {:.2e} s", added_mass_total, 100.0 * added_mass_total / total_mass, total_mass, dt_floor);
    }
    for (p, part) in model.mesh.parts.iter().enumerate() {
        let nodes = model.mesh.part_nodes(p);
        let (m, v1) = results.mean_velocity(&nodes);
        let (_, v0) = Model::mean_velocity(&masses, &model.initial_velocity, &nodes);
        let dv = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        info!("part '{}': mass {:.1} kg, delta-v [{:+.3}, {:+.3}, {:+.3}] m/s (|dv| = {:.3})", part.name, m, dv[0], dv[1], dv[2], (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt());
    }
    results
}

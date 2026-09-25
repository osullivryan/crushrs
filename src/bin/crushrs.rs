//! crushrs command line.
//!
//!   crushrs run model.toml                     # any TOML setup
//!   crushrs barrier neon [--calibrate N]       # NCAP-style barrier test vs NHTSA targets
//!   crushrs crash 30 30 --gif crash.gif        # Silverado vs Neon head-on
//!   crushrs tbone 30 0 -1.2 --vtk out/tbone    # Silverado into the Neon's side

use clap::{Args, Parser, Subcommand};
use crushrs::output::gif::{write_gif, GifOptions};
use crushrs::vehicle::{add_vehicle, add_vehicle_accelerometer, assign_vehicle, barrier_metrics_window, barrier_model, calibrate, calibrate_pulse, BarrierTargets, Heading, PulseComparison, Vehicle, PULSE_DT, PULSE_END};
use crushrs::{BlockFace, Mesh, Model, Results, MPH};
use nalgebra::Vector3;
use std::path::Path;

#[derive(Parser)]
#[command(name = "crushrs", about = "Fast explicit crash solver for delta-v estimation")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a TOML setup file.
    Run {
        file: String,
        #[command(flatten)]
        out: Out,
    },
    /// NCAP-style 35 mph rigid-barrier test of a calibrated vehicle.
    Barrier {
        /// neon | silverado | neon-side, with -raw for the analytic starting point
        vehicle: String,
        #[arg(long, default_value_t = 0.3)]
        element_size: f64,
        /// Calibrate from the starting point for N iterations.
        #[arg(long)]
        calibrate: Option<usize>,
        /// Measured barrier pulse (NHTSA ASCII signal file, time[s] TAB g):
        /// compare against it and, with --calibrate, fit the force-crush
        /// table to it instead of the KW400 targets.
        #[arg(long)]
        pulse: Option<String>,
        /// Override the crush-zone modulus (Pa) before running / calibrating.
        #[arg(long)]
        modulus: Option<f64>,
        /// Override the body modulus (Pa) before running / calibrating.
        #[arg(long)]
        body_modulus: Option<f64>,
        /// Override the crush-zone length (m).
        #[arg(long)]
        crush_zone: Option<f64>,
        /// Override the crush-zone mass (kg).
        #[arg(long)]
        crush_mass: Option<f64>,
        /// Override the transverse cap factor of the crush material.
        #[arg(long)]
        transverse: Option<f64>,
        #[command(flatten)]
        out: Out,
    },
    /// Head-on 2007 Silverado vs 1996 Neon.
    Crash {
        #[arg(default_value_t = 30.0)]
        truck_mph: f64,
        #[arg(default_value_t = 30.0)]
        car_mph: f64,
        #[command(flatten)]
        out: Out,
    },
    /// Head-on between any two vehicles (neon | neon-pulse | silverado | *-raw).
    Headon {
        vehicle_a: String,
        vehicle_b: String,
        #[arg(default_value_t = 35.0)]
        mph_a: f64,
        #[arg(default_value_t = 35.0)]
        mph_b: f64,
        /// Lateral offset of vehicle B (m): 0 = full overlap.
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        offset: f64,
        #[command(flatten)]
        out: Out,
    },
    /// T-bone: Silverado into the side of a Neon.
    Tbone {
        #[arg(default_value_t = 30.0)]
        truck_mph: f64,
        /// Neon speed along its own axis (+y).
        #[arg(default_value_t = 0.0)]
        car_mph: f64,
        /// Impact point along the Neon relative to its centre (+ toward its front).
        #[arg(default_value_t = 0.0, allow_hyphen_values = true)]
        offset: f64,
        #[command(flatten)]
        out: Out,
    },
}

#[derive(Args, Clone, Default)]
struct Out {
    /// Animated GIF of the deformed mesh.
    #[arg(long)]
    gif: Option<String>,
    /// VTK time series: <base>.pvd + <base>_NNNN.vtu.
    #[arg(long)]
    vtk: Option<String>,
    /// Parquet time series: <base>.nodes/.frames/.parts.parquet.
    #[arg(long)]
    history: Option<String>,
    /// Sample the history every N steps.
    #[arg(long, default_value_t = 1)]
    history_steps: usize,
    /// Safety factor on the stable time step.
    #[arg(long)]
    dt_scale: Option<f64>,
}

impl Out {
    /// Turn on frame / history capture as needed.
    fn configure(&self, model: &mut Model) {
        if (self.gif.is_some() || self.vtk.is_some()) && model.settings.frame_steps == 0 {
            model.settings.frame_steps = 15;
        }
        if self.history.is_some() {
            model.settings.history_steps = self.history_steps.max(1);
        }
        if let Some(f) = self.dt_scale {
            model.settings.dt_scale = f;
        }
    }
}

fn outputs(model: &Model, results: &Results, out: &Out, view: Option<Vector3<f64>>) {
    let (gif, vtk) = (&out.gif, &out.vtk);
    if let Some(base) = &out.history {
        let paths = crushrs::output::history::write_history(model, results, Path::new(base)).expect("write history");
        println!("wrote {} ({} node samples, {} part samples)", paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", "), results.node_history.time.len(), results.history.len());
    }
    if let Some(path) = gif {
        let mut opts = GifOptions { delay_cs: 5, plastic_strain_scale: 1.0, ..Default::default() };
        if let Some(v) = view {
            opts.view_dir = v;
        }
        write_gif(&model.mesh, &results.frames, path, &opts).expect("write gif");
        println!("wrote {}", path);
    }
    if let Some(base) = vtk {
        crushrs::output::vtk::write_series(&model.mesh, &results.frames, Path::new(base)).expect("write vtk");
        println!("wrote {}.pvd ({} frames)", base, results.frames.len());
    }
}

fn part_delta_v(model: &Model, results: &Results, part: &str) -> (f64, [f64; 3], [f64; 3]) {
    let nodes = model.mesh.nodes_of(part).unwrap();
    let (m, v1) = results.mean_velocity(&nodes);
    let (_, v0) = Model::mean_velocity(&results.masses, &model.initial_velocity, &nodes);
    (m, [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]], v1)
}

fn vehicle_by_name(name: &str) -> Vehicle {
    match name {
        "silverado" => Vehicle::chevrolet_silverado_2007_tuned(),
        "silverado-raw" => Vehicle::chevrolet_silverado_2007(),
        "neon" => Vehicle::dodge_neon_1996_tuned(),
        "neon-raw" => Vehicle::dodge_neon_1996(),
        "neon-pulse" => Vehicle::dodge_neon_1996_pulse(),
        other => panic!("unknown vehicle '{}' (neon | neon-raw | neon-pulse | silverado | silverado-raw)", other),
    }
}

/// Dynamic crush as an accelerometer (rear-seat) node's x displacement
/// relative to the mean of the front face: what an NHTSA rear-seat
/// accelerometer integrates to.
fn crush_rear(model: &Model, results: &Results, accelerometer: &str, front_set: &str) -> f64 {
    let acc = model.accelerometers.iter().find(|a| a.name == accelerometer).expect("accelerometer");
    let node = acc.nodes[0];
    let front = model.mesh.face_set_nodes(front_set).unwrap();
    let front_u = front.iter().map(|n| results.displacement[*n][0]).sum::<f64>() / front.len() as f64;
    (results.displacement[node][0] - front_u).abs()
}

/// CFC 60 peak magnitude of an accelerometer's local-x pulse (g), or None.
fn peak_g(model: &Model, results: &Results, accelerometer: &str) -> Option<f64> {
    crushrs::signal::Pulse::from_history(model, results, accelerometer, crushrs::vehicle::PULSE_DT).ok().map(|p| p.filtered(60.0).accel.iter().map(|a| a.abs()).fold(0.0, f64::max) / 9.81)
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).format_timestamp(None).init();
    match Cli::parse().cmd {
        Cmd::Run { file, mut out } => {
            let path = Path::new(&file);
            let cfg = crushrs::input::config::Config::load(path).unwrap_or_else(|e| panic!("{}", e));
            let mut model = cfg.build(path.parent().unwrap_or(Path::new("."))).unwrap_or_else(|e| panic!("{}", e));
            out.gif = out.gif.or(cfg.output.gif.clone());
            out.vtk = out.vtk.or(cfg.output.vtk.clone());
            out.history = out.history.or(cfg.output.history.clone());
            if model.settings.history_steps > 0 {
                out.history_steps = model.settings.history_steps;
            }
            out.configure(&mut model);
            let results = crushrs::run(&model);
            println!("{} elements, {} steps, {:.2?} ({:.0} ms simulated)", model.mesh.hexes.len(), results.steps, results.wall_time, results.time * 1e3);
            for (p, part) in model.mesh.parts.iter().enumerate() {
                let (m, dv, _) = part_delta_v(&model, &results, &part.name);
                let _ = p;
                println!("{:<16} {:8.0} kg   delta-v [{:+7.2} {:+7.2} {:+7.2}] m/s  |dv| {:5.1} mph", part.name, m, dv[0], dv[1], dv[2], (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt() / MPH);
            }
            outputs(&model, &results, &out, None);
        }
        Cmd::Barrier { vehicle, element_size, calibrate: cal, pulse, modulus, body_modulus, crush_zone, crush_mass, transverse, out } => {
            if let Some(pulse_file) = pulse {
                let mut v = match vehicle.as_str() {
                    "silverado" => Vehicle::chevrolet_silverado_2007_tuned(),
                    "silverado-raw" => Vehicle::chevrolet_silverado_2007(),
                    "neon-raw" => Vehicle::dodge_neon_1996(),
                    "neon-pulse" => Vehicle::dodge_neon_1996_pulse(),
                    _ => Vehicle::dodge_neon_1996_tuned(),
                };
                if let Some(e) = modulus {
                    v.modulus = e;
                }
                if let Some(e) = body_modulus {
                    v.body_modulus = e;
                }
                if let Some(l) = crush_zone {
                    v.crush_zone_length = l;
                    v.force_table = None;
                }
                if let Some(m) = crush_mass {
                    v.crush_zone_mass = Some(m);
                }
                if let Some(t) = transverse {
                    v.transverse_factor = t;
                }
                let speed = 35.0 * MPH;
                let measured = crushrs::signal::Pulse::read_nhtsa_tsv(Path::new(&pulse_file), true).unwrap_or_else(|e| panic!("{}", e));
                println!("{}: {:.0} kg at {:.1} m/s vs measured pulse {} ({} samples at {:.0} kHz)", v.name, v.mass, speed, pulse_file, measured.time.len(), 1e-3 / measured.dt());
                let v = match cal {
                    Some(n) => {
                        let (tuned, _) = calibrate_pulse(&v, &measured, speed, element_size, n, true);
                        tuned
                    }
                    None => v,
                };
                let mut model = barrier_model(&v, element_size, speed, PULSE_END, if out.gif.is_some() || out.vtk.is_some() { 20 } else { 0 });
                let acc = add_vehicle_accelerometer(&mut model, &v.name, &format!("{}_front", v.name));
                out.configure(&mut model);
                model.settings.history_steps = 1;
                let results = crushrs::run(&model);
                let sim = crushrs::signal::Pulse::from_history(&model, &results, &acc, PULSE_DT).unwrap().filtered(60.0);
                let cmp = PulseComparison::new(&measured.filtered(60.0).resampled(PULSE_DT).truncated(PULSE_END), &sim, v.mass, speed);
                if let Some(t) = &v.force_table {
                    println!("force-crush table (crush mm, force kN): {}", t.iter().map(|[x, f]| format!("({:.0}, {:.0})", x * 1e3, f / 1e3)).collect::<Vec<_>>().join(" "));
                    if let Some(cm) = &v.compaction_map {
                        println!("knot compactions: {}", cm.iter().map(|c| format!("{:.3}", c)).collect::<Vec<_>>().join(" "));
                    }
                    println!("modulus {:.1} MPa, body modulus {:.0} MPa, crush zone {:.2} m ({:.0} kg), elements {:.2} m", v.modulus / 1e6, v.body_modulus / 1e6, v.crush_zone_length, v.crush_zone_mass.unwrap_or(v.mass * v.crush_zone_length / v.size[0]), v.crush_element_size.unwrap_or(element_size));
                }
                println!("{} elements, {} steps, {:.2?}", model.mesh.hexes.len(), results.steps, results.wall_time);
                println!("  t[ms]   a_meas   a_sim [g]    v_meas   v_sim [m/s]   x_meas   x_sim [mm]");
                let step = (0.005 / PULSE_DT).round() as usize;
                for i in (0..cmp.time.len()).step_by(step) {
                    println!("  {:5.0}   {:6.1}   {:6.1}       {:6.2}   {:6.2}       {:6.0}   {:6.0}", cmp.time[i] * 1e3, cmp.a_meas[i] / 9.81, cmp.a_sim[i] / 9.81, cmp.v_meas[i], cmp.v_sim[i], cmp.x_meas[i] * 1e3, cmp.x_sim[i] * 1e3);
                }
                println!("                    measured   simulated");
                println!("peak accel (CFC60) {:8.1}    {:8.1} g", cmp.meas.peak_accel / 9.81, cmp.sim.peak_accel / 9.81);
                println!("max crush          {:8.0}    {:8.0} mm  at {:.0} / {:.0} ms", cmp.meas.max_crush * 1e3, cmp.sim.max_crush * 1e3, cmp.meas.t_max_crush * 1e3, cmp.sim.t_max_crush * 1e3);
                println!("restitution        {:8.3}    {:8.3}", cmp.meas.restitution, cmp.sim.restitution);
                println!("v at {:.0} ms       {:8.2}    {:8.2} m/s", cmp.time.last().unwrap() * 1e3, cmp.v_meas.last().unwrap(), cmp.v_sim.last().unwrap());
                println!("rms accel error    {:8.2} g", cmp.rms_accel_error() / 9.81);
                if let Some(base) = &out.history {
                    let path = format!("{}.pulse.parquet", base);
                    cmp.write_parquet(Path::new(&path)).expect("write pulse");
                    println!("wrote {}", path);
                }
                outputs(&model, &results, &out, None);
                return;
            }
            let (v, target) = match vehicle.as_str() {
                "silverado" => (Vehicle::chevrolet_silverado_2007_tuned(), Vehicle::chevrolet_silverado_2007()),
                "silverado-raw" => (Vehicle::chevrolet_silverado_2007(), Vehicle::chevrolet_silverado_2007()),
                "neon-raw" => (Vehicle::dodge_neon_1996(), Vehicle::dodge_neon_1996()),
                "neon-side" => (Vehicle::dodge_neon_1996_tuned().side_profile(), Vehicle::dodge_neon_1996().side_profile()),
                "neon-side-raw" => (Vehicle::dodge_neon_1996().side_profile(), Vehicle::dodge_neon_1996().side_profile()),
                _ => (Vehicle::dodge_neon_1996_tuned(), Vehicle::dodge_neon_1996()),
            };
            let side = vehicle.starts_with("neon-side");
            let speed = if side { 45.0 * MPH } else { 35.0 * MPH };
            let window = if side { (0.025, 0.15) } else { (0.025, 0.4) };
            let (crush_pred, f_pred) = target.barrier_prediction(speed);
            let kw_target = target.curve.kw_window(window);
            println!("{}: {:.0} kg, F_y = {:.1} kN, k = {:.2} MN/m, E = {:.1} MPa; target KW{:.0} = {:.0} N/mm, crush {:.0} mm", v.name, v.mass, v.curve.yield_force / 1e3, v.curve.stiffness / 1e6, v.modulus / 1e6, window.1 * 1e3, kw_target / 1e3, crush_pred * 1e3);
            let v = match cal {
                Some(n) => {
                    let t = BarrierTargets::from_curve(&target, speed, window, 0.12);
                    let (tuned, _) = calibrate(&v, &t, element_size, n, true);
                    let cm = tuned.crush_material();
                    println!("calibrated at h = {} m: F_y = {:.2} kN, k = {:.3} MN/m, E = {:.1} MPa  (sigma_y = {:.0} Pa, H = {:.0} Pa)", element_size, tuned.curve.yield_force / 1e3, tuned.curve.stiffness / 1e6, tuned.modulus / 1e6, cm.plasticity.as_ref().unwrap().yield_stress, cm.plasticity.as_ref().unwrap().hardening);
                    tuned
                }
                None => v,
            };
            let mut model = barrier_model(&v, element_size, speed, 0.14, if out.gif.is_some() || out.vtk.is_some() { 20 } else { 0 });
            add_vehicle_accelerometer(&mut model, &v.name, &format!("{}_front", v.name));
            out.configure(&mut model);
            let results = crushrs::run(&model);
            let m = barrier_metrics_window(&model, &results, &v.name, 0, window);
            println!("{} elements, {} steps, {:.2?}", model.mesh.hexes.len(), results.steps, results.wall_time);
            println!("                 simulated   target");
            println!("KW{:<3.0}          {:8.0}    {:8.0} N/mm", window.1 * 1e3, m.kw400 / 1e3, kw_target / 1e3);
            println!("max crush      {:8.0}    {:8.0} mm", m.max_crush * 1e3, crush_pred * 1e3);
            println!("peak force     {:8.0}    {:8.0} kN", m.peak_force / 1e3, f_pred / 1e3);
            println!("mean force     {:8.0}    {:8.0} kN", m.mean_force / 1e3, 0.5 * v.mass * speed * speed / crush_pred / 1e3);
            println!("restitution    {:8.3}    0.03-0.20", m.restitution);
            println!("delta-v        {:8.2} m/s", m.delta_v);
            outputs(&model, &results, &out, None);
        }
        Cmd::Crash { truck_mph, car_mph, out } => {
            let truck = Vehicle::chevrolet_silverado_2007_tuned();
            let car = Vehicle::dodge_neon_1996_tuned();
            let h = Vehicle::TUNED_ELEMENT_SIZE;
            let gap = 0.005;
            let mut mesh = Mesh::new();
            add_vehicle(&mut mesh, &truck, -gap, 0.0, 0.0, Heading::PlusX, h);
            add_vehicle(&mut mesh, &car, gap, 0.0, 0.0, Heading::MinusX, h);
            let mut model = Model::new(mesh);
            assign_vehicle(&mut model, &truck, Heading::PlusX, truck_mph * MPH);
            assign_vehicle(&mut model, &car, Heading::MinusX, car_mph * MPH);
            let n_face = model.mesh.face_set_nodes("neon_front").unwrap().len();
            model.add_contact_pair("silverado_front", "neon_front", car.contact_stiffness_per_node(n_face, h).min(truck.contact_stiffness_per_node(n_face, h)), 0.3);
            model.settings.end_time = 0.15;
            add_vehicle_accelerometer(&mut model, "silverado", "silverado_front");
            add_vehicle_accelerometer(&mut model, "neon", "neon_front");
            out.configure(&mut model);
            let results = crushrs::run(&model);
            let (mt, dvt, vt) = part_delta_v(&model, &results, "silverado");
            let (mc, dvc, vc) = part_delta_v(&model, &results, "neon");
            let (v_t0, v_c0) = (truck_mph * MPH, -car_mph * MPH);
            let v_common = (mt * v_t0 + mc * v_c0) / (mt + mc);
            println!("2007 Silverado {:.0} kg @ {:+.0} mph  vs  1996 Neon {:.0} kg @ {:+.0} mph", mt, truck_mph, mc, -car_mph);
            println!("{} elements, {} steps, {:.2?} (forces {:.2?}, contact {:.2?})", model.mesh.hexes.len(), results.steps, results.wall_time, results.force_time, results.contact_time);
            println!("               delta-v         rear-seat crush   (perfectly plastic limit)");
            println!("Silverado  {:+7.2} m/s {:+6.1} mph   {:4.0} mm   ({:+.2} m/s)", dvt[0], dvt[0] / MPH, crush_rear(&model, &results, "silverado_rear", "silverado_front") * 1e3, v_common - v_t0);
            println!("Neon       {:+7.2} m/s {:+6.1} mph   {:4.0} mm   ({:+.2} m/s)", dvc[0], dvc[0] / MPH, crush_rear(&model, &results, "neon_rear", "neon_front") * 1e3, v_common - v_c0);
            println!("restitution e = {:.3};  momentum {:.0} -> {:.0} kg·m/s", (vc[0] - vt[0]) / (v_t0 - v_c0), mt * v_t0 + mc * v_c0, mt * vt[0] + mc * vc[0]);
            outputs(&model, &results, &out, None);
        }
        Cmd::Headon { vehicle_a, vehicle_b, mph_a, mph_b, offset, out } => {
            let mut a = vehicle_by_name(&vehicle_a);
            let mut b = vehicle_by_name(&vehicle_b);
            a.name = format!("{}_a", a.name);
            b.name = format!("{}_b", b.name);
            let h = Vehicle::TUNED_ELEMENT_SIZE;
            let gap = 0.005;
            let mut mesh = Mesh::new();
            add_vehicle(&mut mesh, &a, -gap, 0.0, 0.0, Heading::PlusX, h);
            add_vehicle(&mut mesh, &b, gap, offset, 0.0, Heading::MinusX, h);
            let mut model = Model::new(mesh);
            assign_vehicle(&mut model, &a, Heading::PlusX, mph_a * MPH);
            assign_vehicle(&mut model, &b, Heading::MinusX, mph_b * MPH);
            let (fa, fb) = (format!("{}_front", a.name), format!("{}_front", b.name));
            let n_face = model.mesh.face_set_nodes(&fa).unwrap().len().min(model.mesh.face_set_nodes(&fb).unwrap().len());
            model.add_contact_pair(&fa, &fb, a.contact_stiffness_per_node(n_face, h).min(b.contact_stiffness_per_node(n_face, h)), 0.3);
            model.settings.end_time = 0.15;
            let acc_a = add_vehicle_accelerometer(&mut model, &a.name, &fa);
            let acc_b = add_vehicle_accelerometer(&mut model, &b.name, &fb);
            model.settings.history_steps = 1;
            out.configure(&mut model);
            let results = crushrs::run(&model);
            let (ma, dva, va) = part_delta_v(&model, &results, &a.name);
            let (mb, dvb, vb) = part_delta_v(&model, &results, &b.name);
            let (v_a0, v_b0) = (mph_a * MPH, -mph_b * MPH);
            let v_common = (ma * v_a0 + mb * v_b0) / (ma + mb);
            println!("{} {:.0} kg @ {:+.0} mph  vs  {} {:.0} kg @ {:+.0} mph", a.name, ma, mph_a, b.name, mb, -mph_b);
            println!("{} elements, {} steps, {:.2?} (forces {:.2?}, contact {:.2?})", model.mesh.hexes.len(), results.steps, results.wall_time, results.force_time, results.contact_time);
            println!("                 delta-v         rear-seat crush  peak (CFC60)  (perfectly plastic limit)");
            println!("{:<12} {:+7.2} m/s {:+6.1} mph   {:4.0} mm   {:6.1} g      ({:+.2} m/s)", a.name, dva[0], dva[0] / MPH, crush_rear(&model, &results, &acc_a, &fa) * 1e3, peak_g(&model, &results, &acc_a).unwrap_or(f64::NAN), v_common - v_a0);
            println!("{:<12} {:+7.2} m/s {:+6.1} mph   {:4.0} mm   {:6.1} g      ({:+.2} m/s)", b.name, dvb[0], dvb[0] / MPH, crush_rear(&model, &results, &acc_b, &fb) * 1e3, peak_g(&model, &results, &acc_b).unwrap_or(f64::NAN), v_common - v_b0);
            println!("restitution e = {:.3};  momentum {:.0} -> {:.0} kg·m/s", (vb[0] - va[0]) / (v_a0 - v_b0), ma * v_a0 + mb * v_b0, ma * va[0] + mb * vb[0]);
            outputs(&model, &results, &out, None);
        }
        Cmd::Tbone { truck_mph, car_mph, offset, out } => {
            let truck = Vehicle::chevrolet_silverado_2007_tuned();
            let car = Vehicle::dodge_neon_1996_tuned();
            let side = car.side_profile();
            let h = Vehicle::TUNED_ELEMENT_SIZE;
            let gap = 0.005;
            let mut mesh = Mesh::new();
            add_vehicle(&mut mesh, &truck, -gap, 0.0, 0.0, Heading::PlusX, h);
            let car_origin = [gap, -offset - side.size[1] / 2.0, 0.0];
            mesh.add_hex_block(&side.name, car_origin, side.size, h, &[(BlockFace::XMin, "neon_side")]);
            let mut model = Model::new(mesh);
            assign_vehicle(&mut model, &truck, Heading::PlusX, truck_mph * MPH);
            model.set_material(&side.name, side.crush_material());
            model.set_initial_velocity(&side.name, [0.0, car_mph * MPH, 0.0]);
            let n_face = model.mesh.face_set_nodes("neon_side").unwrap().len();
            model.add_contact_pair("silverado_front", "neon_side", side.contact_stiffness_per_node(n_face, h).min(truck.contact_stiffness_per_node(n_face, h)), 0.3);
            model.settings.end_time = 0.15;
            add_vehicle_accelerometer(&mut model, "silverado", "silverado_front");
            add_vehicle_accelerometer(&mut model, &side.name, "neon_side");
            out.configure(&mut model);
            let results = crushrs::run(&model);
            let (mt, dvt, _) = part_delta_v(&model, &results, "silverado");
            let (mc, dvc, _) = part_delta_v(&model, &results, &side.name);
            let car_nodes = model.mesh.nodes_of(&side.name).unwrap();
            let yaw = results.mean_angular_velocity(&model.mesh.nodes, &car_nodes)[2];
            let v_common = mt * truck_mph * MPH / (mt + mc);
            // Intrusion: width reduction of the struck side vs the far side at the same (y, z).
            let far_x = car_origin[0] + side.size[0];
            let side_nodes = model.mesh.face_set_nodes("neon_side").unwrap();
            let far: Vec<usize> = car_nodes.iter().copied().filter(|n| (model.mesh.nodes[*n][0] - far_x).abs() < 1e-9).collect();
            let mut intr: Vec<(f64, f64)> = Vec::new();
            for n in &side_nodes {
                let p = model.mesh.nodes[*n];
                let opp = far.iter().find(|m| (model.mesh.nodes[**m][1] - p[1]).abs() < 1e-9 && (model.mesh.nodes[**m][2] - p[2]).abs() < 1e-9).map(|m| results.displacement[*m][0]).unwrap_or(0.0);
                intr.push((p[1], results.displacement[*n][0] - opp));
            }
            let max_intr = intr.iter().map(|(_, d)| *d).fold(0.0, f64::max);
            let under: Vec<f64> = intr.iter().filter(|(y, _)| y.abs() <= truck.size[1] / 2.0).map(|(_, d)| *d).collect();
            let mean_intr = under.iter().sum::<f64>() / under.len().max(1) as f64;
            println!("T-bone: 2007 Silverado {:.0} kg @ {:.0} mph into the side of a 1996 Neon {:.0} kg @ {:.0} mph, offset {:+.2} m", mt, truck_mph, mc, car_mph, offset);
            println!("{} elements, {} steps, {:.2?} (forces {:.2?}, contact {:.2?})", model.mesh.hexes.len(), results.steps, results.wall_time, results.force_time, results.contact_time);
            println!("                delta-v (x, y)               |dv|      (plastic, no-yaw limit)");
            println!("Silverado   {:+6.2} {:+6.2} m/s  {:+6.1} mph   ({:+.2} m/s)", dvt[0], dvt[1], (dvt[0] * dvt[0] + dvt[1] * dvt[1]).sqrt() / MPH, v_common - truck_mph * MPH);
            println!("Neon        {:+6.2} {:+6.2} m/s  {:+6.1} mph   ({:+.2} m/s)", dvc[0], dvc[1], (dvc[0] * dvc[0] + dvc[1] * dvc[1]).sqrt() / MPH, v_common);
            println!("Neon yaw rate {:+.2} rad/s ({:+.0} deg/s);  side intrusion: max {:.0} mm, mean under truck {:.0} mm", yaw, yaw.to_degrees(), max_intr * 1e3, mean_intr * 1e3);
            outputs(&model, &results, &out, Some(Vector3::new(0.35, -0.55, 0.75)));
        }
    }
}

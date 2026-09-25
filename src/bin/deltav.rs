//! deltav command line.
//!
//!   deltav run model.toml                     # any TOML setup
//!   deltav barrier neon [--calibrate N]       # NCAP-style barrier test vs NHTSA targets
//!   deltav crash 30 30 --gif crash.gif        # Silverado vs Neon head-on
//!   deltav tbone 30 0 -1.2 --vtk out/tbone    # Silverado into the Neon's side

use clap::{Parser, Subcommand};
use deltav::output::gif::{write_gif, GifOptions};
use deltav::vehicle::{add_vehicle, assign_vehicle, barrier_metrics_window, calibrate, run_barrier_test, BarrierTargets, Heading, Vehicle};
use deltav::{BlockFace, Mesh, Model, Results, MPH};
use nalgebra::Vector3;
use std::path::Path;

#[derive(Parser)]
#[command(name = "deltav", about = "Fast explicit crash solver for delta-v estimation")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a TOML setup file.
    Run {
        file: String,
        #[arg(long)]
        gif: Option<String>,
        #[arg(long)]
        vtk: Option<String>,
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
        #[arg(long)]
        gif: Option<String>,
        #[arg(long)]
        vtk: Option<String>,
    },
    /// Head-on 2007 Silverado vs 1996 Neon.
    Crash {
        #[arg(default_value_t = 30.0)]
        truck_mph: f64,
        #[arg(default_value_t = 30.0)]
        car_mph: f64,
        #[arg(long)]
        gif: Option<String>,
        #[arg(long)]
        vtk: Option<String>,
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
        #[arg(long)]
        gif: Option<String>,
        #[arg(long)]
        vtk: Option<String>,
    },
}

fn outputs(model: &Model, results: &Results, gif: &Option<String>, vtk: &Option<String>, view: Option<Vector3<f64>>) {
    if let Some(path) = gif {
        let mut opts = GifOptions { delay_cs: 5, plastic_strain_scale: 1.0, ..Default::default() };
        if let Some(v) = view {
            opts.view_dir = v;
        }
        write_gif(&model.mesh, &results.frames, path, &opts).expect("write gif");
        println!("wrote {}", path);
    }
    if let Some(base) = vtk {
        deltav::output::vtk::write_series(&model.mesh, &results.frames, Path::new(base)).expect("write vtk");
        println!("wrote {}.pvd ({} frames)", base, results.frames.len());
    }
}

fn part_delta_v(model: &Model, results: &Results, part: &str) -> (f64, [f64; 3], [f64; 3]) {
    let nodes = model.mesh.nodes_of(part).unwrap();
    let (m, v1) = results.mean_velocity(&nodes);
    let (_, v0) = Model::mean_velocity(&results.masses, &model.initial_velocity, &nodes);
    (m, [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]], v1)
}

/// Crush along x of a part: original length minus current front-to-rear distance.
fn crush_x(model: &Model, results: &Results, part: &str, front_set: &str, len0: f64) -> f64 {
    let x_now = |n: usize| model.mesh.nodes[n][0] + results.displacement[n][0];
    let front = model.mesh.face_set_nodes(front_set).unwrap();
    let body = model.mesh.nodes_of(part).unwrap();
    let front_x0 = model.mesh.nodes[front[0]][0];
    let (xmin, xmax) = body.iter().map(|n| model.mesh.nodes[*n][0]).fold((f64::MAX, f64::MIN), |(a, b), x| (a.min(x), b.max(x)));
    let rear_x0 = if (front_x0 - xmin).abs() < 1e-9 { xmax } else { xmin };
    let mean = |ids: &[usize]| ids.iter().map(|n| x_now(*n)).sum::<f64>() / ids.len() as f64;
    let rear: Vec<usize> = body.iter().copied().filter(|n| (model.mesh.nodes[*n][0] - rear_x0).abs() < 1e-9).collect();
    len0 - (mean(&front) - mean(&rear)).abs()
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).format_timestamp(None).init();
    match Cli::parse().cmd {
        Cmd::Run { file, gif, vtk } => {
            let path = Path::new(&file);
            let cfg = deltav::input::config::Config::load(path).unwrap_or_else(|e| panic!("{}", e));
            let mut model = cfg.build(path.parent().unwrap_or(Path::new("."))).unwrap_or_else(|e| panic!("{}", e));
            let gif = gif.or(cfg.output.gif.clone());
            let vtk = vtk.or(cfg.output.vtk.clone());
            if (gif.is_some() || vtk.is_some()) && model.settings.frame_steps == 0 {
                model.settings.frame_steps = 15;
            }
            let results = deltav::run(&model);
            println!("{} elements, {} steps, {:.2?} ({:.0} ms simulated)", model.mesh.hexes.len(), results.steps, results.wall_time, results.time * 1e3);
            for (p, part) in model.mesh.parts.iter().enumerate() {
                let (m, dv, _) = part_delta_v(&model, &results, &part.name);
                let _ = p;
                println!("{:<16} {:8.0} kg   delta-v [{:+7.2} {:+7.2} {:+7.2}] m/s  |dv| {:5.1} mph", part.name, m, dv[0], dv[1], dv[2], (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt() / MPH);
            }
            outputs(&model, &results, &gif, &vtk, None);
        }
        Cmd::Barrier { vehicle, element_size, calibrate: cal, gif, vtk } => {
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
                    println!("calibrated at h = {} m: F_y = {:.2} kN, k = {:.3} MN/m, E = {:.1} MPa  (sigma_y = {:.0} Pa, H = {:.0} Pa)", element_size, tuned.curve.yield_force / 1e3, tuned.curve.stiffness / 1e6, tuned.modulus / 1e6, cm.plasticity.unwrap().yield_stress, cm.plasticity.unwrap().hardening);
                    tuned
                }
                None => v,
            };
            let (model, results, _) = run_barrier_test(&v, element_size, speed, 0.14, if gif.is_some() || vtk.is_some() { 20 } else { 0 });
            let m = barrier_metrics_window(&model, &results, &v.name, 0, window);
            println!("{} elements, {} steps, {:.2?}", model.mesh.hexes.len(), results.steps, results.wall_time);
            println!("                 simulated   target");
            println!("KW{:<3.0}          {:8.0}    {:8.0} N/mm", window.1 * 1e3, m.kw400 / 1e3, kw_target / 1e3);
            println!("max crush      {:8.0}    {:8.0} mm", m.max_crush * 1e3, crush_pred * 1e3);
            println!("peak force     {:8.0}    {:8.0} kN", m.peak_force / 1e3, f_pred / 1e3);
            println!("mean force     {:8.0}    {:8.0} kN", m.mean_force / 1e3, 0.5 * v.mass * speed * speed / crush_pred / 1e3);
            println!("restitution    {:8.3}    0.03-0.20", m.restitution);
            println!("delta-v        {:8.2} m/s", m.delta_v);
            outputs(&model, &results, &gif, &vtk, None);
        }
        Cmd::Crash { truck_mph, car_mph, gif, vtk } => {
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
            let n_face = model.mesh.face_set_nodes("neon_front").unwrap().len() as f64;
            model.add_contact_pair("silverado_front", "neon_front", 400.0 * car.curve.stiffness / n_face, 0.3);
            model.settings.end_time = 0.15;
            if gif.is_some() || vtk.is_some() {
                model.settings.frame_steps = 15;
            }
            let results = deltav::run(&model);
            let (mt, dvt, vt) = part_delta_v(&model, &results, "silverado");
            let (mc, dvc, vc) = part_delta_v(&model, &results, "neon");
            let (v_t0, v_c0) = (truck_mph * MPH, -car_mph * MPH);
            let v_common = (mt * v_t0 + mc * v_c0) / (mt + mc);
            println!("2007 Silverado {:.0} kg @ {:+.0} mph  vs  1996 Neon {:.0} kg @ {:+.0} mph", mt, truck_mph, mc, -car_mph);
            println!("{} elements, {} steps, {:.2?} (forces {:.2?}, contact {:.2?})", model.mesh.hexes.len(), results.steps, results.wall_time, results.force_time, results.contact_time);
            println!("               delta-v              crush   (perfectly plastic limit)");
            println!("Silverado  {:+7.2} m/s {:+6.1} mph   {:4.0} mm   ({:+.2} m/s)", dvt[0], dvt[0] / MPH, crush_x(&model, &results, "silverado", "silverado_front", truck.size[0]) * 1e3, v_common - v_t0);
            println!("Neon       {:+7.2} m/s {:+6.1} mph   {:4.0} mm   ({:+.2} m/s)", dvc[0], dvc[0] / MPH, crush_x(&model, &results, "neon", "neon_front", car.size[0]) * 1e3, v_common - v_c0);
            println!("restitution e = {:.3};  momentum {:.0} -> {:.0} kg·m/s", (vc[0] - vt[0]) / (v_t0 - v_c0), mt * v_t0 + mc * v_c0, mt * vt[0] + mc * vc[0]);
            outputs(&model, &results, &gif, &vtk, None);
        }
        Cmd::Tbone { truck_mph, car_mph, offset, gif, vtk } => {
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
            let n_face = model.mesh.face_set_nodes("neon_side").unwrap().len() as f64;
            model.add_contact_pair("silverado_front", "neon_side", 400.0 * side.curve.stiffness / n_face, 0.3);
            model.settings.end_time = 0.15;
            if gif.is_some() || vtk.is_some() {
                model.settings.frame_steps = 15;
            }
            let results = deltav::run(&model);
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
            outputs(&model, &results, &gif, &vtk, Some(Vector3::new(0.35, -0.55, 0.75)));
        }
    }
}

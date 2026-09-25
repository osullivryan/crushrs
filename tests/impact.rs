//! End-to-end checks: momentum conservation and plausible delta-v in a
//! head-on crash, the tuned Neon reproducing its NHTSA barrier numbers, and
//! the TOML path matching the Rust API.

use crushrs::input::config::Config;
use crushrs::vehicle::{add_vehicle, assign_vehicle, run_barrier_test, Heading, Vehicle};
use crushrs::{Mesh, Model, MPH};

fn head_on(truck_mph: f64, car_mph: f64) -> (Model, crushrs::Results) {
    let truck = Vehicle::chevrolet_silverado_2007_tuned();
    let car = Vehicle::dodge_neon_1996_tuned();
    let mut mesh = Mesh::new();
    add_vehicle(&mut mesh, &truck, -0.005, 0.0, 0.0, Heading::PlusX, Vehicle::TUNED_ELEMENT_SIZE);
    add_vehicle(&mut mesh, &car, 0.005, 0.0, 0.0, Heading::MinusX, Vehicle::TUNED_ELEMENT_SIZE);
    let mut model = Model::new(mesh);
    assign_vehicle(&mut model, &truck, Heading::PlusX, truck_mph * MPH);
    assign_vehicle(&mut model, &car, Heading::MinusX, car_mph * MPH);
    let n_face = model.mesh.face_set_nodes("neon_front").unwrap().len();
    model.add_contact_pair("silverado_front", "neon_front", car.contact_stiffness_per_node(n_face, Vehicle::TUNED_ELEMENT_SIZE), 0.3);
    model.settings.end_time = 0.15;
    let results = crushrs::run(&model);
    (model, results)
}

fn delta_v(model: &Model, r: &crushrs::Results, part: &str) -> (f64, f64) {
    let nodes = model.mesh.nodes_of(part).unwrap();
    let (m, v1) = r.mean_velocity(&nodes);
    let (_, v0) = Model::mean_velocity(&r.masses, &model.initial_velocity, &nodes);
    (m, v1[0] - v0[0])
}

#[test]
fn head_on_conserves_momentum_with_plausible_delta_v() {
    let (model, r) = head_on(30.0, 30.0);
    let (mt, dvt) = delta_v(&model, &r, "silverado");
    let (mc, dvc) = delta_v(&model, &r, "neon");
    // Momentum exchanged only through contact: conserved to round-off.
    assert!((mt * dvt + mc * dvc).abs() < 1e-6 * mt * 30.0 * MPH, "momentum change {}", mt * dvt + mc * dvc);
    // Between the perfectly plastic limit and a 30 % overshoot, no bounce.
    let v0 = 30.0 * MPH;
    let v_common = (mt * v0 - mc * v0) / (mt + mc);
    let plastic = v_common + v0;
    assert!(dvc > plastic && dvc < 1.3 * plastic, "car delta-v {} vs plastic {}", dvc, plastic);
    let e = ((dvc - v0) - (v0 + dvt)) / (2.0 * v0);
    assert!(e > 0.0 && e < 0.3, "restitution {}", e);
    assert_eq!(r.eroded_elements, 0);
}

#[test]
fn tuned_neon_reproduces_ncap_barrier_test() {
    let neon = Vehicle::dodge_neon_1996_tuned();
    let target = Vehicle::dodge_neon_1996();
    let speed = 35.0 * MPH;
    let (_, _, m) = run_barrier_test(&neon, Vehicle::TUNED_ELEMENT_SIZE, speed, 0.14, 0);
    let kw400 = target.curve.kw400();
    let (crush, _) = target.barrier_prediction(speed);
    assert!((m.kw400 - kw400).abs() < 0.05 * kw400, "KW400 {} vs {}", m.kw400, kw400);
    assert!((m.max_crush - crush).abs() < 0.10 * crush, "crush {} vs {}", m.max_crush, crush);
    assert!(m.restitution > 0.03 && m.restitution < 0.2, "restitution {}", m.restitution);
}

#[test]
fn toml_setup_matches_rust_api() {
    let (model_api, r_api) = head_on(30.0, 30.0);
    let toml = format!(
        r#"
[[mesh.block]]
part = "silverado"
origin = [-5.855, -1.015, 0.0]
size = [5.85, 2.03, 1.87]
element_size = 0.3
faces = [{{ face = "x_max", set = "silverado_front" }}]

[[mesh.block]]
part = "neon"
origin = [0.005, -0.855, 0.0]
size = [4.36, 1.71, 1.35]
element_size = 0.3
faces = [{{ face = "x_min", set = "neon_front" }}]

[[material]]
part = "silverado"
youngs_modulus = {te}
density = {td}
yield_stress = {ty}
hardening = {th}
model = "honeycomb"

[[material]]
part = "neon"
youngs_modulus = {ce}
density = {cd}
yield_stress = {cy}
hardening = {ch}
model = "honeycomb"

[[initial_velocity]]
set = "silverado"
velocity = [{v}, 0.0, 0.0]

[[initial_velocity]]
set = "neon"
velocity = [-{v}, 0.0, 0.0]

[[contact]]
a = "silverado_front"
b = "neon_front"
stiffness = {k}
max_distance = 0.3

[solver]
end_time = 0.15
"#,
        te = model_api.materials[0].youngs_modulus,
        td = model_api.materials[0].density,
        ty = model_api.materials[0].plasticity.as_ref().unwrap().yield_stress,
        th = model_api.materials[0].plasticity.as_ref().unwrap().hardening,
        ce = model_api.materials[1].youngs_modulus,
        cd = model_api.materials[1].density,
        cy = model_api.materials[1].plasticity.as_ref().unwrap().yield_stress,
        ch = model_api.materials[1].plasticity.as_ref().unwrap().hardening,
        v = 30.0 * MPH,
        k = 2.0 * model_api.contacts[0].stiffness, // a pair stores half per side
    );
    let cfg = Config::from_str(&toml).unwrap();
    let model = cfg.build(std::path::Path::new(".")).unwrap();
    let r = crushrs::run(&model);
    let (_, dv_toml) = delta_v(&model, &r, "neon");
    let (_, dv_api) = delta_v(&model_api, &r_api, "neon");
    assert!((dv_toml - dv_api).abs() < 1e-3 * dv_api.abs(), "toml {} vs api {}", dv_toml, dv_api);
}

#[test]
fn vtk_series_is_written() {
    let (mut model, _) = head_on(30.0, 30.0);
    model.settings.end_time = 0.02;
    model.settings.frame_steps = 20;
    let r = crushrs::run(&model);
    let dir = std::env::temp_dir().join(format!("deltav_vtk_{}", std::process::id()));
    crushrs::output::vtk::write_series(&model.mesh, &r.frames, &dir.join("crash")).unwrap();
    assert!(dir.join("crash.pvd").exists());
    assert!(dir.join("crash_0000.vtu").exists());
    let pvd = std::fs::read_to_string(dir.join("crash.pvd")).unwrap();
    assert_eq!(pvd.matches("<DataSet").count(), r.frames.len());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn accelerometer_history_integrates_to_delta_v() {
    use crushrs::vehicle::add_vehicle_accelerometer;
    use parquet::file::reader::{FileReader, SerializedFileReader};
    let (mut model, _) = head_on(30.0, 30.0);
    model.settings.end_time = 0.06;
    model.settings.history_steps = 1;
    let name = add_vehicle_accelerometer(&mut model, "neon", "neon_front");
    let r = crushrs::run(&model);
    let h = &r.node_history;
    let idx: Vec<usize> = (0..h.time.len()).filter(|i| model.accelerometers[h.accelerometer[*i]].name == name).collect();
    assert_eq!(idx.len(), r.steps + 1);
    // ∫a dt over the samples equals the velocity change of the node.
    let mut dv = 0.0;
    for w in idx.windows(2) {
        let (i, j) = (w[0], w[1]);
        dv += 0.5 * (h.acceleration[i][0] + h.acceleration[j][0]) * (h.time[j] - h.time[i]);
    }
    let dv_node = h.velocity[*idx.last().unwrap()][0] - h.velocity[idx[0]][0];
    assert!(dv_node > 3.0, "node should have slowed: {}", dv_node);
    assert!((dv - dv_node).abs() < 0.02 * dv_node.abs(), "int a dt {} vs dv {}", dv, dv_node);
    // The body-fixed frame starts aligned with the global axes and stays close in a head-on.
    let f = &r.frame_history;
    assert!((f.axes[0][0][0] - 1.0).abs() < 1e-9 && (f.axes[0][1][1] - 1.0).abs() < 1e-9);
    let last = f.axes.last().unwrap();
    assert!(last[0][0] > 0.99, "frame rotated too much: {:?}", last);
    let i = *idx.last().unwrap();
    let la = h.local_acceleration[i];
    let a = h.acceleration[i];
    let rot = [last[0][0] * a[0] + last[0][1] * a[1] + last[0][2] * a[2], last[1][0] * a[0] + last[1][1] * a[1] + last[1][2] * a[2], last[2][0] * a[0] + last[2][1] * a[1] + last[2][2] * a[2]];
    assert!((0..3).all(|d| (la[d] - rot[d]).abs() < 1e-9));
    // Parquet tables round-trip.
    let dir = std::env::temp_dir().join(format!("crushrs_hist_{}", std::process::id()));
    let paths = crushrs::output::history::write_history(&model, &r, &dir.join("run")).unwrap();
    assert_eq!(paths.len(), 3);
    let reader = SerializedFileReader::new(std::fs::File::open(&paths[0]).unwrap()).unwrap();
    assert_eq!(reader.metadata().file_metadata().num_rows() as usize, h.time.len());
    assert_eq!(reader.metadata().file_metadata().schema_descr().num_columns(), 25);
    let reader = SerializedFileReader::new(std::fs::File::open(&paths[2]).unwrap()).unwrap();
    assert_eq!(reader.metadata().file_metadata().num_rows() as usize, r.history.len() * model.mesh.parts.len());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn pulse_calibrated_neon_matches_nhtsa_test_2320() {
    use crushrs::signal::Pulse;
    use crushrs::vehicle::{run_barrier_pulse, PulseComparison, PULSE_DT, PULSE_END};
    let measured = Pulse::read_nhtsa_tsv(std::path::Path::new("data/nhtsa/v02320tsv.078"), true).unwrap();
    let meas = measured.filtered(60.0).resampled(PULSE_DT).truncated(PULSE_END);
    let v = Vehicle::dodge_neon_1996_pulse();
    let speed = 35.0 * MPH;
    let (_, _, sim) = run_barrier_pulse(&v, Vehicle::TUNED_ELEMENT_SIZE, speed, 0);
    let cmp = PulseComparison::new(&meas, &sim, v.mass, speed);
    // Delta-v (impact speed + rebound) within 5 %, peak within 20 %, crush within 15 %.
    let dv_meas = speed - cmp.v_meas.last().unwrap();
    let dv_sim = speed - cmp.v_sim.last().unwrap();
    assert!((dv_sim - dv_meas).abs() < 0.05 * dv_meas, "delta-v {} vs {}", dv_sim, dv_meas);
    assert!((cmp.sim.peak_accel / cmp.meas.peak_accel - 1.0).abs() < 0.2, "peak {} vs {}", cmp.sim.peak_accel, cmp.meas.peak_accel);
    assert!((cmp.sim.max_crush / cmp.meas.max_crush - 1.0).abs() < 0.15, "crush {} vs {}", cmp.sim.max_crush, cmp.meas.max_crush);
    assert!(cmp.rms_accel_error() < 8.0 * 9.81, "rms {}", cmp.rms_accel_error());
    // Velocity history within 1.5 m/s throughout.
    let worst = cmp.v_meas.iter().zip(&cmp.v_sim).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(worst < 1.5, "velocity error {}", worst);
}

#[test]
fn tabulated_curve_round_trips_through_toml() {
    let toml = r#"
[[mesh.block]]
part = "foam"
origin = [0.0, 0.0, 0.0]
size = [0.3, 0.3, 0.3]
element_size = 0.3
faces = [{ face = "x_max", set = "top" }]

[[material]]
part = "foam"
youngs_modulus = 23.1e6
density = 100.0
model = "honeycomb"
curve = [[0.0, 7000.0], [0.2, 9000.0], [0.5, 30000.0]]
densification = [1.4, 7.0e6]

[solver]
end_time = 0.001
"#;
    let cfg = Config::from_str(toml).unwrap();
    let model = cfg.build(std::path::Path::new(".")).unwrap();
    let p = model.materials[0].plasticity.as_ref().unwrap();
    assert_eq!(p.curve.len(), 3);
    assert_eq!(p.densification, Some([1.4, 7.0e6]));
    assert_eq!(p.yield_stress, 7000.0);
    assert!((p.hardening - 10000.0).abs() < 1e-9);
    assert!((p.yield_at(0.1).0 - 8000.0).abs() < 1e-9);
    // Last slope (70 kPa per unit compaction) continues, plus densification beyond 1.4.
    assert!((p.yield_at(2.0).0 - (30000.0 + 1.5 * 70000.0 + 0.6 * 7.0e6)).abs() < 1e-6);
    let _ = crushrs::run(&model);
}

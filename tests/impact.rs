//! End-to-end checks: momentum conservation and plausible delta-v in a
//! head-on crash, the tuned Neon reproducing its NHTSA barrier numbers, and
//! the TOML path matching the Rust API.

use deltav::input::config::Config;
use deltav::vehicle::{add_vehicle, assign_vehicle, run_barrier_test, Heading, Vehicle};
use deltav::{Mesh, Model, MPH};

fn head_on(truck_mph: f64, car_mph: f64) -> (Model, deltav::Results) {
    let truck = Vehicle::chevrolet_silverado_2007_tuned();
    let car = Vehicle::dodge_neon_1996_tuned();
    let mut mesh = Mesh::new();
    add_vehicle(&mut mesh, &truck, -0.005, 0.0, 0.0, Heading::PlusX, Vehicle::TUNED_ELEMENT_SIZE);
    add_vehicle(&mut mesh, &car, 0.005, 0.0, 0.0, Heading::MinusX, Vehicle::TUNED_ELEMENT_SIZE);
    let mut model = Model::new(mesh);
    assign_vehicle(&mut model, &truck, Heading::PlusX, truck_mph * MPH);
    assign_vehicle(&mut model, &car, Heading::MinusX, car_mph * MPH);
    let n_face = model.mesh.face_set_nodes("neon_front").unwrap().len() as f64;
    model.add_contact_pair("silverado_front", "neon_front", 400.0 * car.curve.stiffness / n_face, 0.3);
    model.settings.end_time = 0.15;
    let results = deltav::run(&model);
    (model, results)
}

fn delta_v(model: &Model, r: &deltav::Results, part: &str) -> (f64, f64) {
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
        ty = model_api.materials[0].plasticity.unwrap().yield_stress,
        th = model_api.materials[0].plasticity.unwrap().hardening,
        ce = model_api.materials[1].youngs_modulus,
        cd = model_api.materials[1].density,
        cy = model_api.materials[1].plasticity.unwrap().yield_stress,
        ch = model_api.materials[1].plasticity.unwrap().hardening,
        v = 30.0 * MPH,
        k = model_api.contacts[0].stiffness,
    );
    let cfg = Config::from_str(&toml).unwrap();
    let model = cfg.build(std::path::Path::new(".")).unwrap();
    let r = deltav::run(&model);
    let (_, dv_toml) = delta_v(&model, &r, "neon");
    let (_, dv_api) = delta_v(&model_api, &r_api, "neon");
    assert!((dv_toml - dv_api).abs() < 1e-3 * dv_api.abs(), "toml {} vs api {}", dv_toml, dv_api);
}

#[test]
fn vtk_series_is_written() {
    let (mut model, _) = head_on(30.0, 30.0);
    model.settings.end_time = 0.02;
    model.settings.frame_steps = 20;
    let r = deltav::run(&model);
    let dir = std::env::temp_dir().join(format!("deltav_vtk_{}", std::process::id()));
    deltav::output::vtk::write_series(&model.mesh, &r.frames, &dir.join("crash")).unwrap();
    assert!(dir.join("crash.pvd").exists());
    assert!(dir.join("crash_0000.vtu").exists());
    let pvd = std::fs::read_to_string(dir.join("crash.pvd")).unwrap();
    assert_eq!(pvd.matches("<DataSet").count(), r.frames.len());
    std::fs::remove_dir_all(&dir).ok();
}

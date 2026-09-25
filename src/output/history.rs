//! Time-series output (the binout equivalent) as Parquet tables:
//!
//! - `<base>.nodes.parquet`  — accelerometer samples: one row per (time,
//!   node) with position, displacement, velocity, acceleration in global
//!   axes (`x, y, z, ux..`, `vx..`, `ax..`) and in the accelerometer's
//!   body-fixed frame (`lux..`, `lvx..`, `lax..`).
//! - `<base>.frames.parquet` — the body-fixed frames: origin and the three
//!   local unit axes in global components at each sample.
//! - `<base>.parts.parquet`  — per-part mass-weighted mean displacement,
//!   velocity, acceleration and kinetic energy.
//!
//! All samples are taken every `Settings::history_steps` steps (1 = every
//! step). Read with pandas / polars / pyarrow / duckdb.

use crate::model::Model;
use crate::output::parquet::{write_table, Column};
use crate::solver::Results;
use std::path::{Path, PathBuf};

fn split3(v: &[[f64; 3]]) -> [Vec<f64>; 3] {
    std::array::from_fn(|d| v.iter().map(|x| x[d]).collect())
}

fn xyz<'a>(prefix: &'a str, v: &[[f64; 3]]) -> Vec<(String, Column)> {
    let [x, y, z] = split3(v);
    vec![(format!("{}x", prefix), Column::F64(x)), (format!("{}y", prefix), Column::F64(y)), (format!("{}z", prefix), Column::F64(z))]
}

fn with_suffix(base: &Path, suffix: &str) -> PathBuf {
    let mut s = base.as_os_str().to_os_string();
    s.push(suffix);
    PathBuf::from(s)
}

/// Write the three tables; returns the paths written.
pub fn write_history(model: &Model, results: &Results, base: &Path) -> Result<Vec<PathBuf>, String> {
    let mut written = Vec::new();
    let names = |idx: &[usize]| Column::Str(idx.iter().map(|k| model.accelerometers[*k].name.clone()).collect());
    let ints = |v: &[usize]| Column::I64(v.iter().map(|x| *x as i64).collect());

    // Nodes.
    let h = &results.node_history;
    let mut cols: Vec<(String, Column)> = vec![
        ("time".into(), Column::F64(h.time.clone())),
        ("step".into(), ints(&h.step)),
        ("accelerometer".into(), names(&h.accelerometer)),
        ("node".into(), ints(&h.node)),
    ];
    cols.extend(xyz("", &h.position));
    cols.extend(xyz("u", &h.displacement));
    cols.extend(xyz("v", &h.velocity));
    cols.extend(xyz("a", &h.acceleration));
    cols.extend(xyz("lu", &h.local_displacement));
    cols.extend(xyz("lv", &h.local_velocity));
    cols.extend(xyz("la", &h.local_acceleration));
    let path = with_suffix(base, ".nodes.parquet");
    let borrowed: Vec<(&str, Column)> = cols.iter().map(|(n, c)| (n.as_str(), c.clone())).collect();
    write_table(&path, "nodes", &borrowed)?;
    written.push(path);

    // Frames.
    let f = &results.frame_history;
    let mut cols: Vec<(String, Column)> = vec![("time".into(), Column::F64(f.time.clone())), ("step".into(), ints(&f.step)), ("accelerometer".into(), names(&f.accelerometer))];
    cols.extend(xyz("o", &f.origin));
    for (i, axis) in ["ex", "ey", "ez"].iter().enumerate() {
        let rows: Vec<[f64; 3]> = f.axes.iter().map(|a| a[i]).collect();
        cols.extend(xyz(&format!("{}_", axis), &rows));
    }
    let path = with_suffix(base, ".frames.parquet");
    let borrowed: Vec<(&str, Column)> = cols.iter().map(|(n, c)| (n.as_str(), c.clone())).collect();
    write_table(&path, "frames", &borrowed)?;
    written.push(path);

    // Parts.
    let mut time = Vec::new();
    let mut part = Vec::new();
    let mut mass = Vec::new();
    let mut ke = Vec::new();
    let mut disp = Vec::new();
    let mut vel = Vec::new();
    let mut accel = Vec::new();
    for (t, states) in &results.history {
        for s in states {
            time.push(*t);
            part.push(model.mesh.parts[s.part].name.clone());
            mass.push(s.mass);
            ke.push(s.kinetic_energy);
            disp.push(s.displacement);
            vel.push(s.velocity);
            accel.push(s.acceleration);
        }
    }
    let mut cols: Vec<(String, Column)> = vec![("time".into(), Column::F64(time)), ("part".into(), Column::Str(part)), ("mass".into(), Column::F64(mass))];
    cols.extend(xyz("u", &disp));
    cols.extend(xyz("v", &vel));
    cols.extend(xyz("a", &accel));
    cols.push(("kinetic_energy".into(), Column::F64(ke)));
    let path = with_suffix(base, ".parts.parquet");
    let borrowed: Vec<(&str, Column)> = cols.iter().map(|(n, c)| (n.as_str(), c.clone())).collect();
    write_table(&path, "parts", &borrowed)?;
    written.push(path);
    Ok(written)
}

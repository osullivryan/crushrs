//! TOML setup file → [`Model`].
//!
//! ```toml
//! [mesh]
//! file = "crash.inp"                   # Abaqus .inp mesh (Gmsh export)
//! # or generated blocks:
//! [[mesh.block]]
//! part = "neon"
//! origin = [0.005, -0.855, 0.0]
//! size = [4.36, 1.71, 1.35]
//! element_size = 0.3
//! faces = [{ face = "x_min", set = "neon_front" }]
//!
//! [[material]]
//! part = "neon"
//! youngs_modulus = 23.1e6
//! density = 134.53
//! yield_stress = 7248.0
//! hardening = 553354.0
//! model = "honeycomb"                  # j2 | crushable | honeycomb (default)
//!
//! [[initial_velocity]]
//! set = "neon"                         # node set or part
//! velocity = [-13.4112, 0.0, 0.0]
//!
//! [[fixed]]
//! set = "wall"
//!
//! [[contact]]                          # two-way pair of face sets
//! a = "silverado_front"
//! b = "neon_front"
//! stiffness = 2e6
//! max_distance = 0.3
//!
//! [[accelerometer]]                    # high-frequency nodal history
//! name = "neon_rear"
//! part = "neon"
//! at = [3.3, 0.0, 0.5]                 # nearest node of the part, body-fixed frame from its neighbours
//! # or explicit nodes and frame (0-based node indices):
//! # nodes = [12, 13]
//! # frame = { origin = 12, x_axis = 13, plane = 20 }
//!
//! [solver]
//! end_time = 0.15
//! frame_steps = 15
//! history_steps = 1                    # sample accelerometers / parts every step
//!
//! [output]
//! gif = "crash.gif"
//! vtk = "out/crash"                    # writes out/crash.pvd + out/crash_NNNN.vtu
//! history = "out/crash"                # writes out/crash.nodes/.frames/.parts.parquet
//! ```

use crate::material::Material;
use crate::mesh::{BlockFace, Mesh};
use crate::model::{LocalFrame, Model, Settings};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub mesh: MeshConfig,
    #[serde(default)]
    pub material: Vec<MaterialConfig>,
    #[serde(default)]
    pub initial_velocity: Vec<InitialVelocity>,
    #[serde(default)]
    pub fixed: Vec<SetRef>,
    #[serde(default)]
    pub contact: Vec<ContactConfig>,
    #[serde(default)]
    pub accelerometer: Vec<AccelerometerConfig>,
    #[serde(default)]
    pub solver: Settings,
    #[serde(default)]
    pub output: Output,
}

#[derive(Debug, Deserialize, Default)]
pub struct MeshConfig {
    pub file: Option<String>,
    #[serde(default)]
    pub block: Vec<BlockConfig>,
}

#[derive(Debug, Deserialize)]
pub struct BlockConfig {
    pub part: String,
    pub origin: [f64; 3],
    pub size: [f64; 3],
    pub element_size: f64,
    #[serde(default)]
    pub faces: Vec<FaceConfig>,
}

#[derive(Debug, Deserialize)]
pub struct FaceConfig {
    pub face: String,
    pub set: String,
}

#[derive(Debug, Deserialize)]
pub struct MaterialConfig {
    pub part: String,
    #[serde(flatten)]
    pub material: Material,
}

#[derive(Debug, Deserialize)]
pub struct InitialVelocity {
    pub set: String,
    pub velocity: [f64; 3],
}

#[derive(Debug, Deserialize)]
pub struct SetRef {
    pub set: String,
}

#[derive(Debug, Deserialize)]
pub struct ContactConfig {
    pub a: String,
    pub b: String,
    pub stiffness: f64,
    #[serde(default = "default_max_distance")]
    pub max_distance: f64,
}

fn default_max_distance() -> f64 {
    0.3
}

/// Accelerometer: either `at` (+ `part`) for an auto frame at the nearest
/// node, or explicit `nodes` / `set` with an optional `frame`.
#[derive(Debug, Deserialize)]
pub struct AccelerometerConfig {
    pub name: String,
    pub part: Option<String>,
    pub at: Option<[f64; 3]>,
    #[serde(default)]
    pub nodes: Vec<usize>,
    pub set: Option<String>,
    pub frame: Option<LocalFrame>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Output {
    pub gif: Option<String>,
    pub vtk: Option<String>,
    /// Base path for the Parquet history tables.
    pub history: Option<String>,
}

fn parse_face(s: &str) -> Result<BlockFace, String> {
    Ok(match s.to_lowercase().as_str() {
        "x_min" | "xmin" | "-x" => BlockFace::XMin,
        "x_max" | "xmax" | "+x" => BlockFace::XMax,
        "y_min" | "ymin" | "-y" => BlockFace::YMin,
        "y_max" | "ymax" | "+y" => BlockFace::YMax,
        "z_min" | "zmin" | "-z" => BlockFace::ZMin,
        "z_max" | "zmax" | "+z" => BlockFace::ZMax,
        other => return Err(format!("unknown block face '{}'", other)),
    })
}

/// Read a mesh file by extension (`.inp` Abaqus).
pub fn read_mesh(path: &Path) -> Result<Mesh, String> {
    let p = path.to_str().ok_or("bad path")?;
    match path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()).as_deref() {
        Some("inp") => crate::input::inp::read_inp(p),
        Some(other) => Err(format!("unsupported mesh format '.{}'", other)),
        None => Err("mesh file has no extension".into()),
    }
}

impl Config {
    fn accelerometer_nodes(&self, a: &AccelerometerConfig, model: &Model) -> Result<Vec<usize>, String> {
        let mut nodes = a.nodes.clone();
        if let Some(set) = &a.set {
            nodes.extend(model.mesh.nodes_of(set).ok_or_else(|| format!("accelerometer '{}': unknown set '{}'", a.name, set))?);
        }
        Ok(nodes)
    }

    pub fn from_str(text: &str) -> Result<Self, String> {
        toml::from_str(text).map_err(|e| e.to_string())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        Config::from_str(&text)
    }

    /// Build the model; relative mesh paths resolve against `base_dir`.
    pub fn build(&self, base_dir: &Path) -> Result<Model, String> {
        let mut mesh = match &self.mesh.file {
            Some(f) => {
                let p = Path::new(f);
                read_mesh(&if p.is_absolute() { p.to_path_buf() } else { base_dir.join(p) })?
            }
            None => Mesh::new(),
        };
        for b in &self.mesh.block {
            let faces: Vec<(BlockFace, &str)> = b.faces.iter().map(|f| parse_face(&f.face).map(|bf| (bf, f.set.as_str()))).collect::<Result<_, _>>()?;
            mesh.add_hex_block(&b.part, b.origin, b.size, b.element_size, &faces);
        }
        if mesh.hexes.is_empty() {
            return Err("mesh has no hexahedra".into());
        }
        let mut model = Model::new(mesh);
        model.settings = self.solver.clone();
        for m in &self.material {
            if model.mesh.part_index(&m.part).is_none() {
                return Err(format!("material for unknown part '{}'", m.part));
            }
            let mut material = m.material.clone();
            if let Some(p) = material.plasticity.as_mut() {
                p.normalise().map_err(|e| format!("material for part '{}': {}", m.part, e))?;
            }
            model.set_material(&m.part, material);
        }
        for iv in &self.initial_velocity {
            if model.mesh.nodes_of(&iv.set).is_none() {
                return Err(format!("initial velocity for unknown set '{}'", iv.set));
            }
            model.set_initial_velocity(&iv.set, iv.velocity);
        }
        for f in &self.fixed {
            if model.mesh.nodes_of(&f.set).is_none() {
                return Err(format!("fixed: unknown set '{}'", f.set));
            }
            model.fix(&f.set);
        }
        for c in &self.contact {
            for s in [&c.a, &c.b] {
                if model.mesh.face_set(s).is_none() {
                    return Err(format!("contact: unknown face set '{}'", s));
                }
            }
            model.add_contact_pair(&c.a, &c.b, c.stiffness, c.max_distance);
        }
        for a in &self.accelerometer {
            if let Some(at) = a.at {
                let part = a.part.as_deref().ok_or_else(|| format!("accelerometer '{}': 'at' needs 'part'", a.name))?;
                if model.mesh.part_index(part).is_none() {
                    return Err(format!("accelerometer '{}': unknown part '{}'", a.name, part));
                }
                model.add_accelerometer_at(&a.name, part, at);
                if !a.nodes.is_empty() || a.set.is_some() {
                    let extra = self.accelerometer_nodes(a, &model)?;
                    model.accelerometers.last_mut().unwrap().nodes.extend(extra);
                }
                if let Some(f) = a.frame {
                    model.accelerometers.last_mut().unwrap().frame = Some(f);
                }
            } else {
                let nodes = self.accelerometer_nodes(a, &model)?;
                if nodes.is_empty() {
                    return Err(format!("accelerometer '{}': give 'at', 'nodes' or 'set'", a.name));
                }
                model.add_accelerometer(&a.name, nodes, a.frame);
            }
            let n = model.mesh.nodes.len();
            let acc = model.accelerometers.last().unwrap();
            for m in acc.nodes.iter().copied().chain(acc.frame.iter().flat_map(|f| [f.origin, f.x_axis, f.plane])) {
                if m >= n {
                    return Err(format!("accelerometer '{}': node {} out of range (mesh has {} nodes)", a.name, m, n));
                }
            }
        }
        Ok(model)
    }
}

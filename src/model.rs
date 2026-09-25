//! The simulation model: mesh + materials + initial conditions + contact +
//! solver settings. Built by the TOML loader, the keyword-deck loader, or
//! directly from Rust (see `vehicle`).

use crate::material::Material;
use crate::mesh::Mesh;
use serde::{Deserialize, Serialize};

/// One-way node-to-face penalty contact: nodes of `node_set` (usually a
/// face set's nodes) against the faces of `face_set`. Use
/// [`Model::add_contact_pair`] for the usual two-way pairing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contact {
    pub nodes: Vec<usize>,
    pub faces: Vec<usize>,
    /// Penalty stiffness per contacting node (N/m).
    pub stiffness: f64,
    /// Search distance (m): nodes further than this from the faces' bounding
    /// box at start are ignored, and penetrations deeper than this are
    /// treated as spurious (the far side of a thin body).
    pub max_distance: f64,
}

/// A body-fixed coordinate system defined by three nodes, like LS-DYNA's
/// `*ELEMENT_SEATBELT_ACCELEROMETER`: local x runs from `origin` towards
/// `x_axis`, local z is normal to the plane of the three nodes, local y
/// completes the right-handed triad. Re-evaluated from the deformed
/// positions every time it is used, so it follows the body's rotation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct LocalFrame {
    pub origin: usize,
    pub x_axis: usize,
    /// A node in the local x–y plane (on the +y side).
    pub plane: usize,
}

impl LocalFrame {
    /// Rows are the local unit axes in global components.
    pub fn axes(&self, x: impl Fn(usize) -> [f64; 3]) -> [[f64; 3]; 3] {
        use nalgebra::Vector3;
        let o = Vector3::from(x(self.origin));
        let mut e1 = Vector3::from(x(self.x_axis)) - o;
        let mut p = Vector3::from(x(self.plane)) - o;
        if e1.norm() < 1e-300 {
            e1 = Vector3::x();
        }
        e1.normalize_mut();
        if p.cross(&e1).norm() < 1e-12 * p.norm().max(1e-300) {
            p = if e1.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
        }
        let mut e3 = e1.cross(&p);
        e3.normalize_mut();
        let e2 = e3.cross(&e1);
        [[e1.x, e1.y, e1.z], [e2.x, e2.y, e2.z], [e3.x, e3.y, e3.z]]
    }
}

/// High-frequency nodal history request: displacement, velocity and
/// acceleration of `nodes` every `Settings::history_steps`, in global axes
/// and, if `frame` is set, in that body-fixed frame too.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Accelerometer {
    pub name: String,
    pub nodes: Vec<usize>,
    #[serde(default)]
    pub frame: Option<LocalFrame>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Simulated duration (s).
    pub end_time: f64,
    /// Safety factor on the stability limit.
    #[serde(default = "default_dt_scale")]
    pub dt_scale: f64,
    /// Fixed time step (s); overrides the stability estimate if set.
    #[serde(default)]
    pub time_step: Option<f64>,
    /// Re-evaluate the stable step from the current element sizes every N steps (0 = never).
    #[serde(default = "default_adaptive")]
    pub adaptive_check_steps: usize,
    /// Hourglass stabilisation stiffness as a fraction of E.
    #[serde(default = "default_hourglass")]
    pub hourglass: f64,
    /// Per-step velocity scale factor (1.0 = conserve momentum).
    #[serde(default = "default_damping")]
    pub velocity_damping: f64,
    /// Capture a deformed-mesh frame every N steps (0 = never).
    #[serde(default)]
    pub frame_steps: usize,
    /// Record per-part mean motion every N steps (0 = never).
    #[serde(default)]
    pub history_steps: usize,
    /// Lane-parallel f32 kernel for honeycomb elements.
    #[serde(default = "default_true")]
    pub simd: bool,
    /// Worker threads (0 = all available).
    #[serde(default)]
    pub threads: usize,
}

fn default_dt_scale() -> f64 {
    0.85
}
fn default_adaptive() -> usize {
    5
}
fn default_hourglass() -> f64 {
    0.1
}
fn default_damping() -> f64 {
    1.0
}
fn default_true() -> bool {
    true
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            end_time: 0.15,
            dt_scale: 0.85,
            time_step: None,
            adaptive_check_steps: 5,
            hourglass: 0.1,
            velocity_damping: 1.0,
            frame_steps: 0,
            history_steps: 0,
            simd: true,
            threads: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub mesh: Mesh,
    /// Material per part.
    pub materials: Vec<Material>,
    /// Initial velocity per node.
    pub initial_velocity: Vec<[f64; 3]>,
    /// Nodes held fixed.
    pub fixed_nodes: Vec<usize>,
    pub contacts: Vec<Contact>,
    #[serde(default)]
    pub accelerometers: Vec<Accelerometer>,
    pub settings: Settings,
}

impl Model {
    pub fn new(mesh: Mesh) -> Self {
        let n = mesh.nodes.len();
        let materials = vec![Material::elastic(1.0, 0.0, 1.0); mesh.parts.len()];
        Model { mesh, materials, initial_velocity: vec![[0.0; 3]; n], fixed_nodes: Vec::new(), contacts: Vec::new(), accelerometers: Vec::new(), settings: Settings::default() }
    }

    /// Record the motion of `nodes` (global axes, plus `frame` axes if given).
    pub fn add_accelerometer(&mut self, name: &str, nodes: Vec<usize>, frame: Option<LocalFrame>) -> &mut Self {
        self.accelerometers.push(Accelerometer { name: name.to_string(), nodes, frame });
        self
    }

    /// Accelerometer at the node of `part` nearest `at`, with a body-fixed
    /// frame built from that node's neighbours so the local axes start
    /// parallel to the global ones (local x along +X, local y along +Y) and
    /// then follow the body. `at` is typically the vehicle's CG or a
    /// rear-seat / sill accelerometer location, away from the crush zone.
    pub fn add_accelerometer_at(&mut self, name: &str, part: &str, at: [f64; 3]) -> &mut Self {
        let p = self.mesh.part_index(part).unwrap_or_else(|| panic!("no part '{}'", part));
        let rec = self.mesh.nearest_node(at, Some(p));
        let radius = 1.75 * self.mesh.min_edge_length();
        // x axis: neighbour towards +X; if there is none, take the one
        // towards -X and make it the origin so local x still points +X.
        let (mut origin, mut x_axis) = (rec, rec);
        match self.mesh.best_aligned_neighbour(rec, [1.0, 0.0, 0.0], radius, Some(p), &[]) {
            Some((n, c)) if c > 0.5 => x_axis = n,
            _ => {
                if let Some((n, _)) = self.mesh.best_aligned_neighbour(rec, [-1.0, 0.0, 0.0], radius, Some(p), &[]) {
                    origin = n;
                }
            }
        }
        // plane node: neighbour of the origin towards +Y (falling back to -Y
        // with a warning that local y is then flipped).
        let plane = match self.mesh.best_aligned_neighbour(origin, [0.0, 1.0, 0.0], radius, Some(p), &[x_axis]) {
            Some((n, c)) if c > 0.5 => n,
            _ => {
                let (n, _) = self.mesh.best_aligned_neighbour(origin, [0.0, -1.0, 0.0], radius, Some(p), &[x_axis]).unwrap_or_else(|| panic!("accelerometer '{}': part '{}' has no neighbouring nodes to define a frame", name, part));
                log::warn!("accelerometer '{}': no +Y neighbour, local y/z axes are flipped", name);
                n
            }
        };
        if origin == x_axis {
            panic!("accelerometer '{}': part '{}' has no neighbouring nodes along X to define a frame", name, part);
        }
        self.add_accelerometer(name, vec![rec], Some(LocalFrame { origin, x_axis, plane }))
    }

    pub fn set_material(&mut self, part: &str, material: Material) -> &mut Self {
        let p = self.mesh.part_index(part).unwrap_or_else(|| panic!("no part '{}'", part));
        self.materials[p] = material;
        self
    }

    /// Initial velocity for a node set or part.
    pub fn set_initial_velocity(&mut self, set: &str, v: [f64; 3]) -> &mut Self {
        let nodes = self.mesh.nodes_of(set).unwrap_or_else(|| panic!("no node set or part '{}'", set));
        for n in nodes {
            self.initial_velocity[n] = v;
        }
        self
    }

    pub fn fix(&mut self, set: &str) -> &mut Self {
        let nodes = self.mesh.nodes_of(set).unwrap_or_else(|| panic!("no node set or part '{}'", set));
        self.fixed_nodes.extend(nodes);
        self
    }

    /// Two-way contact between two face sets.
    pub fn add_contact_pair(&mut self, a: &str, b: &str, stiffness: f64, max_distance: f64) -> &mut Self {
        for (p, s) in [(a, b), (b, a)] {
            let nodes = self.mesh.face_set_nodes(p).unwrap_or_else(|| panic!("no face set '{}'", p));
            let faces = self.mesh.face_set(s).unwrap_or_else(|| panic!("no face set '{}'", s)).clone();
            self.contacts.push(Contact { nodes, faces, stiffness, max_distance });
        }
        self
    }

    /// Lumped nodal masses.
    pub fn lumped_masses(&self, hourglass: f64) -> Vec<f64> {
        let mut m = vec![0.0; self.mesh.nodes.len()];
        for (e, h) in self.mesh.hexes.iter().enumerate() {
            let x = nalgebra::SMatrix::<f64, 8, 3>::from_fn(|i, d| self.mesh.nodes[h[i]][d]);
            let geo = crate::element::HexGeometry::new(&x, &self.materials[self.mesh.hex_part[e]], hourglass);
            for i in 0..8 {
                m[h[i]] += geo.lumped_mass[i];
            }
        }
        m
    }

    /// Mass and mass-weighted mean velocity of a node set.
    pub fn mean_velocity(masses: &[f64], velocity: &[[f64; 3]], nodes: &[usize]) -> (f64, [f64; 3]) {
        let mut m = 0.0;
        let mut p = [0.0; 3];
        for n in nodes {
            m += masses[*n];
            for d in 0..3 {
                p[d] += masses[*n] * velocity[*n][d];
            }
        }
        if m > 0.0 {
            (m, [p[0] / m, p[1] / m, p[2] / m])
        } else {
            (0.0, [0.0; 3])
        }
    }
}

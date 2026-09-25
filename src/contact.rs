//! Node-to-face penalty contact on the mesh's quad faces.

use crate::model::Contact;
use nalgebra::Vector3;

/// Runtime state of one one-way contact.
#[derive(Debug, Clone)]
pub struct ContactRuntime {
    nodes: Vec<usize>,
    /// Stiffness factor per entry of `nodes`.
    scales: Vec<f64>,
    faces: Vec<[usize; 4]>,
    stiffness: f64,
    max_distance: f64,
    /// Penetration beyond which the penalty force stops growing.
    depth_cap: f64,
    pub penetration_count: usize,
    pub max_penetration: f64,
}

struct Face {
    x: [Vector3<f64>; 4],
    normal: Vector3<f64>,
    center: Vector3<f64>,
    lo: Vector3<f64>,
    hi: Vector3<f64>,
}

fn point_in_triangle(p: &Vector3<f64>, a: &Vector3<f64>, b: &Vector3<f64>, c: &Vector3<f64>) -> bool {
    // Generous edge tolerance (10 % of the triangle) so a node sitting on a
    // shared edge, the diagonal of a warped quad, or wobbling just outside
    // the surface's outer edge is never missed (faces are searched in order
    // and the first hit is used, so overlaps do no harm).
    let tol = 0.1;
    let v0 = c - a;
    let v1 = b - a;
    let v2 = p - a;
    let (d00, d01, d02, d11, d12) = (v0.dot(&v0), v0.dot(&v1), v0.dot(&v2), v1.dot(&v1), v1.dot(&v2));
    let inv = 1.0 / (d00 * d11 - d01 * d01);
    let u = (d11 * d02 - d01 * d12) * inv;
    let v = (d00 * d12 - d01 * d02) * inv;
    u >= -tol && v >= -tol && u + v <= 1.0 + tol
}

impl ContactRuntime {
    /// Keep only the primary nodes within `max_distance` of the faces'
    /// initial bounding box (a static broad phase, as in the original).
    pub fn new(c: &Contact, faces: &[[usize; 4]], positions: &[[f64; 3]]) -> Self {
        let mut lo = Vector3::repeat(f64::INFINITY);
        let mut hi = Vector3::repeat(f64::NEG_INFINITY);
        for f in &c.faces {
            for n in faces[*f] {
                let p = Vector3::from(positions[n]);
                lo = lo.inf(&p);
                hi = hi.sup(&p);
            }
        }
        lo = lo.add_scalar(-c.max_distance);
        hi = hi.add_scalar(c.max_distance);
        let mut nodes = Vec::new();
        let mut scales = Vec::new();
        for (i, &n) in c.nodes.iter().enumerate() {
            let p = Vector3::from(positions[n]);
            if p >= lo && p <= hi {
                nodes.push(n);
                scales.push(c.node_scale.get(i).copied().unwrap_or(1.0));
            }
        }
        ContactRuntime { nodes, scales, faces: c.faces.iter().map(|f| faces[*f]).collect(), stiffness: c.stiffness, max_distance: c.max_distance, depth_cap: 0.1 * c.max_distance, penetration_count: 0, max_penetration: 0.0 }
    }

    /// Maximum contact acceleration a node may receive (m/s²): bounds the
    /// force on light (corner) nodes so a missed-then-caught penetration
    /// cannot kick them at hundreds of m/s.
    pub const MAX_ACCEL: f64 = 1.0e5;

    /// Viscous damping ratio on the normal relative velocity of a
    /// penetrating node (fraction of critical for the node's penalty
    /// spring). Off by default: it dissipates energy at the interface that
    /// should go into the structure (a 0.2 ratio raised the barrier KW400
    /// by 6 %).
    pub const DAMPING: f64 = 0.0;

    /// Soft-constraint factor: a node's penalty stiffness is limited to
    /// `SOFT · m_node / dt²` so light (corner) nodes get springs they can
    /// ride stably instead of being kicked (LS-DYNA SOFT=1 style).
    pub const SOFT: f64 = 0.1;

    /// Add penalty forces for the current configuration `x = X + u` with
    /// nodal velocities `v` (for the damping term); `dt` is the time step
    /// the penalty must be stable at.
    pub fn apply(&mut self, positions: &[[f64; 3]], u: &[f64], v: &[f64], masses: &[f64], dt: f64, force: &mut [f64]) {
        let cur = |n: usize| Vector3::new(positions[n][0] + u[3 * n], positions[n][1] + u[3 * n + 1], positions[n][2] + u[3 * n + 2]);
        let vel = |n: usize| Vector3::new(v[3 * n], v[3 * n + 1], v[3 * n + 2]);
        let faces: Vec<Face> = self
            .faces
            .iter()
            .map(|f| {
                let x = [cur(f[0]), cur(f[1]), cur(f[2]), cur(f[3])];
                let normal = (x[1] - x[0]).cross(&(x[2] - x[0])).normalize();
                let center = (x[0] + x[1] + x[2] + x[3]) / 4.0;
                let mut lo = x[0];
                let mut hi = x[0];
                for p in &x[1..] {
                    lo = lo.inf(p);
                    hi = hi.sup(p);
                }
                Face { x, normal, center, lo: lo.add_scalar(-self.max_distance), hi: hi.add_scalar(self.max_distance) }
            })
            .collect();
        self.penetration_count = 0;
        self.max_penetration = 0.0;
        for (&n, &scale) in self.nodes.iter().zip(&self.scales) {
            let point = cur(n);
            for (fi, face) in faces.iter().enumerate() {
                if point.x < face.lo.x || point.x > face.hi.x || point.y < face.lo.y || point.y > face.hi.y || point.z < face.lo.z || point.z > face.hi.z {
                    continue;
                }
                let signed = -face.normal.dot(&(face.center - point));
                if signed > 0.0 || signed < -self.max_distance {
                    continue;
                }
                let projected = point - face.normal * signed;
                // Both diagonal splits of the (possibly warped) quad.
                let inside = point_in_triangle(&projected, &face.x[0], &face.x[1], &face.x[2])
                    || point_in_triangle(&projected, &face.x[0], &face.x[2], &face.x[3])
                    || point_in_triangle(&projected, &face.x[1], &face.x[2], &face.x[3])
                    || point_in_triangle(&projected, &face.x[1], &face.x[3], &face.x[0]);
                if !inside {
                    continue;
                }
                // Penalty force, capped at the value for `depth_cap`
                // penetration: a node that got deep (swept by a face while
                // detection missed it) is pushed out steadily instead of
                // being kicked at hundreds of m/s.
                let depth = (-signed).min(self.depth_cap);
                let k = (self.stiffness * scale).min(Self::SOFT * masses[n] / (dt * dt));
                // Damping on the approach velocity (node into face), never
                // pulling: keeps light nodes from chattering on the spring.
                let fn_ = self.faces[fi];
                let v_face = (vel(fn_[0]) + vel(fn_[1]) + vel(fn_[2]) + vel(fn_[3])) / 4.0;
                let v_rel = face.normal.dot(&(vel(n) - v_face)); // > 0 = separating
                let damping = 2.0 * Self::DAMPING * (k * masses[n]).sqrt() * (-v_rel).max(0.0);
                let magnitude = (k * depth + damping).min(masses[n] * Self::MAX_ACCEL);
                let f = magnitude * face.normal;
                for d in 0..3 {
                    force[3 * n + d] += f[d];
                }
                // Equal and opposite reaction spread over the face's nodes.
                for m in self.faces[fi] {
                    for d in 0..3 {
                        force[3 * m + d] -= 0.25 * f[d];
                    }
                }
                self.penetration_count += 1;
                self.max_penetration = self.max_penetration.max(-signed);
                break;
            }
        }
    }
}

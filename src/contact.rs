//! Node-to-face penalty contact on the mesh's quad faces.

use crate::model::Contact;
use nalgebra::Vector3;

/// Runtime state of one one-way contact.
#[derive(Debug, Clone)]
pub struct ContactRuntime {
    nodes: Vec<usize>,
    faces: Vec<[usize; 4]>,
    stiffness: f64,
    max_distance: f64,
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
    let tol = 1e-10;
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
        let nodes = c
            .nodes
            .iter()
            .copied()
            .filter(|n| {
                let p = Vector3::from(positions[*n]);
                p >= lo && p <= hi
            })
            .collect();
        ContactRuntime { nodes, faces: c.faces.iter().map(|f| faces[*f]).collect(), stiffness: c.stiffness, max_distance: c.max_distance, penetration_count: 0, max_penetration: 0.0 }
    }

    /// Add penalty forces for the current configuration `x = X + u`.
    pub fn apply(&mut self, positions: &[[f64; 3]], u: &[f64], force: &mut [f64]) {
        let cur = |n: usize| Vector3::new(positions[n][0] + u[3 * n], positions[n][1] + u[3 * n + 1], positions[n][2] + u[3 * n + 2]);
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
        for &n in &self.nodes {
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
                if !(point_in_triangle(&projected, &face.x[0], &face.x[1], &face.x[2]) || point_in_triangle(&projected, &face.x[0], &face.x[2], &face.x[3])) {
                    continue;
                }
                let f = self.stiffness * (-signed) * face.normal;
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

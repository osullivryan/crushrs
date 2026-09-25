//! 8-node hexahedron geometry for one-point (mean-strain) integration with
//! hourglass stabilisation: everything computed once per element in the
//! reference configuration.

use crate::material::Material;
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

/// Standard C3D8 parametric node coordinates.
pub const XI: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

const GAUSS: f64 = 0.577_350_269_189_625_8;

/// ∂N/∂ξ (8 × 3) at a parametric point.
pub fn shape_derivatives(xi: f64, eta: f64, zeta: f64) -> SMatrix<f64, 8, 3> {
    SMatrix::<f64, 8, 3>::from_fn(|i, d| {
        let s = XI[i];
        match d {
            0 => 0.125 * s[0] * (1.0 + s[1] * eta) * (1.0 + s[2] * zeta),
            1 => 0.125 * s[1] * (1.0 + s[0] * xi) * (1.0 + s[2] * zeta),
            _ => 0.125 * s[2] * (1.0 + s[0] * xi) * (1.0 + s[1] * eta),
        }
    })
}

/// Shape functions at a parametric point.
pub fn shape_functions(xi: f64, eta: f64, zeta: f64) -> SVector<f64, 8> {
    SVector::<f64, 8>::from_fn(|i, _| {
        let s = XI[i];
        0.125 * (1.0 + s[0] * xi) * (1.0 + s[1] * eta) * (1.0 + s[2] * zeta)
    })
}

/// ∂N/∂X (8 × 3) and det(J) at a parametric point for nodal positions `x`.
pub fn reference_gradients(x: &SMatrix<f64, 8, 3>, d_n: &SMatrix<f64, 8, 3>) -> (SMatrix<f64, 8, 3>, f64) {
    let j: Matrix3<f64> = d_n.transpose() * x;
    let j_inv = j.try_inverse().expect("degenerate hexahedron (singular Jacobian)");
    (d_n * j_inv.transpose(), j.determinant())
}

/// Small-strain B matrix (6 × 24, engineering shear) from ∂N/∂X.
pub fn b_matrix(grad: &SMatrix<f64, 8, 3>) -> SMatrix<f64, 6, 24> {
    let mut b = SMatrix::<f64, 6, 24>::zeros();
    for i in 0..8 {
        let n = [grad[(i, 0)], grad[(i, 1)], grad[(i, 2)]];
        b[(0, 3 * i)] = n[0];
        b[(1, 3 * i + 1)] = n[1];
        b[(2, 3 * i + 2)] = n[2];
        b[(3, 3 * i)] = n[1];
        b[(3, 3 * i + 1)] = n[0];
        b[(4, 3 * i + 1)] = n[2];
        b[(4, 3 * i + 2)] = n[1];
        b[(5, 3 * i)] = n[2];
        b[(5, 3 * i + 2)] = n[0];
    }
    b
}

/// Hourglass stabilisation stiffness `K = Σ_gp w (B_gp − B_c)ᵀ C_s (B_gp − B_c)`
/// in its exact rank-12 form `K = Γᵀ H Γ`: `Γ` maps the 24 nodal
/// displacements onto the 4 Flanagan–Belytschko hourglass modes × 3
/// directions (`γ_α` orthogonal to every uniform-gradient field, so the
/// stabilisation is exactly zero on uniform strain and rigid rotation), and
/// `H` is 12 × 12.
#[derive(Debug, Clone, PartialEq)]
pub struct Hourglass {
    /// γ_α as columns (8 nodes × 4 modes).
    pub gamma: SMatrix<f64, 8, 4>,
    /// Coordinates ordered q[3·mode + direction].
    pub h: SMatrix<f64, 12, 12>,
}

impl Hourglass {
    /// Hourglass base vectors for the standard node ordering.
    pub const MODES: [[f64; 8]; 4] = [
        [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0],
    ];

    /// From the dense stabilisation matrix.
    pub fn from_dense(x: &SMatrix<f64, 8, 3>, center_grad: &SMatrix<f64, 8, 3>, k: &SMatrix<f64, 24, 24>) -> Self {
        let mut gamma = SMatrix::<f64, 8, 4>::zeros();
        for a in 0..4 {
            let mut hx = Vector3::zeros();
            for j in 0..8 {
                hx += Self::MODES[a][j] * Vector3::new(x[(j, 0)], x[(j, 1)], x[(j, 2)]);
            }
            for i in 0..8 {
                let b = Vector3::new(center_grad[(i, 0)], center_grad[(i, 1)], center_grad[(i, 2)]);
                gamma[(i, a)] = Self::MODES[a][i] - hx.dot(&b);
            }
        }
        let mut g = SMatrix::<f64, 12, 24>::zeros();
        for a in 0..4 {
            for i in 0..8 {
                for d in 0..3 {
                    g[(3 * a + d, 3 * i + d)] = gamma[(i, a)];
                }
            }
        }
        let ggt_inv = (g * g.transpose()).try_inverse().expect("hourglass modes independent");
        let h = ggt_inv * g * k * g.transpose() * ggt_inv;
        Hourglass { gamma, h }
    }

    /// Adds the stabilisation force for nodal displacements `u`.
    pub fn add_force(&self, u: &SVector<f64, 24>, f: &mut SVector<f64, 24>) {
        let u_mat = SMatrix::<f64, 3, 8>::from_column_slice(u.as_slice());
        let q: SMatrix<f64, 3, 4> = u_mat * self.gamma;
        let r: SVector<f64, 12> = self.h * SVector::<f64, 12>::from_column_slice(q.as_slice());
        let f_mat: SMatrix<f64, 3, 8> = SMatrix::<f64, 3, 4>::from_column_slice(r.as_slice()) * self.gamma.transpose();
        for k in 0..24 {
            f[k] += f_mat.as_slice()[k];
        }
    }
}

/// Reference-configuration data of one hexahedron.
#[derive(Debug, Clone, PartialEq)]
pub struct HexGeometry {
    /// Mean gradient ∂N/∂X at the centre (8 × 3).
    pub center_grad: SMatrix<f64, 8, 3>,
    pub volume: f64,
    /// Lumped nodal masses (row sums of the consistent mass matrix).
    pub lumped_mass: [f64; 8],
    pub hourglass: Hourglass,
}

impl HexGeometry {
    /// `hourglass` is the stabilisation stiffness as a fraction of E.
    pub fn new(x: &SMatrix<f64, 8, 3>, material: &Material, hourglass: f64) -> Self {
        let d_n_c = shape_derivatives(0.0, 0.0, 0.0);
        let (center_grad, _) = reference_gradients(x, &d_n_c);
        let c_s = crate::material::elastic_matrix(hourglass * material.youngs_modulus, material.poisson_ratio);
        let b_c = b_matrix(&center_grad);
        let mut k = SMatrix::<f64, 24, 24>::zeros();
        let mut volume = 0.0;
        let mut mass = SMatrix::<f64, 8, 8>::zeros();
        for s in XI {
            let (xi, eta, zeta) = (s[0] * GAUSS, s[1] * GAUSS, s[2] * GAUSS);
            let d_n = shape_derivatives(xi, eta, zeta);
            let (g, det_j) = reference_gradients(x, &d_n);
            let db = b_matrix(&g) - b_c;
            k += db.transpose() * c_s * db * det_j;
            volume += det_j;
            let n = shape_functions(xi, eta, zeta);
            mass += material.density * n * n.transpose() * det_j;
        }
        let lumped_mass = std::array::from_fn(|i| mass.row(i).sum());
        HexGeometry { center_grad, volume, lumped_mass, hourglass: Hourglass::from_dense(x, &center_grad, &k) }
    }

    /// Mean deformation gradient for current nodal positions (8 × 3).
    pub fn deformation_gradient(&self, x: &SMatrix<f64, 8, 3>) -> Matrix3<f64> {
        x.transpose() * self.center_grad
    }
}

/// Current characteristic length V / A_max of a hexahedron (LS-DYNA
/// convention), from its 8 current node positions; +∞ if inverted.
pub fn characteristic_length(p: &[Vector3<f64>; 8]) -> f64 {
    const TETS: [[usize; 4]; 6] = [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]];
    const FACES: [[usize; 4]; 6] = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]];
    let vol: f64 = TETS.iter().map(|t| (p[t[1]] - p[t[0]]).cross(&(p[t[2]] - p[t[0]])).dot(&(p[t[3]] - p[t[0]])) / 6.0).sum();
    let a_max = FACES.iter().map(|q| 0.5 * (p[q[2]] - p[q[0]]).cross(&(p[q[3]] - p[q[1]])).norm()).fold(0.0_f64, f64::max);
    if vol <= 0.0 || a_max <= 0.0 {
        f64::INFINITY
    } else {
        vol / a_max
    }
}

/// Volume of a hexahedron from its 8 node positions (6-tet split).
pub fn hex_volume(p: &[Vector3<f64>; 8]) -> f64 {
    const TETS: [[usize; 4]; 6] = [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]];
    TETS.iter().map(|t| (p[t[1]] - p[t[0]]).cross(&(p[t[2]] - p[t[0]])).dot(&(p[t[3]] - p[t[0]])) / 6.0).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(l: f64) -> SMatrix<f64, 8, 3> {
        SMatrix::<f64, 8, 3>::from_fn(|i, d| 0.5 * l * (XI[i][d] + 1.0))
    }

    #[test]
    fn cube_geometry() {
        let m = Material::honeycomb(1e6, 100.0, 1e3, 1e5);
        let g = HexGeometry::new(&cube(0.3), &m, 0.1);
        assert!((g.volume - 0.027).abs() < 1e-12);
        for mi in g.lumped_mass {
            assert!((mi - 100.0 * 0.027 / 8.0).abs() < 1e-12);
        }
        // Mean gradient of a cube: ∂N_i/∂x = ±1/(4L).
        assert!((g.center_grad[(0, 0)] + 1.0 / 1.2).abs() < 1e-12);
    }

    #[test]
    fn rank12_equals_dense_on_distorted_hex() {
        let x = SMatrix::<f64, 8, 3>::from_row_slice(&[
            0.0, 0.0, 0.0, 1.2, 0.1, 0.0, 1.1, 0.9, 0.1, -0.1, 1.0, 0.0, 0.1, 0.0, 1.0, 1.0, 0.2, 1.1, 1.3, 1.1, 0.9, 0.0, 0.8, 1.2,
        ]);
        let m = Material::crushable_foam(20e6, 200.0, 0.2e6, 0.5e6);
        let geo = HexGeometry::new(&x, &m, 0.1);
        let c = crate::material::elastic_matrix(0.1 * 20e6, 0.0);
        let b_c = b_matrix(&geo.center_grad);
        let mut k = SMatrix::<f64, 24, 24>::zeros();
        for s in XI {
            let d_n = shape_derivatives(s[0] * GAUSS, s[1] * GAUSS, s[2] * GAUSS);
            let (g, det_j) = reference_gradients(&x, &d_n);
            let db = b_matrix(&g) - b_c;
            k += db.transpose() * c * db * det_j;
        }
        let u = SVector::<f64, 24>::from_fn(|i, _| ((i * 7 % 11) as f64 - 5.0) * 1e-3);
        let mut f = SVector::<f64, 24>::zeros();
        geo.hourglass.add_force(&u, &mut f);
        let dense = k * u;
        assert!((f - dense).norm() < 1e-9 * dense.norm());
        // Uniform gradient field (including rotation) → zero stabilisation force.
        let a = nalgebra::Rotation3::from_euler_angles(0.3, -0.7, 1.1).into_inner() * Matrix3::new(1.1, 0.05, 0.0, 0.0, 0.9, 0.02, 0.0, 0.0, 1.05);
        let mut u2 = SVector::<f64, 24>::zeros();
        for i in 0..8 {
            let x0 = Vector3::new(x[(i, 0)], x[(i, 1)], x[(i, 2)]);
            let xn = a * x0;
            for d in 0..3 {
                u2[3 * i + d] = xn[d] - x0[d];
            }
        }
        let mut f2 = SVector::<f64, 24>::zeros();
        geo.hourglass.add_force(&u2, &mut f2);
        assert!(f2.norm() < 1e-8 * dense.norm(), "uniform field produced stabilisation force {}", f2.norm());
    }
}

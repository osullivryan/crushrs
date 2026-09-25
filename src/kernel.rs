//! Contiguous element storage and the force kernels.
//!
//! [`Elements`] holds every hexahedron's hot data in struct-of-arrays form.
//! Honeycomb elements run through the lane-parallel f32 kernel
//! (`kernel_simd`) when enabled; everything else (and the honeycomb scalar
//! reference) runs through the per-element f64 path here.

use crate::element::{characteristic_length_and_volume, HexGeometry};
use crate::material::{finite_strain_return, honeycomb_rate_update, FinitePlasticPoint, HoneycombParams, Material, PlasticModel, RateFoamState};
use crate::model::Model;
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use rayon::prelude::*;

/// Material state of one element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum State {
    Rate(RateFoamState),
    Hencky(FinitePlasticPoint),
    Elastic,
}

#[derive(Debug, Default)]
pub struct Elements {
    pub conn: Vec<[u32; 8]>,
    /// Mean gradient, row-major (node, component).
    pub center_grad: Vec<[f64; 24]>,
    pub volume: Vec<f64>,
    pub gamma: Vec<SMatrix<f64, 8, 4>>,
    pub h: Vec<SMatrix<f64, 12, 12>>,
    pub material: Vec<Material>,
    /// Stability factor f(ν)/c_d (s/m).
    pub dt_factor: Vec<f64>,
    pub state: Vec<State>,
    pub eroded: Vec<bool>,
    /// Reference node positions.
    pub positions: Vec<[f64; 3]>,
    /// Elements handled by the SIMD kernel (indices) and by the scalar path.
    pub simd_ids: Vec<usize>,
    pub scalar_ids: Vec<usize>,
    simd: Option<crate::kernel_simd::Blocks>,
    force: Vec<[f64; 24]>,
}

impl Elements {
    pub fn len(&self) -> usize {
        self.conn.len()
    }

    pub fn is_empty(&self) -> bool {
        self.conn.is_empty()
    }

    pub fn build(model: &Model, simd: bool) -> Self {
        let mesh = &model.mesh;
        let mut el = Elements::default();
        el.positions = mesh.nodes.clone();
        for (e, h) in mesh.hexes.iter().enumerate() {
            let mat = model.materials[mesh.hex_part[e]].clone();
            let x = SMatrix::<f64, 8, 3>::from_fn(|i, d| mesh.nodes[h[i]][d]);
            let geo = HexGeometry::new(&x, &mat, model.settings.hourglass);
            el.conn.push(std::array::from_fn(|i| h[i] as u32));
            let mut g = [0.0; 24];
            for i in 0..8 {
                for d in 0..3 {
                    g[3 * i + d] = geo.center_grad[(i, d)];
                }
            }
            el.center_grad.push(g);
            el.volume.push(geo.volume);
            el.gamma.push(geo.hourglass.gamma);
            el.h.push(geo.hourglass.h);
            el.material.push(mat.clone());
            el.dt_factor.push((1.0 - 0.36 * mat.poisson_ratio) / mat.dilatational_wave_speed());
            el.state.push(match mat.model() {
                Some(PlasticModel::Honeycomb) => State::Rate(RateFoamState::default()),
                Some(_) => State::Hencky(FinitePlasticPoint::default()),
                None => State::Elastic,
            });
            el.eroded.push(false);
            if simd && mat.model() == Some(PlasticModel::Honeycomb) {
                el.simd_ids.push(e);
            } else {
                el.scalar_ids.push(e);
            }
        }
        el.force = vec![[0.0; 24]; el.len()];
        if !el.simd_ids.is_empty() {
            el.simd = Some(crate::kernel_simd::Blocks::build(&el));
        }
        el
    }

    pub fn uses_simd(&self) -> bool {
        self.simd.is_some()
    }

    /// Kirchhoff stress of one element at deformation gradient `f`, committing the state.
    fn stress_update(material: &Material, f: &Matrix3<f64>, state: &mut State) -> Matrix3<f64> {
        match state {
            State::Rate(s) => {
                let (tau, ns) = honeycomb_rate_update(&HoneycombParams::from_material(material), f, s);
                *s = ns;
                tau
            }
            State::Hencky(s) => {
                let (tau, ns) = finite_strain_return(material, f, s);
                *s = ns;
                tau
            }
            State::Elastic => finite_strain_return(material, f, &FinitePlasticPoint::default()).0,
        }
    }

    /// Scalar f64 force of one element; `false` if it inverted.
    #[inline]
    fn element_force(u: &[f64], positions: &[[f64; 3]], conn: &[u32; 8], grad: &[f64; 24], volume: f64, gamma: &SMatrix<f64, 8, 4>, h: &SMatrix<f64, 12, 12>, material: &Material, state: &mut State, out: &mut [f64; 24]) -> bool {
        let mut x = [[0.0; 3]; 8];
        let mut ul = [0.0; 24];
        for i in 0..8 {
            let n = conn[i] as usize;
            let p = &positions[n];
            let d = &u[3 * n..3 * n + 3];
            x[i] = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
            ul[3 * i] = d[0];
            ul[3 * i + 1] = d[1];
            ul[3 * i + 2] = d[2];
        }
        let mut f = Matrix3::<f64>::zeros();
        for i in 0..8 {
            for a in 0..3 {
                for b in 0..3 {
                    f[(a, b)] += x[i][a] * grad[3 * i + b];
                }
            }
        }
        if f.determinant() <= 0.0 {
            *out = [0.0; 24];
            return false;
        }
        let tau = Self::stress_update(material, &f, state);
        let m = tau * f.try_inverse().unwrap().transpose() * volume;
        for i in 0..8 {
            let fi = m * Vector3::new(grad[3 * i], grad[3 * i + 1], grad[3 * i + 2]);
            out[3 * i] = fi[0];
            out[3 * i + 1] = fi[1];
            out[3 * i + 2] = fi[2];
        }
        let u_mat = SMatrix::<f64, 3, 8>::from_column_slice(&ul);
        let q: SMatrix<f64, 3, 4> = u_mat * gamma;
        let r: SVector<f64, 12> = h * SVector::<f64, 12>::from_column_slice(q.as_slice());
        let f_hg: SMatrix<f64, 3, 8> = SMatrix::<f64, 3, 4>::from_column_slice(r.as_slice()) * gamma.transpose();
        for k in 0..24 {
            out[k] += f_hg.as_slice()[k];
        }
        true
    }

    /// Element forces for displacement `u` (commits states), added into
    /// `global`. Returns the number of elements that inverted this call.
    pub fn compute_forces(&mut self, u: &[f64], global: &mut [f64], threads: usize) -> usize {
        let mut newly_eroded = 0;
        if let Some(blocks) = &mut self.simd {
            newly_eroded += blocks.compute_forces(u, &self.positions, threads);
            blocks.scatter(global);
            blocks.mirror_erosion(&self.simd_ids, &mut self.eroded);
        }
        if !self.scalar_ids.is_empty() {
            let positions = &self.positions;
            let (conn, grad, volume, gamma, h, material) = (&self.conn, &self.center_grad, &self.volume, &self.gamma, &self.h, &self.material);
            // Gather the scalar elements' mutable state and output slots.
            let ids = &self.scalar_ids;
            let mut states: Vec<(State, bool, [f64; 24])> = ids.iter().map(|&e| (self.state[e], self.eroded[e], [0.0; 24])).collect();
            let work = |(&e, (st, er, out)): (&usize, &mut (State, bool, [f64; 24]))| -> usize {
                if *er {
                    *out = [0.0; 24];
                    return 0;
                }
                let ok = Self::element_force(u, positions, &conn[e], &grad[e], volume[e], &gamma[e], &h[e], &material[e], st, out);
                if !ok {
                    *er = true;
                    1
                } else {
                    0
                }
            };
            if threads > 1 {
                newly_eroded += ids.par_iter().zip(states.par_iter_mut()).with_min_len(32).map(work).sum::<usize>();
            } else {
                newly_eroded += ids.iter().zip(states.iter_mut()).map(work).sum::<usize>();
            }
            for (&e, (st, er, out)) in ids.iter().zip(states.iter()) {
                self.state[e] = *st;
                self.eroded[e] = *er;
                self.force[e] = *out;
                for i in 0..8 {
                    let n = self.conn[e][i] as usize;
                    for d in 0..3 {
                        global[3 * n + d] += out[3 * i + d];
                    }
                }
            }
        }
        newly_eroded
    }

    /// Explicit stability limit min(f(ν)·L/c_d) from the current geometry.
    pub fn stable_time_step(&self, u: &[f64]) -> Option<f64> {
        let mut dt = f64::INFINITY;
        for (e, conn) in self.conn.iter().enumerate() {
            if self.eroded[e] {
                continue;
            }
            let p: [Vector3<f64>; 8] = std::array::from_fn(|i| {
                let n = conn[i] as usize;
                Vector3::new(self.positions[n][0] + u[3 * n], self.positions[n][1] + u[3 * n + 1], self.positions[n][2] + u[3 * n + 2])
            });
            // dt = L / c₀ with the current characteristic length V/A_max.
            // (Using the current density, c = c₀·√J, would allow
            // L/√J — up to 2× larger for heavily crushed elements — but
            // proved marginal for locked-up honeycomb, so the initial wave
            // speed is kept: conservative for crushed elements.)
            let (l, vol) = characteristic_length_and_volume(&p);
            if l.is_finite() && vol > 0.0 {
                dt = dt.min(l * self.dt_factor[e]);
            }
        }
        dt.is_finite().then_some(dt)
    }

    /// Pull SIMD states into `state` (for output).
    pub fn sync_states(&mut self) {
        if let Some(blocks) = &self.simd {
            for (k, &e) in self.simd_ids.iter().enumerate() {
                self.state[e] = State::Rate(blocks.state(k));
            }
        }
    }

    /// Equivalent plastic strain / compaction per element (call `sync_states` first).
    pub fn plastic_strain(&self) -> Vec<f64> {
        self.state
            .iter()
            .map(|s| match s {
                State::Rate(r) => r.compaction,
                State::Hencky(h) => h.eq_plastic_strain,
                State::Elastic => 0.0,
            })
            .collect()
    }

    /// Cauchy stress (Voigt) per element for output, from the committed
    /// state (call `sync_states` first). Uses the current displacement for
    /// the Hencky/elastic models.
    pub fn cauchy_stress(&self, u: &[f64]) -> Vec<[f64; 6]> {
        (0..self.len())
            .map(|e| {
                if self.eroded[e] {
                    return [0.0; 6];
                }
                match self.state[e] {
                    State::Rate(s) => {
                        let sh = Matrix3::new(s.stress[0], s.stress[3], s.stress[5], s.stress[3], s.stress[1], s.stress[4], s.stress[5], s.stress[4], s.stress[2]);
                        let c = s.rotation * sh * s.rotation.transpose();
                        [c[(0, 0)], c[(1, 1)], c[(2, 2)], c[(0, 1)], c[(1, 2)], c[(0, 2)]]
                    }
                    st => {
                        let conn = &self.conn[e];
                        let grad = &self.center_grad[e];
                        let mut f = Matrix3::<f64>::zeros();
                        for i in 0..8 {
                            let n = conn[i] as usize;
                            for a in 0..3 {
                                let xa = self.positions[n][a] + u[3 * n + a];
                                for b in 0..3 {
                                    f[(a, b)] += xa * grad[3 * i + b];
                                }
                            }
                        }
                        let j = f.determinant();
                        if j <= 0.0 {
                            return [0.0; 6];
                        }
                        let mut s = st;
                        let tau = Self::stress_update(&self.material[e], &f, &mut s);
                        let c = tau / j;
                        [c[(0, 0)], c[(1, 1)], c[(2, 2)], c[(0, 1)], c[(1, 2)], c[(0, 2)]]
                    }
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::XI;
    use nalgebra::Rotation3;

    fn single(material: Material) -> (Elements, Vec<[f64; 3]>) {
        let mut mesh = crate::mesh::Mesh::new();
        let p = mesh.add_part("a");
        mesh.nodes = XI.iter().map(|s| [0.5 * (s[0] + 1.0), 0.5 * (s[1] + 1.0), 0.5 * (s[2] + 1.0)]).collect();
        mesh.hexes.push([0, 1, 2, 3, 4, 5, 6, 7]);
        mesh.hex_part.push(p);
        let mut model = Model::new(mesh);
        model.materials[0] = material;
        let el = Elements::build(&model, false);
        let nodes = model.mesh.nodes.clone();
        (el, nodes)
    }

    fn displacement(nodes: &[[f64; 3]], a: &Matrix3<f64>, c: Vector3<f64>) -> Vec<f64> {
        let mut u = vec![0.0; 24];
        for i in 0..8 {
            let x0 = Vector3::from(nodes[i]);
            let x = a * x0 + c;
            for d in 0..3 {
                u[3 * i + d] = x[d] - x0[d];
            }
        }
        u
    }

    #[test]
    fn rigid_motion_gives_zero_force() {
        for mat in [Material::honeycomb(20e6, 200.0, 0.2e6, 0.5e6), Material::j2(200e9, 0.3, 7800.0, 250e6, 1e9)] {
            let (mut el, nodes) = single(mat);
            let r = Rotation3::from_euler_angles(0.4, 1.2, -0.8).into_inner();
            let u = displacement(&nodes, &r, Vector3::new(3.0, -1.0, 2.0));
            let mut g = vec![0.0; 24];
            el.compute_forces(&u, &mut g, 1);
            let norm: f64 = g.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(norm < 1e-3, "rigid motion produced force {}", norm);
        }
    }

    #[test]
    fn uniform_crush_matches_material_law() {
        // Honeycomb cube crushed 40 % along x in 100 steps: face force = σ_y(c)·A (true stress).
        let (mut el, nodes) = single(Material::honeycomb(20e6, 200.0, 0.2e6, 0.5e6));
        let mut g = vec![0.0; 24];
        for k in 1..=100 {
            let a = Matrix3::new(1.0 - 0.4 * k as f64 / 100.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
            let u = displacement(&nodes, &a, Vector3::zeros());
            g = vec![0.0; 24];
            el.compute_forces(&u, &mut g, 1);
        }
        let fx_face: f64 = [1usize, 2, 5, 6].iter().map(|&i| g[3 * i]).sum();
        let c = el.plastic_strain()[0];
        let sigma_y = 0.2e6 + 0.5e6 * c;
        assert!((fx_face + sigma_y).abs() < 1e-6 * sigma_y, "{} vs {}", fx_face, -sigma_y);
        // Compaction = total log strain minus the elastic part σ_y(c)/E.
        let expected = -(0.6_f64).ln() - sigma_y / 20e6;
        assert!((c - expected).abs() < 2e-3, "compaction {} vs {}", c, expected);
    }

    #[test]
    fn hourglass_mode_is_resisted() {
        let (mut el, _) = single(Material::honeycomb(20e6, 200.0, 0.2e6, 0.5e6));
        let hmode = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let mut u = vec![0.0; 24];
        for i in 0..8 {
            u[3 * i] = 1e-3 * hmode[i];
        }
        let mut g = vec![0.0; 24];
        el.compute_forces(&u, &mut g, 1);
        let norm: f64 = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 1.0, "hourglass mode not resisted: {}", norm);
        assert_eq!(el.plastic_strain()[0], 0.0, "pure hourglass mode has zero mean strain");
    }
}

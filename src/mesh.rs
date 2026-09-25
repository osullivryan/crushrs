//! Mesh: nodes, 8-node hexahedra grouped into parts, contact faces, and
//! named node / face sets. One flat representation, indexed by position.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A part: a named group of hexahedra sharing one material.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Part {
    pub name: String,
}

/// Face of an axis-aligned block (for generated meshes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockFace {
    XMin,
    XMax,
    YMin,
    YMax,
    ZMin,
    ZMax,
}

impl BlockFace {
    fn axis_and_side(self) -> (usize, bool) {
        match self {
            BlockFace::XMin => (0, false),
            BlockFace::XMax => (0, true),
            BlockFace::YMin => (1, false),
            BlockFace::YMax => (1, true),
            BlockFace::ZMin => (2, false),
            BlockFace::ZMax => (2, true),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Mesh {
    /// Reference node positions.
    pub nodes: Vec<[f64; 3]>,
    /// Hexahedra, standard ordering (bottom face counter-clockwise, then top).
    pub hexes: Vec<[usize; 8]>,
    /// Part index of each hexahedron.
    pub hex_part: Vec<usize>,
    pub parts: Vec<Part>,
    /// Contact faces (quads, counter-clockwise about the outward normal).
    pub faces: Vec<[usize; 4]>,
    /// Named node sets.
    pub node_sets: BTreeMap<String, Vec<usize>>,
    /// Named face sets (indices into `faces`).
    pub face_sets: BTreeMap<String, Vec<usize>>,
}

impl Mesh {
    pub fn new() -> Self {
        Mesh::default()
    }

    pub fn add_part(&mut self, name: &str) -> usize {
        self.parts.push(Part { name: name.to_string() });
        self.parts.len() - 1
    }

    pub fn part_index(&self, name: &str) -> Option<usize> {
        self.parts.iter().position(|p| p.name == name)
    }

    /// Hexahedra of a part.
    pub fn part_hexes(&self, part: usize) -> Vec<usize> {
        (0..self.hexes.len()).filter(|e| self.hex_part[*e] == part).collect()
    }

    /// Nodes of a part (sorted, unique).
    pub fn part_nodes(&self, part: usize) -> Vec<usize> {
        let mut set = std::collections::BTreeSet::new();
        for (e, h) in self.hexes.iter().enumerate() {
            if self.hex_part[e] == part {
                set.extend(h.iter().copied());
            }
        }
        set.into_iter().collect()
    }

    /// Nodes of a named node set or, failing that, of a part by name.
    pub fn nodes_of(&self, name: &str) -> Option<Vec<usize>> {
        if let Some(s) = self.node_sets.get(name) {
            return Some(s.clone());
        }
        self.part_index(name).map(|p| self.part_nodes(p))
    }

    pub fn face_set(&self, name: &str) -> Option<&Vec<usize>> {
        self.face_sets.get(name)
    }

    /// Nodes referenced by a face set (sorted, unique).
    pub fn face_set_nodes(&self, name: &str) -> Option<Vec<usize>> {
        let set = self.face_sets.get(name)?;
        let mut nodes = std::collections::BTreeSet::new();
        for f in set {
            nodes.extend(self.faces[*f].iter().copied());
        }
        Some(nodes.into_iter().collect())
    }

    /// Append an axis-aligned block of hexahedra as a new part.
    ///
    /// `surfaces` adds contact faces on the given block faces with outward
    /// normals, as face sets named accordingly. The part's nodes are also
    /// registered as a node set with the part's name.
    pub fn add_hex_block(&mut self, part_name: &str, origin: [f64; 3], size: [f64; 3], element_size: f64, surfaces: &[(BlockFace, &str)]) -> usize {
        let n: [usize; 3] = std::array::from_fn(|a| (size[a] / element_size).round().max(1.0) as usize);
        let d: [f64; 3] = std::array::from_fn(|a| size[a] / n[a] as f64);
        let node0 = self.nodes.len();
        let id = |ijk: [usize; 3]| node0 + ijk[0] + (n[0] + 1) * (ijk[1] + (n[1] + 1) * ijk[2]);
        for k in 0..=n[2] {
            for j in 0..=n[1] {
                for i in 0..=n[0] {
                    self.nodes.push([origin[0] + i as f64 * d[0], origin[1] + j as f64 * d[1], origin[2] + k as f64 * d[2]]);
                }
            }
        }
        let part = self.add_part(part_name);
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    self.hexes.push([
                        id([i, j, k]),
                        id([i + 1, j, k]),
                        id([i + 1, j + 1, k]),
                        id([i, j + 1, k]),
                        id([i, j, k + 1]),
                        id([i + 1, j, k + 1]),
                        id([i + 1, j + 1, k + 1]),
                        id([i, j + 1, k + 1]),
                    ]);
                    self.hex_part.push(part);
                }
            }
        }
        for (face, set_name) in surfaces {
            let (a, max_side) = face.axis_and_side();
            let (b, c) = ((a + 1) % 3, (a + 2) % 3);
            let fixed = if max_side { n[a] } else { 0 };
            let at = |ib: usize, ic: usize| {
                let mut ijk = [0; 3];
                ijk[a] = fixed;
                ijk[b] = ib;
                ijk[c] = ic;
                id(ijk)
            };
            let mut set = Vec::new();
            for ic in 0..n[c] {
                for ib in 0..n[b] {
                    let (p0, p1, p2, p3) = (at(ib, ic), at(ib + 1, ic), at(ib + 1, ic + 1), at(ib, ic + 1));
                    self.faces.push(if max_side { [p0, p1, p2, p3] } else { [p0, p3, p2, p1] });
                    set.push(self.faces.len() - 1);
                }
            }
            self.face_sets.insert(set_name.to_string(), set);
        }
        let all: Vec<usize> = (node0..self.nodes.len()).collect();
        self.node_sets.insert(part_name.to_string(), all);
        part
    }

    /// Smallest edge length over all hexahedra.
    /// Node nearest a point, optionally restricted to a part.
    pub fn nearest_node(&self, at: [f64; 3], part: Option<usize>) -> usize {
        let candidates: Vec<usize> = match part {
            Some(p) => self.part_nodes(p),
            None => (0..self.nodes.len()).collect(),
        };
        let d2 = |n: usize| (0..3).map(|d| (self.nodes[n][d] - at[d]).powi(2)).sum::<f64>();
        candidates.into_iter().min_by(|a, b| d2(*a).partial_cmp(&d2(*b)).unwrap()).expect("mesh has no nodes")
    }

    /// Among the nodes of `part` within `radius` of `node` (excluding
    /// `exclude`), the one whose direction from `node` is best aligned with
    /// `dir`; returns it with the cosine of the alignment.
    pub fn best_aligned_neighbour(&self, node: usize, dir: [f64; 3], radius: f64, part: Option<usize>, exclude: &[usize]) -> Option<(usize, f64)> {
        let candidates: Vec<usize> = match part {
            Some(p) => self.part_nodes(p),
            None => (0..self.nodes.len()).collect(),
        };
        let o = self.nodes[node];
        let dn = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        let mut best: Option<(usize, f64)> = None;
        for n in candidates {
            if n == node || exclude.contains(&n) {
                continue;
            }
            let r = [self.nodes[n][0] - o[0], self.nodes[n][1] - o[1], self.nodes[n][2] - o[2]];
            let rn = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            if rn > radius || rn == 0.0 {
                continue;
            }
            let c = (r[0] * dir[0] + r[1] * dir[1] + r[2] * dir[2]) / (rn * dn);
            if best.map_or(true, |(_, bc)| c > bc) {
                best = Some((n, c));
            }
        }
        best
    }

    pub fn min_edge_length(&self) -> f64 {
        const EDGES: [[usize; 2]; 12] = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
        let mut min = f64::INFINITY;
        for h in &self.hexes {
            for [a, b] in EDGES {
                let (p, q) = (self.nodes[h[a]], self.nodes[h[b]]);
                let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
                min = min.min(d);
            }
        }
        min
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_generator_counts_and_faces() {
        let mut m = Mesh::new();
        m.add_hex_block("a", [0.0, 0.0, 0.0], [1.0, 0.5, 0.5], 0.25, &[(BlockFace::XMax, "a_front")]);
        assert_eq!(m.hexes.len(), 4 * 2 * 2);
        assert_eq!(m.nodes.len(), 5 * 3 * 3);
        assert_eq!(m.face_set("a_front").unwrap().len(), 4);
        // Outward normal of the +x face points +x.
        let f = m.faces[m.face_set("a_front").unwrap()[0]];
        let (a, b, c) = (m.nodes[f[0]], m.nodes[f[1]], m.nodes[f[2]]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let nx = u[1] * v[2] - u[2] * v[1];
        assert!(nx > 0.0);
        assert_eq!(m.nodes_of("a").unwrap().len(), 45);
    }
}

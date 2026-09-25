//! Minimal Abaqus `.inp` reader (as exported by Gmsh): `*NODE`,
//! `*ELEMENT, type=C3D8` (hexahedra; one part per ELSET), `*ELEMENT,
//! type=CPS4` (quads → contact faces; one face set per ELSET), `*NSET`.

use crate::mesh::{Mesh, Part};
use std::collections::HashMap;

pub fn read_inp(path: &str) -> Result<Mesh, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    let mut mesh = Mesh::new();
    let mut node_index: HashMap<i64, usize> = HashMap::new();
    let mut section = String::new();
    let mut params: HashMap<String, String> = HashMap::new();
    let mut set_name = String::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("**") {
            continue;
        }
        if let Some(rest) = line.strip_prefix('*') {
            let mut parts = rest.split(',').map(|s| s.trim());
            section = parts.next().unwrap_or("").to_uppercase();
            params.clear();
            for p in parts {
                if let Some((k, v)) = p.split_once('=') {
                    params.insert(k.trim().to_uppercase(), v.trim().to_string());
                }
            }
            set_name = params.get("ELSET").or(params.get("NSET")).cloned().unwrap_or_default();
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        match section.as_str() {
            "NODE" => {
                if fields.len() < 4 {
                    continue;
                }
                let id: i64 = fields[0].parse().map_err(|_| format!("bad node line: {}", line))?;
                let x: [f64; 3] = std::array::from_fn(|k| fields[k + 1].parse().unwrap_or(0.0));
                node_index.insert(id, mesh.nodes.len());
                mesh.nodes.push(x);
            }
            "ELEMENT" => {
                let ty = params.get("TYPE").map(|s| s.to_uppercase()).unwrap_or_default();
                let ids: Vec<usize> = fields[1..].iter().map(|f| f.parse::<i64>().ok().and_then(|id| node_index.get(&id).copied())).collect::<Option<Vec<_>>>().ok_or_else(|| format!("unknown node in element line: {}", line))?;
                if ty == "C3D8" && ids.len() == 8 {
                    let part = match mesh.part_index(&set_name) {
                        Some(p) => p,
                        None => {
                            mesh.parts.push(Part { name: set_name.clone() });
                            mesh.parts.len() - 1
                        }
                    };
                    mesh.hexes.push(std::array::from_fn(|k| ids[k]));
                    mesh.hex_part.push(part);
                } else if ty == "CPS4" && ids.len() == 4 {
                    mesh.faces.push(std::array::from_fn(|k| ids[k]));
                    mesh.face_sets.entry(set_name.clone()).or_default().push(mesh.faces.len() - 1);
                }
            }
            "NSET" => {
                let set = mesh.node_sets.entry(set_name.clone()).or_default();
                for f in &fields {
                    if let Some(i) = f.parse::<i64>().ok().and_then(|id| node_index.get(&id).copied()) {
                        set.push(i);
                    }
                }
            }
            _ => {}
        }
    }
    for p in 0..mesh.parts.len() {
        let name = mesh.parts[p].name.clone();
        let nodes = mesh.part_nodes(p);
        mesh.node_sets.entry(name).or_insert(nodes);
    }
    Ok(mesh)
}

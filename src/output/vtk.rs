//! VTK output: binary-appended XML `.vtu` per frame plus a `.pvd` time
//! series, readable by ParaView, VisIt, PyVista, meshio.
//!
//! Point data: `displacement`, `velocity` (final frame only has velocity
//! from the results). Cell data: `plastic_strain`, `stress` (6 components,
//! Voigt xx yy zz xy yz xz), `part`, `eroded`.

use crate::mesh::Mesh;
use crate::solver::Frame;
use std::io::Write;
use std::path::Path;

fn push_f32(buf: &mut Vec<u8>, v: &[f32]) {
    buf.extend_from_slice(&(std::mem::size_of_val(v) as u64).to_le_bytes());
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

fn push_i32(buf: &mut Vec<u8>, v: &[i32]) {
    buf.extend_from_slice(&(std::mem::size_of_val(v) as u64).to_le_bytes());
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

fn push_i64(buf: &mut Vec<u8>, v: &[i64]) {
    buf.extend_from_slice(&(std::mem::size_of_val(v) as u64).to_le_bytes());
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

/// Write one frame as a `.vtu` (unstructured grid, hexahedra) file.
pub fn write_vtu(mesh: &Mesh, frame: &Frame, path: &Path) -> std::io::Result<()> {
    let n_pts = mesh.nodes.len();
    let n_cells = mesh.hexes.len();
    let mut data = Vec::new();
    let mut offsets = Vec::new();
    let mut xml = String::new();

    // Deformed points.
    let mut pts = Vec::with_capacity(3 * n_pts);
    for (i, p) in mesh.nodes.iter().enumerate() {
        for d in 0..3 {
            pts.push(p[d] as f32 + frame.displacement[3 * i + d]);
        }
    }
    offsets.push(data.len());
    push_f32(&mut data, &pts);
    // Point data.
    offsets.push(data.len());
    push_f32(&mut data, &frame.displacement);
    // Cells.
    let conn: Vec<i64> = mesh.hexes.iter().flat_map(|h| h.iter().map(|n| *n as i64)).collect();
    let cell_offsets: Vec<i64> = (1..=n_cells as i64).map(|k| 8 * k).collect();
    let types: Vec<u8> = vec![12; n_cells];
    offsets.push(data.len());
    push_i64(&mut data, &conn);
    offsets.push(data.len());
    push_i64(&mut data, &cell_offsets);
    offsets.push(data.len());
    data.extend_from_slice(&(types.len() as u64).to_le_bytes());
    data.extend_from_slice(&types);
    // Cell data.
    offsets.push(data.len());
    push_f32(&mut data, &frame.plastic_strain);
    let stress: Vec<f32> = frame.stress.iter().flat_map(|s| s.iter().copied()).collect();
    offsets.push(data.len());
    push_f32(&mut data, &stress);
    let part: Vec<i32> = mesh.hex_part.iter().map(|p| *p as i32).collect();
    offsets.push(data.len());
    push_i32(&mut data, &part);
    let eroded: Vec<i32> = frame.eroded.iter().map(|e| *e as i32).collect();
    offsets.push(data.len());
    push_i32(&mut data, &eroded);

    xml.push_str("<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
    xml.push_str(&format!("  <UnstructuredGrid>\n    <Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">\n", n_pts, n_cells));
    xml.push_str(&format!("      <Points>\n        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" offset=\"{}\"/>\n      </Points>\n", offsets[0]));
    xml.push_str(&format!("      <PointData Vectors=\"displacement\">\n        <DataArray type=\"Float32\" Name=\"displacement\" NumberOfComponents=\"3\" format=\"appended\" offset=\"{}\"/>\n      </PointData>\n", offsets[1]));
    xml.push_str(&format!(
        "      <Cells>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" offset=\"{}\"/>\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" offset=\"{}\"/>\n        <DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" offset=\"{}\"/>\n      </Cells>\n",
        offsets[2], offsets[3], offsets[4]
    ));
    xml.push_str(&format!(
        "      <CellData Scalars=\"plastic_strain\">\n        <DataArray type=\"Float32\" Name=\"plastic_strain\" format=\"appended\" offset=\"{}\"/>\n        <DataArray type=\"Float32\" Name=\"stress\" NumberOfComponents=\"6\" format=\"appended\" offset=\"{}\"/>\n        <DataArray type=\"Int32\" Name=\"part\" format=\"appended\" offset=\"{}\"/>\n        <DataArray type=\"Int32\" Name=\"eroded\" format=\"appended\" offset=\"{}\"/>\n      </CellData>\n",
        offsets[5], offsets[6], offsets[7], offsets[8]
    ));
    xml.push_str("    </Piece>\n  </UnstructuredGrid>\n  <AppendedData encoding=\"raw\">\n   _");
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(xml.as_bytes())?;
    f.write_all(&data)?;
    f.write_all(b"\n  </AppendedData>\n</VTKFile>\n")?;
    Ok(())
}

/// Write all frames as `<base>_NNNN.vtu` plus `<base>.pvd`.
pub fn write_series(mesh: &Mesh, frames: &[Frame], base: &Path) -> std::io::Result<()> {
    if let Some(dir) = base.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    let stem = base.file_name().and_then(|s| s.to_str()).unwrap_or("frame").to_string();
    let dir = base.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let mut pvd = String::from("<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\">\n  <Collection>\n");
    for (k, frame) in frames.iter().enumerate() {
        let name = format!("{}_{:04}.vtu", stem, k);
        write_vtu(mesh, frame, &dir.join(&name))?;
        pvd.push_str(&format!("    <DataSet timestep=\"{}\" file=\"{}\"/>\n", frame.time, name));
    }
    pvd.push_str("  </Collection>\n</VTKFile>\n");
    std::fs::write(dir.join(format!("{}.pvd", stem)), pvd)
}

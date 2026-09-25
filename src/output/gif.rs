//! Animated GIF of solver frames: a small software rasteriser. The outer
//! surface of the hex mesh is extracted once, faces are z-buffered with an
//! orthographic camera, shaded by orientation and coloured by plastic
//! strain (grey = elastic, yellow → red). Pure Rust.

use crate::mesh::Mesh;
use crate::solver::Frame;
use nalgebra::Vector3;
use std::collections::HashMap;

/// Camera and canvas settings.
#[derive(Clone, Debug)]
pub struct GifOptions {
    pub width: u16,
    pub height: u16,
    /// Frame delay in hundredths of a second.
    pub delay_cs: u16,
    /// Viewing direction (from the scene towards the camera), normalised internally.
    pub view_dir: Vector3<f64>,
    /// Up vector for the camera.
    pub up: Vector3<f64>,
    /// Plastic strain that maps to full red.
    pub plastic_strain_scale: f32,
    /// Show the simulated time in the corner.
    pub show_time: bool,
}

impl Default for GifOptions {
    fn default() -> Self {
        GifOptions {
            width: 640,
            height: 360,
            delay_cs: 4,
            view_dir: Vector3::new(0.45, -1.0, 0.55),
            up: Vector3::new(0.0, 0.0, 1.0),
            plastic_strain_scale: 0.5,
            show_time: true,
        }
    }
}

/// Outer surface of the hexahedral mesh: (element index, 4 node ids) per face.
fn surface_faces(mesh: &Mesh) -> Vec<(usize, [usize; 4])> {
    const FACES: [[usize; 4]; 6] = [
        [0, 3, 2, 1], // -z
        [4, 5, 6, 7], // +z
        [0, 1, 5, 4], // -y
        [1, 2, 6, 5], // +x
        [2, 3, 7, 6], // +y
        [3, 0, 4, 7], // -x
    ];
    let mut count: HashMap<[usize; 4], (usize, [usize; 4], usize)> = HashMap::new();
    for (id, conn) in mesh.hexes.iter().enumerate() {
        let id = &id;
        for f in FACES.iter() {
            let face = [conn[f[0]], conn[f[1]], conn[f[2]], conn[f[3]]];
            let mut key = face;
            key.sort_unstable();
            let e = count.entry(key).or_insert((*id, face, 0));
            e.2 += 1;
        }
    }
    count.into_values().filter(|(_, _, n)| *n == 1).map(|(id, face, _)| (id, face)).collect()
}

struct Canvas {
    w: usize,
    h: usize,
    rgb: Vec<[u8; 3]>,
    depth: Vec<f32>,
}

impl Canvas {
    fn new(w: usize, h: usize) -> Self {
        Canvas { w, h, rgb: vec![[245, 245, 245]; w * h], depth: vec![f32::NEG_INFINITY; w * h] }
    }

    /// Rasterise a triangle with per-triangle colour; `p` are (x, y, depth).
    fn triangle(&mut self, p: [[f32; 3]; 3], color: [u8; 3]) {
        let min_x = p.iter().map(|q| q[0]).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
        let max_x = p.iter().map(|q| q[0]).fold(f32::MIN, f32::max).ceil().min(self.w as f32 - 1.0) as usize;
        let min_y = p.iter().map(|q| q[1]).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
        let max_y = p.iter().map(|q| q[1]).fold(f32::MIN, f32::max).ceil().min(self.h as f32 - 1.0) as usize;
        if min_x > max_x || min_y > max_y {
            return;
        }
        let area = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
        if area.abs() < 1e-12 {
            return;
        }
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let w0 = ((p[1][0] - px) * (p[2][1] - py) - (p[2][0] - px) * (p[1][1] - py)) / area;
                let w1 = ((p[2][0] - px) * (p[0][1] - py) - (p[0][0] - px) * (p[2][1] - py)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let z = w0 * p[0][2] + w1 * p[1][2] + w2 * p[2][2];
                let idx = y * self.w + x;
                if z > self.depth[idx] {
                    self.depth[idx] = z;
                    self.rgb[idx] = color;
                }
            }
        }
    }

    fn line(&mut self, a: [f32; 3], b: [f32; 3], color: [u8; 3]) {
        let n = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()).ceil() as usize).max(1);
        for k in 0..=n {
            let t = k as f32 / n as f32;
            let x = a[0] + t * (b[0] - a[0]);
            let y = a[1] + t * (b[1] - a[1]);
            let z = a[2] + t * (b[2] - a[2]) + 1e-3;
            if x < 0.0 || y < 0.0 || x >= self.w as f32 || y >= self.h as f32 {
                continue;
            }
            let idx = y as usize * self.w + x as usize;
            if z >= self.depth[idx] - 2e-3 {
                self.rgb[idx] = color;
            }
        }
    }
}

/// 3×5 pixel digits and a few symbols for the time stamp.
fn glyph(c: char) -> [u8; 5] {
    match c {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        'm' => [0b000, 0b000, 0b111, 0b111, 0b101],
        's' => [0b000, 0b000, 0b011, 0b010, 0b110],
        _ => [0; 5],
    }
}

fn draw_text(canvas: &mut Canvas, text: &str, x0: usize, y0: usize, scale: usize) {
    for (ci, c) in text.chars().enumerate() {
        let g = glyph(c);
        for (row, bits) in g.iter().enumerate() {
            for col in 0..3 {
                if bits & (0b100 >> col) != 0 {
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let x = x0 + (ci * 4 + col) * scale + dx;
                            let y = y0 + row * scale + dy;
                            if x < canvas.w && y < canvas.h {
                                canvas.rgb[y * canvas.w + x] = [30, 30, 30];
                            }
                        }
                    }
                }
            }
        }
    }
}

fn strain_color(p: f32, scale: f32) -> [u8; 3] {
    if p <= 0.02 * scale {
        return [176, 190, 205]; // elastic: blue-grey
    }
    let t = (p / scale).clamp(0.0, 1.0);
    // yellow (255, 220, 60) → red (200, 30, 30)
    [
        (255.0 - 55.0 * t) as u8,
        (220.0 - 190.0 * t) as u8,
        (60.0 - 30.0 * t) as u8,
    ]
}

/// Render `frames` of `mesh` into an animated GIF at `path`.
///
/// The camera framing covers every frame's bounding box, so bodies move
/// across the canvas as they would in a high-speed film.
pub fn write_gif(mesh: &Mesh, frames: &[Frame], path: &str, opts: &GifOptions) -> std::io::Result<()> {
    if frames.is_empty() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "no frames captured; set frame_steps"));
    }
    let faces = surface_faces(mesh);
    let (w, h) = (opts.width as usize, opts.height as usize);

    // Camera basis.
    let view = opts.view_dir.normalize();
    let right = opts.up.cross(&view).normalize();
    let up = view.cross(&right).normalize();
    let project = |p: Vector3<f64>| -> (f64, f64, f64) { (p.dot(&right), p.dot(&up), p.dot(&view)) };

    // Framing from the union of all frames' bounding boxes.
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for frame in frames {
        for (i, node) in mesh.nodes.iter().enumerate() {
            let p = Vector3::from(*node)
                + Vector3::new(frame.displacement[3 * i] as f64, frame.displacement[3 * i + 1] as f64, frame.displacement[3 * i + 2] as f64);
            let (x, y, _) = project(p);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }
    let margin = 0.06;
    let span_x = (max_x - min_x).max(1e-9) * (1.0 + 2.0 * margin);
    let span_y = (max_y - min_y).max(1e-9) * (1.0 + 2.0 * margin);
    let scale = (w as f64 / span_x).min(h as f64 / span_y);
    let cx = 0.5 * (min_x + max_x);
    let cy = 0.5 * (min_y + max_y);
    let to_px = |(x, y, z): (f64, f64, f64)| -> [f32; 3] {
        [
            (w as f64 / 2.0 + (x - cx) * scale) as f32,
            (h as f64 / 2.0 - (y - cy) * scale) as f32,
            z as f32,
        ]
    };

    let light = Vector3::new(-0.3, -0.5, 0.8).normalize();
    let file = std::fs::File::create(path)?;
    let io_err = |e: gif::EncodingError| std::io::Error::new(std::io::ErrorKind::Other, e.to_string());
    let mut encoder = gif::Encoder::new(std::io::BufWriter::new(file), opts.width, opts.height, &[]).map_err(io_err)?;
    encoder.set_repeat(gif::Repeat::Infinite).map_err(io_err)?;

    for frame in frames {
        let mut canvas = Canvas::new(w, h);
        let pos: Vec<Vector3<f64>> = mesh
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| Vector3::from(*n) + Vector3::new(frame.displacement[3 * i] as f64, frame.displacement[3 * i + 1] as f64, frame.displacement[3 * i + 2] as f64))
            .collect();
        for (el, face) in &faces {
            let p: Vec<Vector3<f64>> = face.iter().map(|n| pos[*n]).collect();
            let normal = (p[2] - p[0]).cross(&(p[3] - p[1]));
            let nn = normal.norm();
            if nn < 1e-12 {
                continue;
            }
            let normal = normal / nn;
            if normal.dot(&view) <= 0.0 {
                continue; // back face
            }
            let shade = 0.55 + 0.45 * normal.dot(&light).max(0.0);
            if frame.eroded.get(*el).copied().unwrap_or(false) {
                continue;
            }
            let base = strain_color(frame.plastic_strain.get(*el).copied().unwrap_or(0.0), opts.plastic_strain_scale);
            let color = [(base[0] as f64 * shade) as u8, (base[1] as f64 * shade) as u8, (base[2] as f64 * shade) as u8];
            let q: Vec<[f32; 3]> = p.iter().map(|v| to_px(project(*v))).collect();
            canvas.triangle([q[0], q[1], q[2]], color);
            canvas.triangle([q[0], q[2], q[3]], color);
            let edge = [(color[0] as f32 * 0.6) as u8, (color[1] as f32 * 0.6) as u8, (color[2] as f32 * 0.6) as u8];
            for k in 0..4 {
                canvas.line(q[k], q[(k + 1) % 4], edge);
            }
        }
        if opts.show_time {
            draw_text(&mut canvas, &format!("{:.1}ms", frame.time * 1e3), 8, 8, 3);
        }
        let mut pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
        for px in &canvas.rgb {
            pixels.extend_from_slice(px);
        }
        let mut gif_frame = gif::Frame::from_rgb_speed(opts.width, opts.height, &pixels, 10);
        gif_frame.delay = opts.delay_cs;
        encoder.write_frame(&gif_frame).map_err(io_err)?;
    }
    Ok(())
}

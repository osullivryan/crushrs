//! crushrs: a fast explicit crash solver for vehicle delta-v estimation.
//!
//! Vehicles are homogenised crushable blocks of one-point hexahedra with a
//! corotational honeycomb material, calibrated so a simulated NHTSA rigid-
//! barrier test reproduces the published KW400 stiffness and dynamic crush.
//! The element kernel runs 8 elements per AVX2 lane in f32.

pub mod contact;
pub mod element;
pub mod input;
pub mod kernel;
pub mod kernel_simd;
pub mod material;
pub mod mesh;
pub mod model;
pub mod output;
pub mod solver;
pub mod vehicle;

pub use material::{Material, PlasticModel, Plasticity};
pub use mesh::{BlockFace, Mesh};
pub use model::{Accelerometer, Contact, LocalFrame, Model, Settings};
pub use solver::{run, Results};

pub const MPH: f64 = 0.44704;

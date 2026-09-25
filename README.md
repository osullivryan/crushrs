# crushrs

A fast explicit crash solver for estimating **delta-v** (the velocity change
each vehicle experiences) in vehicle-to-vehicle impacts, given initial
velocities and vehicle types.

Vehicles are homogenised crushable blocks of one-point hexahedral elements
with a corotational **honeycomb** material (normal stresses capped at the crush
strength in the element's own axes, hardening on compaction). Each block is
calibrated so a simulated 35 mph NCAP rigid-barrier test reproduces the
vehicle's published NHTSA **KW400** stiffness and implied dynamic crush. The
element kernel runs eight elements per AVX2 lane in f32; a 1290-element
head-on crash takes about 0.2 s.

```bash
cargo run --release -- crash 30 30 --gif crash.gif          # Silverado vs Neon head-on
cargo run --release -- tbone 30 0 -1.2 --vtk out/tbone      # Silverado into the Neon's side
cargo run --release -- barrier neon                         # NCAP barrier test vs NHTSA targets
cargo run --release -- run examples/crash_impact.toml       # any TOML setup
```

```
2007 Silverado 2622 kg @ +30 mph  vs  1996 Neon 1354 kg @ -30 mph
1290 elements, 644 steps, 187ms (forces 110ms, contact 7ms)
               delta-v              crush   (perfectly plastic limit)
Silverado    -9.71 m/s  -21.7 mph    518 mm   (-9.13 m/s)
Neon        +18.81 m/s  +42.1 mph   1093 mm   (+17.69 m/s)
```

## What it models

- **Elements:** 8-node hexahedra, one-point (mean-strain) integration with an
  exact rank-12 hourglass stabilisation that is zero on any uniform strain or
  rigid rotation (`element.rs`). Inverted elements are eroded.
- **Materials** (`material.rs`): honeycomb (rate form, no eigen-solve — the
  SIMD kernel model); isotropic crushable foam and J2 metal plasticity
  (finite-strain Hencky, reference models, scalar kernel); linear elastic.
- **Contact:** node-to-face penalty on quad face sets, two-way pairs.
- **Explicit integration:** central difference, per-element stability
  limit `f(ν)·L/c_d` re-evaluated as elements crush; f32 lane kernel
  (`kernel_simd.rs`) with runtime AVX2 dispatch, scalar f64 reference kernel
  (`kernel.rs`). Do **not** build with `-C target-cpu=native`: the AVX-512
  code LLVM emits for the lane kernel is several times slower.
- **Vehicles** (`vehicle.rs`): NCAP KW400 + CRASH3 A/B class coefficients set
  a bilinear force–crush curve; `calibrate` tunes the material so the
  *simulated* barrier test matches. Pre-tuned: 1996 Dodge Neon (front and
  side), 2007 Chevrolet Silverado (front). Element size 0.3 m; re-run
  `barrier <vehicle>-raw --calibrate 14` for another size.

| Vehicle | Test mass | NHTSA KW400 | Simulated | Crush sim / target |
|---|---|---|---|---|
| 1996 Dodge Neon | 1354 kg | 1251 N/mm | 1276 N/mm | 540 / 527 mm |
| 2007 Chevrolet Silverado | 2622 kg | 2550 N/mm | 2561 N/mm | 542 / 513 mm |

Side impacts use `Vehicle::side_profile()` (the block turned 90° with a side
crush curve from FMVSS 214 barrier-test coefficients).

## Inputs and outputs

- **TOML** setup (`input/config.rs` has the full schema): generated blocks or
  an Abaqus `.inp` mesh (Gmsh export: C3D8 hexahedra, CPS4 contact faces,
  NSET), materials per part, initial velocities, fixed sets, contact pairs,
  solver settings.
- **VTK:** binary `.vtu` per frame + `.pvd` time series (ParaView, VisIt,
  PyVista, meshio). Point data: displacement. Cell data: plastic strain /
  compaction, stress, part, eroded.
- **GIF:** built-in software renderer coloured by plastic strain.

## Caveats

- Homogenised blocks reproduce the *energy and stiffness* of a real front
  (hence delta-v), not the crush *shape*; peak forces are ~2× real because
  crush localises harder than real folding.
- The f32 kernel's precision floor is the stress increment `E·Δε` resolved to
  f32 epsilon (≈ E·6e-8 per step): fine for these materials (the caps bound
  it), but a material with a much smaller yield/E ratio would want f64.
- Calibrated parameters are tied to the element size they were tuned at.

## Sources

- NHTSA DOT HS 811 293 (Neon / Silverado KW400), NHTSA ESV 09-0416
- Vehicle frontal crush stiffness coefficient trends (CRASH3 A/B by class)
- Crush Energy and Stiffness in Side Impacts (FMVSS 214 side coefficients)

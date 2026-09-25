//! Crash-signal processing: SAE J211 channel-frequency-class filtering,
//! resampling and integration, and a `Pulse` (acceleration time history)
//! that can come from an NHTSA test file or from a simulated accelerometer.

use std::path::Path;

/// SAE J211-1 phaseless digital filter: a 2-pole Butterworth run forward
/// and backward. CFC 60 → −3 dB at ~100 Hz (× 1.667), CFC 180 → 300 Hz.
pub fn cfc_filter(x: &[f64], dt: f64, cfc: f64) -> Vec<f64> {
    if x.len() < 4 {
        return x.to_vec();
    }
    let wd = 2.0 * std::f64::consts::PI * cfc * 2.0775;
    let wa = (wd * dt / 2.0).sin() / (wd * dt / 2.0).cos();
    let s2 = 2.0_f64.sqrt();
    let den = 1.0 + s2 * wa + wa * wa;
    let a0 = wa * wa / den;
    let a1 = 2.0 * a0;
    let a2 = a0;
    let b1 = -2.0 * (wa * wa - 1.0) / den;
    let b2 = (-1.0 + s2 * wa - wa * wa) / den;
    let pass = |x: &[f64]| -> Vec<f64> {
        // Pad by reflection about the ends to limit start-up transients.
        let n = x.len();
        let pad = (3.0 / (cfc * dt)).ceil() as usize;
        let pad = pad.min(n - 1).max(1);
        let ext: Vec<f64> = (0..pad).map(|i| 2.0 * x[0] - x[pad - i]).chain(x.iter().copied()).chain((1..=pad).map(|i| 2.0 * x[n - 1] - x[n - 1 - i])).collect();
        let mut y = vec![0.0; ext.len()];
        y[0] = ext[0];
        y[1] = ext[1];
        for i in 2..ext.len() {
            y[i] = a0 * ext[i] + a1 * ext[i - 1] + a2 * ext[i - 2] + b1 * y[i - 1] + b2 * y[i - 2];
        }
        y[pad..pad + n].to_vec()
    };
    let fwd = pass(x);
    let mut rev: Vec<f64> = fwd.into_iter().rev().collect();
    rev = pass(&rev);
    rev.into_iter().rev().collect()
}

/// Linear resampling of `(t, y)` (t ascending, not necessarily uniform)
/// onto a uniform grid `t0 + k·dt` up to `t_end`.
pub fn resample(t: &[f64], y: &[f64], t0: f64, dt: f64, t_end: f64) -> (Vec<f64>, Vec<f64>) {
    let n = ((t_end - t0) / dt).floor() as usize + 1;
    let mut tu = Vec::with_capacity(n);
    let mut yu = Vec::with_capacity(n);
    let mut j = 0;
    for k in 0..n {
        let tk = t0 + k as f64 * dt;
        while j + 1 < t.len() && t[j + 1] < tk {
            j += 1;
        }
        let v = if j + 1 >= t.len() {
            y[t.len() - 1]
        } else if tk <= t[0] {
            y[0]
        } else {
            let w = (tk - t[j]) / (t[j + 1] - t[j]);
            y[j] + w * (y[j + 1] - y[j])
        };
        tu.push(tk);
        yu.push(v);
    }
    (tu, yu)
}

/// Cumulative trapezoidal integral with `y0` as the initial value.
pub fn integrate(t: &[f64], y: &[f64], y0: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(y.len());
    let mut acc = y0;
    out.push(acc);
    for i in 1..y.len() {
        acc += 0.5 * (y[i] + y[i - 1]) * (t[i] - t[i - 1]);
        out.push(acc);
    }
    out
}

/// A uniformly sampled acceleration history (m/s²), t = 0 at first contact.
#[derive(Debug, Clone)]
pub struct Pulse {
    pub time: Vec<f64>,
    pub accel: Vec<f64>,
}

impl Pulse {
    pub fn dt(&self) -> f64 {
        if self.time.len() > 1 {
            (self.time[self.time.len() - 1] - self.time[0]) / (self.time.len() - 1) as f64
        } else {
            0.0
        }
    }

    /// Read an NHTSA vehicle-database ASCII signal file (`v0NNNNtsv.CCC`):
    /// `time<TAB>value` rows, time in s, value in g when `in_g`. Only
    /// t ≥ 0 is kept; samples before contact are dropped.
    pub fn read_nhtsa_tsv(path: &Path, in_g: bool) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        Self::parse_tsv(&text, in_g)
    }

    pub fn parse_tsv(text: &str, in_g: bool) -> Result<Self, String> {
        let mut time = Vec::new();
        let mut accel = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split(|c: char| c == '\t' || c == ',' || c.is_whitespace()).filter(|s| !s.is_empty());
            let t: f64 = it.next().and_then(|s| s.parse().ok()).ok_or_else(|| format!("line {}: bad time '{}'", i + 1, line))?;
            let a: f64 = it.next().and_then(|s| s.parse().ok()).ok_or_else(|| format!("line {}: bad value '{}'", i + 1, line))?;
            if t < -1e-12 {
                continue;
            }
            time.push(t);
            accel.push(if in_g { a * 9.81 } else { a });
        }
        if time.len() < 4 {
            return Err("signal has fewer than 4 samples at t >= 0".into());
        }
        Ok(Pulse { time, accel })
    }

    /// Pulse from a simulated accelerometer's local-x acceleration,
    /// resampled to `dt`.
    pub fn from_history(model: &crate::model::Model, results: &crate::solver::Results, accelerometer: &str, dt: f64) -> Result<Self, String> {
        let k = model.accelerometers.iter().position(|a| a.name == accelerometer).ok_or_else(|| format!("no accelerometer '{}'", accelerometer))?;
        let h = &results.node_history;
        let node = model.accelerometers[k].nodes.first().copied().ok_or("accelerometer has no nodes")?;
        let mut t = Vec::new();
        let mut a = Vec::new();
        for i in 0..h.time.len() {
            if h.accelerometer[i] == k && h.node[i] == node {
                t.push(h.time[i]);
                a.push(h.local_acceleration[i][0]);
            }
        }
        if t.len() < 4 {
            return Err("accelerometer history is empty (set history_steps)".into());
        }
        let (time, accel) = resample(&t, &a, 0.0, dt, t[t.len() - 1]);
        Ok(Pulse { time, accel })
    }

    /// Copy resampled onto a uniform grid with step `dt` (linear).
    pub fn resampled(&self, dt: f64) -> Pulse {
        let (time, accel) = resample(&self.time, &self.accel, self.time[0], dt, *self.time.last().unwrap());
        Pulse { time, accel }
    }

    /// Copy with the sign convention "forward = positive": if the record's
    /// net velocity change is positive (the vehicle was travelling in −X
    /// of the test's global frame) the signal is negated, so a crash
    /// deceleration is always negative.
    pub fn forward(&self) -> Pulse {
        let dv: f64 = integrate(&self.time, &self.accel, 0.0).last().copied().unwrap_or(0.0);
        if dv > 0.0 {
            Pulse { time: self.time.clone(), accel: self.accel.iter().map(|a| -a).collect() }
        } else {
            self.clone()
        }
    }

    /// Point-wise mean of several pulses (e.g. left and right sill), on the
    /// first pulse's time grid.
    pub fn average(pulses: &[Pulse]) -> Pulse {
        let base = &pulses[0];
        let mut accel = vec![0.0; base.time.len()];
        for p in pulses {
            let (_, y) = resample(&p.time, &p.accel, base.time[0], base.dt(), *base.time.last().unwrap());
            for (a, b) in accel.iter_mut().zip(y) {
                *a += b / pulses.len() as f64;
            }
        }
        Pulse { time: base.time.clone(), accel }
    }

    /// Read one or more NHTSA TSV files (comma-separated list) and average
    /// them, in the forward-positive convention.
    pub fn read_nhtsa_list(list: &str, in_g: bool) -> Result<Self, String> {
        let pulses: Vec<Pulse> = list.split(',').map(|f| Pulse::read_nhtsa_tsv(Path::new(f.trim()), in_g).map(|p| p.forward())).collect::<Result<_, _>>()?;
        Ok(Pulse::average(&pulses))
    }

    /// CFC-filtered copy.
    pub fn filtered(&self, cfc: f64) -> Pulse {
        Pulse { time: self.time.clone(), accel: cfc_filter(&self.accel, self.dt(), cfc) }
    }

    /// Velocity from `v0` (m/s), positive along the accelerometer axis.
    pub fn velocity(&self, v0: f64) -> Vec<f64> {
        integrate(&self.time, &self.accel, v0)
    }

    /// Displacement from rest position (m).
    pub fn displacement(&self, v0: f64) -> Vec<f64> {
        integrate(&self.time, &self.velocity(v0), 0.0)
    }

    /// Truncate to `t_end`.
    pub fn truncated(&self, t_end: f64) -> Pulse {
        let n = self.time.iter().take_while(|t| **t <= t_end + 1e-12).count();
        Pulse { time: self.time[..n].to_vec(), accel: self.accel[..n].to_vec() }
    }

    /// Value at `t` (linear).
    pub fn at(&self, t: f64) -> f64 {
        let (_, y) = resample(&self.time, &self.accel, t, 1.0, t);
        y[0]
    }
}

/// Barrier-test summary of a pulse: the force–crush curve (`F = −m·a`
/// against the integrated displacement), maximum dynamic crush, rebound
/// velocity, restitution.
#[derive(Debug, Clone)]
pub struct PulseMetrics {
    /// Time of maximum crush (s).
    pub t_max_crush: f64,
    pub max_crush: f64,
    /// Velocity at the end of the record (negative = rebounding).
    pub v_end: f64,
    pub restitution: f64,
    /// Peak (most negative) filtered acceleration (m/s²).
    pub peak_accel: f64,
    /// (crush, force) samples over the loading phase.
    pub force_crush: Vec<[f64; 2]>,
}

impl PulseMetrics {
    pub fn new(p: &Pulse, mass: f64, v0: f64) -> Self {
        let v = p.velocity(v0);
        let x = p.displacement(v0);
        let (i_max, &max_crush) = x.iter().enumerate().fold((0, &0.0), |m, (i, xi)| if xi > m.1 { (i, xi) } else { m });
        let v_end = *v.last().unwrap();
        let v_min = v.iter().copied().fold(f64::INFINITY, f64::min);
        let force_crush: Vec<[f64; 2]> = (0..=i_max).map(|i| [x[i], -mass * p.accel[i]]).collect();
        PulseMetrics { t_max_crush: p.time[i_max], max_crush, v_end, restitution: (-v_min / v0).max(0.0), peak_accel: p.accel.iter().copied().fold(0.0, f64::min), force_crush }
    }

    /// Energy-equivalent stiffness over a crush window (N/m): NHTSA's
    /// KW400 for `(0.025, 0.4)`, `2·(E(x1) − E(x0)) / (x1² − x0²)` with
    /// `E` the energy absorbed up to a crush. `None` when the record does
    /// not reach `x1`.
    pub fn kw_window(&self, (x0, x1): (f64, f64)) -> Option<f64> {
        if self.max_crush < x1 {
            return None;
        }
        let energy_at = |d: f64| -> f64 {
            let mut e = 0.0;
            for w in self.force_crush.windows(2) {
                let ([c0, f0], [c1, f1]) = (w[0], w[1]);
                if c1 <= c0 {
                    continue;
                }
                if c1 <= d {
                    e += 0.5 * (f0 + f1) * (c1 - c0);
                } else if c0 < d {
                    let f_d = f0 + (f1 - f0) * (d - c0) / (c1 - c0);
                    e += 0.5 * (f0 + f_d) * (d - c0);
                    break;
                } else {
                    break;
                }
            }
            e
        };
        Some(2.0 * (energy_at(x1) - energy_at(x0)) / (x1 * x1 - x0 * x0))
    }

    /// Mean force over a crush window `[x0, x1]` of the loading phase.
    pub fn mean_force(&self, x0: f64, x1: f64) -> Option<f64> {
        let pts: Vec<&[f64; 2]> = self.force_crush.iter().filter(|p| p[0] >= x0 && p[0] <= x1).collect();
        if pts.len() < 2 {
            return None;
        }
        let mut e = 0.0;
        for w in pts.windows(2) {
            e += 0.5 * (w[0][1] + w[1][1]) * (w[1][0] - w[0][0]);
        }
        let span = pts.last().unwrap()[0] - pts[0][0];
        if span > 0.0 {
            Some(e / span)
        } else {
            None
        }
    }

    /// Energy-equivalent stiffness `2·E/(x1² − x0²)` over a crush window.
    pub fn kw(&self, x0: f64, x1: f64) -> Option<f64> {
        let pts: Vec<&[f64; 2]> = self.force_crush.iter().filter(|p| p[0] >= x0 && p[0] <= x1).collect();
        if pts.len() < 2 || pts.last().unwrap()[0] < x1 - 0.05 {
            return None;
        }
        let mut e = 0.0;
        for w in pts.windows(2) {
            e += 0.5 * (w[0][1] + w[1][1]) * (w[1][0] - w[0][0]);
        }
        Some(2.0 * e / (x1 * x1 - x0 * x0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfc60_passes_dc_and_kills_1khz() {
        let dt = 1e-4;
        let t: Vec<f64> = (0..5000).map(|i| i as f64 * dt).collect();
        let x: Vec<f64> = t.iter().map(|t| 10.0 + 5.0 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin()).collect();
        let y = cfc_filter(&x, dt, 60.0);
        let mid = &y[1000..4000];
        let max_dev = mid.iter().map(|v| (v - 10.0).abs()).fold(0.0, f64::max);
        assert!(max_dev < 0.05, "1 kHz ripple left: {}", max_dev);
        // −3 dB near 100 Hz.
        let x100: Vec<f64> = t.iter().map(|t| (2.0 * std::f64::consts::PI * 100.0 * t).sin()).collect();
        let y100 = cfc_filter(&x100, dt, 60.0);
        let amp = y100[1000..4000].iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!((amp - 0.707).abs() < 0.05, "100 Hz gain {}", amp);
    }

    #[test]
    fn parse_and_integrate_tsv() {
        let text = "-0.001\t-0.1\r\n0\t0\n0.001\t-1\n0.002\t-1\n0.003\t-1\n0.004\t-1\n";
        let p = Pulse::parse_tsv(text, true).unwrap();
        assert_eq!(p.time.len(), 5);
        assert!((p.dt() - 0.001).abs() < 1e-12);
        let v = p.velocity(10.0);
        assert!((v[4] - (10.0 - 9.81 * 0.0035)).abs() < 1e-9);
        let m = PulseMetrics::new(&p, 1000.0, 10.0);
        assert!(m.max_crush > 0.039 && m.max_crush < 0.04);
    }

    #[test]
    fn resample_is_linear() {
        let t = [0.0, 1.0, 3.0];
        let y = [0.0, 2.0, 6.0];
        let (tu, yu) = resample(&t, &y, 0.0, 0.5, 3.0);
        assert_eq!(tu.len(), 7);
        assert!((yu[1] - 1.0).abs() < 1e-12 && (yu[3] - 3.0).abs() < 1e-12 && (yu[6] - 6.0).abs() < 1e-12);
    }
}

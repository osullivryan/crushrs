#!/usr/bin/env python3
"""Plot the measured-vs-simulated rear-seat pulses of a `headon --pulse-a
--pulse-b --history <base>` run (its `<base>.pulse_a.parquet` and
`.pulse_b.parquet`), optionally overlaying several runs.

    python3 examples/plot_headon_pulses.py out/nn_fit "rail boxes + road" \
        out/nn_ground_nocore "uniform + road" out/nn_base "uniform, free" \
        --labels "Navigator 2873 kg" "Neon 1378 kg" -o docs/navigator_neon_validation.png
"""
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+", help="pairs of <history base> <legend label>")
p.add_argument("--labels", nargs=2, default=["vehicle A", "vehicle B"])
p.add_argument("--title", default="rear accelerometers, measured vs simulated")
p.add_argument("-o", "--out", default="pulses.png")
args = p.parse_args()
runs = list(zip(args.runs[::2], args.runs[1::2]))
colors = ["#d0343a", "#4c9be8", "#b0b0b0", "#3aa655", "#e8a13c"]

fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
for row, (v, label) in enumerate(zip("ab", args.labels)):
    dm = pd.read_parquet(f"{runs[0][0]}.pulse_{v}.parquet")
    t = dm.time * 1e3
    ax[row, 0].plot(t, dm.a_meas / 9.81, "k", lw=2, label="measured")
    ax[row, 1].plot(t, dm.v_meas, "k", lw=2)
    ax[row, 2].plot(t, dm.x_meas * 1e3, "k", lw=2)
    for (base, lab), c in zip(runs, colors):
        d = pd.read_parquet(f"{base}.pulse_{v}.parquet")
        ls = "--" if lab.endswith("free") else "-"
        ax[row, 0].plot(d.time * 1e3, d.a_sim / 9.81, c, ls=ls, lw=1.4, label=lab)
        ax[row, 1].plot(d.time * 1e3, d.v_sim, c, ls=ls, lw=1.4)
        ax[row, 2].plot(d.time * 1e3, d.x_sim * 1e3, c, ls=ls, lw=1.4)
    ax[row, 0].set_ylabel(f"{label}\nacceleration [g] (CFC60)")
    ax[row, 1].set_ylabel("velocity [m/s]")
    ax[row, 2].set_ylabel("displacement [mm]")
    for a in ax[row]:
        a.grid(alpha=0.3)
        a.set_xlim(0, t.iloc[-1])
for a in ax[1]:
    a.set_xlabel("time [ms]")
ax[0, 0].legend(fontsize=8, loc="lower right")
fig.suptitle(args.title)
fig.tight_layout()
fig.savefig(args.out, dpi=110)
print("wrote", args.out)

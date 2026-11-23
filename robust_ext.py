"""
Robustness beyond the assumptions of Corollary 4 (revision: R2-M4, R4-5).

The robust mask and the robust EMPC plan with the upper bounds p + dmax,
dmax = (0.25, 0.5, 1.0) h, while the plant violates the assumptions:

  (a) bound exceeded:   d_ik ~ U[0, kappa * dmax_i],  kappa in {1, 1.25, 1.5, 2}
  (b) systematic error: d_ik = kappa * dmax_i for every job (parameter
                        mismatch, e.g. a slower oven), kappa in {1, 1.25, 1.5}
  (c) measurement noise on the completion times used by the mask,
      x_meas = x + U[-e, e], e in {0.25, 0.5} h (random delays within the
      bounds), with and without a margin e added to x_meas.

When the mask becomes empty the environment releases immediately; by
Lemma 1 this minimizes the completion time of every remaining job
(Proposition "fallback"), and the resulting lateness is reported.

Output: results/robust_ext.csv, results/robust_ext.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import empc_measured
from data_loader import HourlyDataset
from evaluate import day_forecast, load_model
from factory_env import PROC_H, FactoryMicrogridEnv

RES = Path("results")
DMAX = np.array([0.25, 0.5, 1.0])
YEARS = (2020, 2023)
SIGMA = 0.15


def delays(year, i, kappa, mode):
    u = np.random.default_rng([year, i, 4242]).uniform(0, 1, (3, 3))
    if mode == "systematic":
        u = np.ones((3, 3))
    return u * DMAX * kappa


def roll(model, solar, price, fc, dl, D, mask_model="robust", noise=0.0, margin=0.0):
    env = FactoryMicrogridEnv(stochastic=False, feasibility="mask", deadline=D,
                              proc_delay_max=DMAX, mask_model=mask_model,
                              meas_noise_max=noise, meas_margin=margin,
                              seed=0)
    env.set_day(solar, price, fc, dl)
    obs, _ = env.reset()
    done = False
    while not done:
        a, _ = model.predict(obs, action_masks=env.action_mask(), deterministic=True)
        obs, _, done, _, info = env.step(int(a))
    return env.total_cost, info["ep_late"], info["ep_lateness_h"], info["ep_mask_empty"]


def summarise(df):
    out = []
    keys = ["D", "mode", "kappa", "noise", "margin", "method"]
    for k, g in df.groupby(keys):
        late = g[g.late > 0]
        out.append(dict(zip(keys, k), units=int(len(g)),
                        late_pct=100 * float((g.late > 0).mean()),
                        late_jobs_per_100days=100 * float(g.late.mean()),
                        mean_lateness_h_of_late_days=float(late.lateness_h.mean())
                        if len(late) and late.lateness_h.notna().any() else 0.0,
                        max_lateness_h=float(g.lateness_h.max()) if g.lateness_h.notna().any() else float("nan"),
                        empty_mask_pct=100 * float((g["empty"] > 0).mean()),
                        mean_cost=float(g.cost.mean())))
    (RES / "robust_ext.json").write_text(json.dumps(out, indent=2))
    for e in out:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()})
    return out


def main(seeds=(0, 1, 2, 3, 4)):
    models = {s: load_model("mask", s) for s in seeds}
    cases = []
    for D in (23.0, 19.0):
        for k in (1.0, 1.25, 1.5, 2.0):
            cases.append(dict(D=D, mode="random", kappa=k, noise=0.0, margin=0.0))
        for k in (1.0, 1.25, 1.5):
            cases.append(dict(D=D, mode="systematic", kappa=k, noise=0.0, margin=0.0))
        for e in (0.25, 0.5):
            for mg in (0.0, e):
                cases.append(dict(D=D, mode="random", kappa=1.0, noise=e, margin=mg))
    rows = []
    for c in cases:
        for year in YEARS:
            ds = HourlyDataset(f"data/merged_hourly_{year}.csv", verbose=False)
            for i, d in enumerate(ds.dates):
                solar, price = (np.asarray(a, float) for a in ds.get_day(d))
                fc = day_forecast(solar, SIGMA, year, i)
                dl = delays(year, i, c["kappa"], c["mode"])
                base = dict(c, year=year, date=str(d))
                if c["noise"] == 0:
                    cost, _, late, nop = empc_measured(fc, price, solar, dl,
                                                       np.array(PROC_H) + DMAX, c["D"])
                    rows.append(dict(base, method="EMPC (robust model)", seed=-1,
                                     cost=cost, late=late, lateness_h=np.nan,
                                     empty=nop))
                for s, m in models.items():
                    cost, late, lh, emp = roll(m, solar, price, fc, dl, c["D"],
                                               noise=c["noise"], margin=c["margin"])
                    rows.append(dict(base, method="Masked agent (robust mask)",
                                     seed=s, cost=cost, late=late, lateness_h=lh,
                                     empty=emp))
        print(c, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "robust_ext.csv", index=False)
    out = []
    keys = ["D", "mode", "kappa", "noise", "margin", "method"]
    for k, g in df.groupby(keys):
        late = g[g.late > 0]
        out.append(dict(zip(keys, k), units=int(len(g)),
                        late_pct=100 * float((g.late > 0).mean()),
                        late_jobs_per_100days=100 * float(g.late.mean()),
                        mean_lateness_h_of_late_days=float(late.lateness_h.mean())
                        if len(late) and late.lateness_h.notna().any() else 0.0,
                        max_lateness_h=float(g.lateness_h.max()) if g.lateness_h.notna().any() else np.nan,
                        empty_mask_pct=100 * float((g["empty"] > 0).mean()),
                        mean_cost=float(g.cost.mean())))
    (RES / "robust_ext.json").write_text(json.dumps(out, indent=2))
    for e in out:
        print(e)


if __name__ == "__main__":
    main()

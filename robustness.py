"""
Processing-delay experiment (Corollary 1 of the paper).

Every job suffers a random delay on every machine, d_ik ~ U[0, dmax_i],
dmax = (0.25, 0.5, 1.0) h, unknown when the job is released; completion
times are measured. The agents trained without delays are deployed
unchanged with
    - the nominal mask   (planning model p),
    - the robust mask    (planning model p + dmax, Corollary 1),
and compared with the shield / penalty agents and with receding-horizon EMPC
planning on the nominal or the robust model from the measured state.

Output: results/robustness.csv, results/robustness.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import empc_measured
from data_loader import HourlyDataset
from evaluate import VARIANT_LABEL, day_forecast, load_model
from factory_env import PROC_H, FactoryMicrogridEnv

RES = Path("results")
DMAX = np.array([0.25, 0.5, 1.0])
DEADLINES = (23.0, 19.0)
YEARS = (2020, 2023)
SIGMA = 0.15
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def day_delays(year, i):
    return np.random.default_rng([year, i, 4242]).uniform(0, 1, (3, 3)) * DMAX


def rollout(model, variant, mask_model, solar, price, fc, delays, D):
    env = FactoryMicrogridEnv(stochastic=False, feasibility=variant, deadline=D,
                              proc_delay_max=DMAX, mask_model=mask_model)
    env.set_day(solar, price, fc, delays)
    obs, _ = env.reset()
    done = False
    while not done:
        if variant == "mask":
            a, _ = model.predict(obs, action_masks=env.action_mask(),
                                 deterministic=True)
        else:
            a, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(a))
    return env.total_cost, info["ep_late"], info["ep_mask_empty"]


def main():
    models = {(v, s): load_model(v, s) for v in VARIANT_LABEL for s in SEEDS}
    rows = []
    for D in DEADLINES:
        for year in YEARS:
            ds = HourlyDataset(f"data/merged_hourly_{year}.csv", verbose=False)
            for i, d in enumerate(ds.dates):
                solar, price = (np.asarray(a, float) for a in ds.get_day(d))
                fc = day_forecast(solar, SIGMA, year, i)
                dl = day_delays(year, i)
                base = dict(deadline=D, year=year, date=str(d))
                for name, proc in (("Max-plus EMPC (nominal model)", np.array(PROC_H)),
                                   ("Max-plus EMPC (robust model)", np.array(PROC_H) + DMAX)):
                    c, _, late, nop = empc_measured(fc, price, solar, dl, proc, D)
                    rows.append(dict(base, method=name, seed=-1, cost=c, late=late,
                                     mask_empty=nop))
                for (v, s), m in models.items():
                    modes = ("nominal", "robust") if v == "mask" else ("nominal",)
                    for mm in modes:
                        c, late, emp = rollout(m, v, mm, solar, price, fc, dl, D)
                        name = VARIANT_LABEL[v] + ("" if v != "mask" else
                                                   f" [{mm} mask]")
                        rows.append(dict(base, method=name, seed=s, cost=c,
                                         late=late, mask_empty=emp))
            print(f"  D={D} {year} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "robustness.csv", index=False)
    out = {}
    for D, g in df.groupby("deadline"):
        ref = g[g.method == "Max-plus EMPC (robust model)"].cost.mean()
        tab = {}
        for m, h in g.groupby("method"):
            tab[m] = dict(mean_cost=float(h.cost.mean()),
                          rel_to_robust_empc_pct=float(100 * (h.cost.mean() / ref - 1)),
                          seed_days=int(len(h)),
                          late_seed_days=int((h.late > 0).sum()),
                          late_jobs=int(h.late.sum()),
                          late_pct=float(100 * (h.late > 0).mean()),
                          mask_empty_seed_days=int((h.mask_empty > 0).sum()))
        out[f"D{int(D)}"] = tab
    (RES / "robustness.json").write_text(json.dumps(out, indent=2))
    for k, tab in out.items():
        print(f"\n== {k}")
        for m, e in tab.items():
            print(f"  {m:55s} cost {e['mean_cost']:8.2f} ({e['rel_to_robust_empc_pct']:+5.1f}%)"
                  f"  late days {e['late_seed_days']:4d}/{e['seed_days']} ({e['late_pct']:.1f}%)"
                  f"  empty-mask days {e['mask_empty_seed_days']}")


if __name__ == "__main__":
    main()

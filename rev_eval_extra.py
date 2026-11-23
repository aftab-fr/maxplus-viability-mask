"""
Evaluation of the additional agents trained for the revision.

  tuning   : the four hyperparameter configurations (A-D, seed 90) of every
             variant on the training years (as in the paper) and on 2021, an
             independent validation year never used for training (R2-M8, R2-m5)
  ablation : penalty sweep (50, 150, 500, 2000 EUR) and model-free agents
             (no slack in the observation) on the test years, sigma = 0.15,
             with D = 23 h and D = 19 h (R2-M8, R2-m2, R3)

  python rev_eval_extra.py tuning
  python rev_eval_extra.py ablation
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from baselines import ScheduleSpace
from data_loader import HourlyDataset
from evaluate import day_forecast
from factory_env import FactoryMicrogridEnv

RES = Path("results")


def load(run):
    algo = MaskablePPO if run.startswith("mask") else PPO
    return algo.load(Path("runs") / run / "model", device="cpu")


def rollout(model, variant, solar, price, fc, D=23.0, observe_slack=True):
    env = FactoryMicrogridEnv(stochastic=False, feasibility=variant, deadline=D,
                              observe_slack=observe_slack)
    env.set_day(solar, price, fc)
    obs, _ = env.reset()
    done = False
    while not done:
        if variant == "mask":
            a, _ = model.predict(obs, action_masks=env.action_mask(), deterministic=True)
        else:
            a, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(a))
    return env.total_cost, info["ep_violations"], info["ep_safety_interventions"]


def run_set(runs, years, deadlines, sigma=0.15):
    models = {r: load(r) for r in runs}
    rows = []
    for D in deadlines:
        sp = ScheduleSpace(deadline=D)
        for y in years:
            ds = HourlyDataset(f"data/merged_hourly_{y}.csv", verbose=False)
            for i, d in enumerate(ds.dates):
                s, p = (np.asarray(a, float) for a in ds.get_day(d))
                fc = day_forecast(s, sigma, y, i)
                b = dict(D=D, year=y, date=str(d))
                rows.append(dict(b, run="oracle", cost=sp.oracle(s, p)[0], viol=0, safety=0))
                rows.append(dict(b, run="empc", cost=sp.empc(fc, p, s)[0], viol=0, safety=0))
                for r, m in models.items():
                    v = r.split("_")[0]
                    c, vi, sa = rollout(m, v, s, p, fc, D, "noslack" not in r)
                    rows.append(dict(b, run=r, cost=c, viol=vi, safety=sa))
            print(D, y, flush=True)
    return pd.DataFrame(rows)


def summarise(df, name):
    out = []
    for (D, y), g in df.groupby(["D", "year"]):
        o = g[g.run == "oracle"].cost.mean()
        for r, h in g.groupby("run"):
            out.append(dict(D=D, year=y, run=r, gap_pct=float(100 * (h.cost.mean() / o - 1)),
                            late_days=int((h.viol > 0).sum()),
                            safety_days=int((h.safety > 0).sum()), days=int(len(h))))
    (RES / f"{name}.json").write_text(json.dumps(out, indent=2, default=float))
    for e in out:
        print(e)


def tariff_retrain(seeds=(0, 1, 2, 3, 4)):
    """Masked agents retrained with the full tariff (scenario ALL), evaluated
    under that tariff; oracle from costmodel_rev.py (re-optimized for ALL)."""
    from train_ppo import TARIFFS
    cm = pd.read_csv(RES / "costmodel_days.csv")
    orc = cm[(cm.scenario == "ALL") & (cm.method == "Oracle")].set_index(["year", "date"]).cost
    rows = []
    for s in seeds:
        m = MaskablePPO.load(Path("runs") / f"mask_tariffALL_s{s}" / "model", device="cpu")
        for y in (2020, 2023):
            ds = HourlyDataset(f"data/merged_hourly_{y}.csv", verbose=False)
            for i, dd in enumerate(ds.dates):
                so, p = (np.asarray(a, float) for a in ds.get_day(dd))
                fc = day_forecast(so, 0.15, y, i)
                env = FactoryMicrogridEnv(stochastic=False, feasibility="mask",
                                          tariff=TARIFFS["ALL"])
                env.set_day(so, p, fc)
                obs, _ = env.reset()
                done = False
                while not done:
                    a, _ = m.predict(obs, action_masks=env.action_mask(), deterministic=True)
                    obs, _, done, _, info = env.step(int(a))
                rows.append(dict(seed=s, year=y, date=str(dd), cost=env.total_cost,
                                 late=info["ep_late"]))
        print("tariff seed", s, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "tariff_retrain.csv", index=False)
    out = {"ALL": {}}
    for y, g in df.groupby("year"):
        o = orc.loc[y].reindex(g.date.unique()).mean()
        out["ALL"][str(y)] = float(100 * (g.cost.mean() / o - 1))
    out["late"] = int(df.late.sum())
    (RES / "tariff_retrain.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "tariff":
        tariff_retrain()
        sys.exit()
    if what == "cfga":
        runs = [f"{v}_cfgA_s{s}" for v in ("penalty", "shield") for s in range(10)]
        runs = [r for r in runs if (Path("runs") / r / "model.zip").exists()]
        df = run_set(runs, (2020, 2023), (23.0, 19.0))
        df.to_csv(RES / "rev_cfga.csv", index=False)
        summarise(df, "rev_cfga")
        sys.exit()
    if what == "tuning":
        runs = [f"{v}_tune{c}_s90" for v in ("mask", "penalty", "shield") for c in "ABCD"]
        runs = [r for r in runs if (Path("runs") / r / "model.zip").exists()]
        df = run_set(runs, (2018, 2019, 2021), (23.0,))
        df.to_csv(RES / "rev_tuning.csv", index=False)
        summarise(df, "rev_tuning")
    else:
        runs = [f"penalty_p{p}_s{s}" for p in (50, 150, 2000) for s in range(5)] + \
               [f"penalty_s{s}" for s in range(5)] + \
               [f"{v}_noslack_s{s}" for v in ("penalty", "shield") for s in range(5)]
        runs = [r for r in runs if (Path("runs") / r / "model.zip").exists()]
        df = run_set(runs, (2020, 2023), (23.0, 19.0))
        df.to_csv(RES / "rev_ablation.csv", index=False)
        summarise(df, "rev_ablation")

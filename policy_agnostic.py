"""
The viability mask as a policy-agnostic safety layer (revision: R3-Intro,
R2-M1, R4-3).

Policies (none of them is trained with the mask):
  random       uniform choice between wait / release at every epoch
  threshold    release if the current price ratio is at or below the 30%
               quantile of the remaining day's prices (simple planner rule)
  stale plan   the EMPC plan computed for D = 23 h executed open loop
               (a plan that is not updated after a rush order)
  EMPC         receding-horizon max-plus EMPC for the actual deadline
  RL agent     masked PPO agents (seeds 0-4)

Each is run on the held-out years (sigma = 0.15) with D in {23, 19} h,
without and with the mask (actions outside the mask are replaced by the
admissible action; for `random` and `threshold` the choice is made among
admissible actions). Reported: share of days with a late or unreleased job,
mean cost, gap to the oracle of the same deadline.

Output: results/policy_agnostic.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import ScheduleSpace
from data_loader import HourlyDataset
from evaluate import day_forecast, load_model
from factory_env import FactoryMicrogridEnv

RES = Path("results")
SIGMA = 0.15


def run(policy, solar, price, fc, D, masked, rng, plan=None, model=None):
    env = FactoryMicrogridEnv(stochastic=False, feasibility="penalty", deadline=D)
    env.set_day(solar, price, fc)
    obs, _ = env.reset()
    done = False
    while not done:
        mk = env.action_mask()
        t = env.hour
        if policy == "random":
            a = int(rng.integers(0, 2))
        elif policy == "threshold":
            rem = price[t:]
            a = int(price[t] <= np.quantile(rem, 0.3))
        elif policy in ("stale plan", "EMPC"):
            a = int(t in plan)
        elif policy == "RL agent":
            a = int(model.predict(obs, action_masks=mk, deterministic=True)[0]) if masked \
                else int(model.predict(obs, action_masks=np.ones(2, bool), deterministic=True)[0])
        if masked and not mk[a]:
            a = 1 - a
        obs, _, done, _, info = env.step(a)
    late = info["ep_late"]
    return env.total_cost, int(late > 0)


def main():
    models = {s: load_model("mask", s) for s in range(5)}
    rows = []
    for D in (23.0, 19.0):
        sp, sp23 = ScheduleSpace(deadline=D), ScheduleSpace(deadline=23.0)
        for year in (2020, 2023):
            ds = HourlyDataset(f"data/merged_hourly_{year}.csv", verbose=False)
            for i, d in enumerate(ds.dates):
                solar, price = (np.asarray(a, float) for a in ds.get_day(d))
                fc = day_forecast(solar, SIGMA, year, i)
                orc = sp.oracle(solar, price)[0]
                stale = sp23.empc(fc, price, solar)[1]
                empc = sp.empc(fc, price, solar)[1]
                rng = np.random.default_rng([year, i, int(D)])
                b = dict(D=D, year=year, date=str(d), oracle=orc)
                for masked in (False, True):
                    for pol in ("random", "threshold", "stale plan", "EMPC"):
                        r = np.random.default_rng([year, i, int(D), 7])
                        c, late = run(pol, solar, price, fc, D, masked, r,
                                      plan=stale if pol == "stale plan" else empc)
                        rows.append(dict(b, policy=pol, masked=masked, seed=-1,
                                         cost=c, late=late))
                    for s, m in models.items():
                        c, late = run("RL agent", solar, price, fc, D, masked, None, model=m)
                        rows.append(dict(b, policy="RL agent", masked=masked, seed=s,
                                         cost=c, late=late))
            print("D", D, year, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "policy_agnostic.csv", index=False)
    out = []
    for (D, pol, mk), g in df.groupby(["D", "policy", "masked"]):
        out.append(dict(D=D, policy=pol, masked=bool(mk), units=int(len(g)),
                        late_pct=100 * float(g.late.mean()),
                        mean_cost=float(g.cost.mean()),
                        gap_pct=float(100 * (g.cost.mean() / g.oracle.mean() - 1)),
                        gap_ontime_days_pct=float(100 * (g[g.late == 0].cost.mean() /
                                                         g[g.late == 0].oracle.mean() - 1))))
    (RES / "policy_agnostic.json").write_text(json.dumps(out, indent=2))
    for e in out:
        print(e)


if __name__ == "__main__":
    main()

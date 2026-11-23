"""
Evaluate all schedulers on training and held-out years.

Every method sees exactly the same PV forecast for a given (year, day,
sigma) -- common random numbers -- so per-day costs are paired.

Output: results/daily_costs.csv  (one row per year x sigma x day x method)
        results/summary.json     (tables used in the paper)
        results/timing.json      (decision-time benchmark)

Usage:  python evaluate.py [--seeds 0 1 2 3 4] [--sigmas 0 0.15 0.3]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from baselines import FIXED_RULES, ScheduleSpace
from data_loader import HourlyDataset
from factory_env import FactoryMicrogridEnv, solar_forecast

RES = Path("results")
TRAIN_YEARS = (2018, 2019)
TEST_YEARS = (2020, 2023)
VARIANT_LABEL = {"mask": "MaskablePPO + max-plus mask (proposed)",
                 "shield": "PPO + post-hoc shield",
                 "penalty": "PPO + deadline penalty"}


def day_forecast(solar, sigma, year, i):
    return solar_forecast(solar, sigma, np.random.default_rng(
        [year, i, int(round(sigma * 1000))]))


def load_model(variant, seed):
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import PPO
    algo = MaskablePPO if variant == "mask" else PPO
    return algo.load(Path("runs") / f"{variant}_s{seed}" / "model", device="cpu")


def rollout(model, variant, solar, price, forecast, deadline=23.0):
    env = FactoryMicrogridEnv(stochastic=False, feasibility=variant,
                              deadline=deadline)
    env.set_day(solar, price, forecast)
    obs, info = env.reset()
    done = False
    while not done:
        if variant == "mask":
            a, _ = model.predict(obs, action_masks=env.action_mask(),
                                 deterministic=True)
        else:
            a, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(a))
    return (env.total_cost, tuple(env.schedule), info["ep_interventions"],
            info["ep_violations"], info["ep_safety_interventions"], env)


def evaluate(seeds, sigmas, years):
    space = ScheduleSpace()
    models = {(v, s): load_model(v, s) for v in VARIANT_LABEL for s in seeds
              if (Path("runs") / f"{v}_s{s}" / "model.zip").exists()}
    rows = []
    for year in years:
        ds = HourlyDataset(f"data/merged_hourly_{year}.csv", verbose=False)
        for sigma in sigmas:
            for i, d in enumerate(ds.dates):
                solar, price = (np.asarray(a, float) for a in ds.get_day(d))
                fc = day_forecast(solar, sigma, year, i)
                base = dict(year=year, sigma=sigma, date=str(d), month=d.month)

                def add(method, seed, cost, sched, interv=0, viol=0, safety=0):
                    rows.append(dict(base, method=method, seed=seed, cost=cost,
                                     schedule=" ".join(map(str, sched)),
                                     interventions=interv, violations=viol,
                                     safety_interventions=safety))

                for name, r in FIXED_RULES.items():
                    add(name, -1, *space.fixed(r, solar, price))
                add("Best block (forecast)", -1, *space.block(fc, price, solar))
                add("Max-plus EMPC (forecast)", -1, *space.empc(fc, price, solar))
                add("Oracle (perfect information)", -1, *space.oracle(solar, price))
                for (v, s), m in models.items():
                    c, sch, iv, vi, sa, _ = rollout(m, v, solar, price, fc)
                    add(VARIANT_LABEL[v], s, c, sch, iv, vi, sa)
        print(f"  evaluated {year}")
    return pd.DataFrame(rows)


def summarise(df):
    """Seed-averaged annual means, spread across seeds, gaps and tests."""
    out = {}
    ref = "Latest start (11-13 h)"
    for (year, sigma), g in df.groupby(["year", "sigma"]):
        key = f"{year}_sigma{sigma}"
        per_day = g.groupby(["method", "date"]).cost.mean().unstack(0)
        oracle = per_day["Oracle (perfect information)"]
        empc = per_day["Max-plus EMPC (forecast)"]
        tab = {}
        for m in per_day.columns:
            seeds = g[g.method == m].groupby("seed").cost.mean()
            ent = dict(mean=float(per_day[m].mean()),
                       seed_std=float(seeds.std(ddof=1)) if len(seeds) > 1 else 0.0,
                       n_seeds=int(len(seeds)),
                       saving_vs_latest_pct=float(100 * (1 - per_day[m].mean()
                                                         / per_day[ref].mean())),
                       gap_to_oracle_pct=float(100 * (per_day[m].mean()
                                                      / oracle.mean() - 1)),
                       violations_days=int((g[g.method == m].violations > 0).sum()),
                       interventions=int(g[g.method == m].interventions.sum()),
                       safety_interventions=int(g[g.method == m].safety_interventions.sum()),
                       safety_days=int((g[g.method == m].safety_interventions > 0).sum()),
                       n_rows=int((g.method == m).sum()))
            if m not in (ref,):
                d = per_day[ref] - per_day[m]
                ent["wilcoxon_p_vs_latest"] = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
            if m != "Max-plus EMPC (forecast)":
                d = per_day[m] - empc
                ent["wilcoxon_p_vs_empc"] = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
                ent["mean_diff_vs_empc"] = float(d.mean())
            tab[m] = ent
        out[key] = tab
    return out


def deadline_shift(seeds, deadlines=(21.0, 19.0), sigma=0.15, years=TEST_YEARS):
    """Deploy the trained agents (trained with D=23) under a tighter deadline.

    The mask and shield are recomputed from the max-plus model with the new D;
    the policies are not retrained. Latest-start rule, EMPC and oracle are
    recomputed for the new D.
    """
    models = {(v, s): load_model(v, s) for v in VARIANT_LABEL for s in seeds}
    rows = []
    for D in deadlines:
        space = ScheduleSpace(deadline=D)
        late = tuple(int(D) - 12 + j for j in range(3))
        for year in years:
            ds = HourlyDataset(f"data/merged_hourly_{year}.csv", verbose=False)
            for i, d in enumerate(ds.dates):
                solar, price = (np.asarray(a, float) for a in ds.get_day(d))
                fc = day_forecast(solar, sigma, year, i)
                base = dict(deadline=D, year=year, date=str(d))
                rows.append(dict(base, method="Latest start", seed=-1,
                                 cost=space.fixed(late, solar, price)[0],
                                 violations=0, safety_interventions=0))
                rows.append(dict(base, method="Max-plus EMPC (forecast)", seed=-1,
                                 cost=space.empc(fc, price, solar)[0],
                                 violations=0, safety_interventions=0))
                rows.append(dict(base, method="Oracle (perfect information)",
                                 seed=-1, cost=space.oracle(solar, price)[0],
                                 violations=0, safety_interventions=0))
                for (v, s), m in models.items():
                    c, sch, iv, vi, sa, _ = rollout(m, v, solar, price, fc, D)
                    rows.append(dict(base, method=VARIANT_LABEL[v], seed=s,
                                     cost=c, violations=vi,
                                     safety_interventions=sa))
        print(f"  deadline {D} done")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "deadline_shift.csv", index=False)
    out = {}
    for D, g in df.groupby("deadline"):
        orc = g[g.method == "Oracle (perfect information)"].cost.mean()
        tab = {}
        for m, h in g.groupby("method"):
            tab[m] = dict(mean=float(h.cost.mean()),
                          gap_to_oracle_pct=float(100 * (h.cost.mean() / orc - 1)),
                          seed_days=int(len(h)),
                          violation_seed_days=int((h.violations > 0).sum()),
                          late_jobs=int(h.violations.sum()),
                          safety_seed_days=int((h.safety_interventions > 0).sum()))
        out[f"D{int(D)}"] = tab
    (RES / "deadline_shift.json").write_text(json.dumps(out, indent=2))
    return out


def timing(seeds):
    """Wall-clock time per decision: max-plus mask, RL policy, EMPC."""
    space = ScheduleSpace()
    ds = HourlyDataset("data/merged_hourly_2020.csv", verbose=False)
    m = load_model("mask", seeds[0])
    t_mask, t_pol, t_empc = [], [], []
    for i, d in enumerate(ds.dates[:100]):
        solar, price = (np.asarray(a, float) for a in ds.get_day(d))
        fc = day_forecast(solar, 0.15, 2020, i)
        env = FactoryMicrogridEnv(stochastic=False, feasibility="mask")
        env.set_day(solar, price, fc)
        obs, _ = env.reset()
        done = False
        while not done:
            t0 = time.perf_counter(); mk = env.action_mask(); t1 = time.perf_counter()
            a, _ = m.predict(obs, action_masks=mk, deterministic=True)
            t2 = time.perf_counter()
            t_mask.append(t1 - t0); t_pol.append(t2 - t1)
            obs, _, done, _, _ = env.step(int(a))
        # full receding-horizon call, including the evaluation of every
        # feasible schedule on the forecast, averaged over the 24 decisions
        t0 = time.perf_counter()
        space.empc(fc, price, solar)
        t_empc.append((time.perf_counter() - t0) / 24)
    ms = lambda v: float(np.mean(v) * 1e3)
    t0 = time.perf_counter()
    ScheduleSpace()
    t_enum = time.perf_counter() - t0
    return dict(mask_ms=ms(t_mask), policy_ms=ms(t_pol), empc_ms=ms(t_empc),
                enumeration_s=float(t_enum), n_feasible_schedules=len(space))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.15, 0.3])
    ap.add_argument("--years", type=int, nargs="+",
                    default=list(TRAIN_YEARS + TEST_YEARS))
    ap.add_argument("--part", action="store_true",
                    help="only evaluate --years and write daily_costs_<year>.csv")
    ap.add_argument("--merge", action="store_true",
                    help="merge daily_costs_<year>.csv files and summarise")
    ap.add_argument("--shift", action="store_true",
                    help="only run the deadline-shift experiment")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    if a.shift:
        for D, tab in deadline_shift(a.seeds).items():
            print(D, {m: (round(e["gap_to_oracle_pct"], 1), e["violation_seed_days"],
                          e["safety_seed_days"]) for m, e in tab.items()})
        return
    if a.part:
        for y in a.years:
            evaluate(a.seeds, a.sigmas, [y]).to_csv(RES / f"daily_costs_{y}.csv",
                                                    index=False)
        return
    if a.merge:
        df = pd.concat([pd.read_csv(RES / f"daily_costs_{y}.csv") for y in a.years],
                       ignore_index=True)
    else:
        df = evaluate(a.seeds, a.sigmas, a.years)
    df.to_csv(RES / "daily_costs.csv", index=False)
    summ = summarise(df)
    (RES / "summary.json").write_text(json.dumps(summ, indent=2))
    (RES / "timing.json").write_text(json.dumps(timing(a.seeds), indent=2))
    for k, tab in summ.items():
        print(f"\n== {k}")
        for m, e in tab.items():
            print(f"  {m:42s} {e['mean']:8.2f} ±{e['seed_std']:5.2f}  "
                  f"gap {e['gap_to_oracle_pct']:6.2f}%  viol-days {e['violations_days']}")


if __name__ == "__main__":
    main()

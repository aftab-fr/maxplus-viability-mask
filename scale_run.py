"""
Train and evaluate the masked agent on the larger instance (revision R2-M1,
R4-3, R4-10).

  python scale_run.py train <seed> [timesteps]
  python scale_run.py eval  <year> <seeds,comma> [n_days]
  python scale_run.py summary
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from data_loader import HourlyDataset
from evaluate import day_forecast
from scale_cpsat import solve
from scale_env import ScaleEnv, schedule_cost
from train_ppo import HYPERPARAMS, N_ENVS, FORECAST_SIGMA_RANGE

N, DT, D = 5, 0.25, 23.0
RUNS = Path("runs")
RES = Path("results")


def train(seed, timesteps=4_000_000):
    torch.set_num_threads(1)
    ds = HourlyDataset([f"data/merged_hourly_{y}.csv" for y in (2018, 2019)], verbose=False)

    def mk(i):
        def f():
            e = ScaleEnv(N, DT, D, dataset=ds, seed=seed * 1000 + i,
                         forecast_sigma_range=FORECAST_SIGMA_RANGE)
            return ActionMasker(e, lambda e: e.action_mask())
        return f
    venv = DummyVecEnv([mk(i) for i in range(N_ENVS)])
    hp = dict(HYPERPARAMS, n_steps=960, batch_size=960)
    model = MaskablePPO("MlpPolicy", venv, seed=seed, verbose=0, **hp)
    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    out = RUNS / f"scale_N{N}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model")
    (out / "meta.json").write_text(json.dumps(dict(seed=seed, timesteps=timesteps,
                                                   wall_s=time.time() - t0, N=N, dt=DT)))
    print("trained", seed, time.time() - t0)


def rollout(model, s, p, fc):
    env = ScaleEnv(N, DT, D)
    env.set_day(s, p, fc)
    obs, _ = env.reset()
    done, tt = False, []
    while not done:
        t1 = time.perf_counter()
        mk = env.action_mask()
        a, _ = model.predict(obs, action_masks=mk, deterministic=True)
        tt.append(time.perf_counter() - t1)
        obs, _, done, _, _ = env.step(int(a))
    return env.cost, tuple(env.sched), float(np.mean(tt))


def best_block(fc, p, spacing):
    """Best start epoch for releases every `spacing` epochs (forecast)."""
    best = None
    for s0 in range(int(24 / DT)):
        sch = tuple(s0 + j * spacing for j in range(N))
        if sch[-1] >= 24 / DT:
            break
        c, y = schedule_cost(sch, fc, p, N, DT)
        if y <= D + 1e-9 and (best is None or c < best[0]):
            best = (c, sch)
    return best[1]


def latest_block(spacing):
    for s0 in reversed(range(int(24 / DT))):
        sch = tuple(s0 + j * spacing for j in range(N))
        if sch[-1] < 24 / DT and schedule_cost(sch, np.zeros(24), np.zeros(24), N, DT)[1] <= D + 1e-9:
            return sch


def evaluate(year, seeds, n_days=None):
    ds = HourlyDataset(f"data/merged_hourly_{year}.csv", verbose=False)
    models = {s: MaskablePPO.load(RUNS / f"scale_N{N}_s{s}" / "model", device="cpu")
              for s in seeds}
    lat = latest_block(4)
    rows = []
    days = list(enumerate(ds.dates))
    if n_days:
        days = days[::max(1, len(days) // n_days)][:n_days]
    for i, d in days:
        s, p = (np.asarray(a, float) for a in ds.get_day(d))
        fc = day_forecast(s, 0.15, year, i)
        b = dict(year=year, date=str(d))
        orc, el_o, st_o = solve(s, p, N, DT, D, time_limit=60, workers=1)
        emp, el_e, st_e = solve(fc, p, N, DT, D, time_limit=60, workers=1)
        rows.append(dict(b, method="Oracle (CP-SAT)", seed=-1,
                         cost=schedule_cost(orc, s, p, N, DT)[0], time_s=el_o, status=st_o))
        rows.append(dict(b, method="Max-plus EMPC (CP-SAT)", seed=-1,
                         cost=schedule_cost(emp, s, p, N, DT)[0], time_s=el_e, status=st_e))
        rows.append(dict(b, method="Latest start", seed=-1,
                         cost=schedule_cost(lat, s, p, N, DT)[0]))
        rows.append(dict(b, method="Best block", seed=-1,
                         cost=schedule_cost(best_block(fc, p, 4), s, p, N, DT)[0]))
        for sd, m in models.items():
            c, sch, tdec = rollout(m, s, p, fc)
            y = schedule_cost(sch, s, p, N, DT)[1]
            rows.append(dict(b, method="MaskablePPO + mask", seed=sd, cost=c,
                             time_s=tdec, late=int(y > D + 1e-9)))
        if i % 30 == 0:
            print(year, i, flush=True)
    pd.DataFrame(rows).to_csv(RES / f"scale_eval_{year}.csv", index=False)


def summary():
    df = pd.concat([pd.read_csv(f) for f in RES.glob("scale_eval_*.csv")])
    out = {}
    for y, g in df.groupby("year"):
        o = g[g.method == "Oracle (CP-SAT)"].cost.mean()
        lt = g[g.method == "Latest start"].cost.mean()
        out[str(y)] = {m: dict(mean=float(h.cost.mean()),
                               gap_pct=float(100 * (h.cost.mean() / o - 1)),
                               saving_vs_latest_pct=float(100 * (1 - h.cost.mean() / lt)),
                               mean_time_s=float(h.time_s.mean()) if "time_s" in h else None,
                               max_time_s=float(h.time_s.max()) if "time_s" in h else None,
                               late=int(h.late.sum()) if "late" in h and h.late.notna().any() else 0,
                               seed_gaps=[float(100 * (v / o - 1)) for v in
                                          h.groupby("seed").cost.mean()] if (h.seed >= 0).any() else None)
                        for m, h in g.groupby("method")}
    (RES / "scale_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 4_000_000)
    elif cmd == "eval":
        evaluate(int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")],
                 int(sys.argv[4]) if len(sys.argv) > 4 else None)
    else:
        summary()

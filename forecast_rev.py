"""
Forecast realism (revision: R2-M5, R4-4).

Scenarios on the held-out years (all methods see the same forecasts):
  ar1_rho0.3 / ar1_rho0.7 / ar1_rho0.95 : synthetic AR(1) error, sigma = 0.15
  real                                   : empirical day-ahead error of the
                                           French TSO solar forecast (fetch_forecast.py)
  ar1_rho0.7+intraday, real+intraday     : the forecast is updated every hour
      from the PV measured in the previous hour,
         S_hat_t(tau) = S_hat(tau) * (1 + alpha^(tau-t+1) * (S(t-1)/S_hat(t-1) - 1)),
      alpha = 0.8 (exponentially fading bias correction, a standard intraday
      persistence update); EMPC re-optimizes every hour on the updated
      forecast, the agents (not retrained) observe it.

Output: results/forecast_rev.csv, results/forecast_rev.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import FIXED_RULES, ScheduleSpace
from data_loader import HourlyDataset
from evaluate import load_model, VARIANT_LABEL
from factory_env import FactoryMicrogridEnv, T_HORIZON, grid_cost, solar_forecast
from fetch_forecast import real_forecast

RES = Path("results")
ALPHA = 0.8
SIGMA = 0.15


def base_forecast(scen, solar, year, i, date):
    if scen.startswith("real"):
        return real_forecast(solar, year, date)
    rho = float(scen.split("rho")[1].split("+")[0])
    return solar_forecast(solar, SIGMA, np.random.default_rng([year, i, 150, int(rho * 100)]),
                          rho=rho)


def updated(fc, solar, t):
    """Forecast available at epoch t (uses actual PV up to t-1)."""
    if t == 0:
        return fc
    prev = fc[t - 1]
    if prev < 20.0:          # night / negligible PV: no information
        return fc
    ratio = solar[t - 1] / prev - 1.0
    out = fc.copy()
    h = np.arange(t, T_HORIZON) - t + 1
    out[t:] = np.clip(fc[t:] * (1 + ALPHA ** h * ratio), 0.0, 2000.0)
    return out


def empc(space, fc, solar, price, intraday):
    released = []
    for t in range(T_HORIZON):
        k = len(released)
        if k == space.n_jobs:
            break
        f = updated(fc, solar, t) if intraday else fc
        c = grid_cost(space.L, f, price)
        ok = np.ones(len(space.S), bool)
        if k:
            ok &= (space.S[:, :k] == np.array(released)).all(1)
        ok &= space.S[:, k] >= t
        i = np.flatnonzero(ok)[np.argmin(c[ok])]
        if space.S[i, k] == t:
            released.append(t)
    return space.cost(tuple(released), solar, price)


def rollout(model, variant, solar, price, fc, intraday):
    env = FactoryMicrogridEnv(stochastic=False, feasibility=variant)
    env.set_day(solar, price, fc)
    obs, _ = env.reset()
    done = False
    while not done:
        if intraday:
            env.forecast = updated(fc, solar, env.hour)
            obs = env._obs()
        if variant == "mask":
            a, _ = model.predict(obs, action_masks=env.action_mask(), deterministic=True)
        else:
            a, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(a))
    return env.total_cost, info["ep_violations"]


def main(seeds, scens):
    space = ScheduleSpace()
    models = {(v, s): load_model(v, s) for v in VARIANT_LABEL for s in seeds}
    rows = []
    for scen in scens:
        intraday = scen.endswith("+intraday")
        for year in (2020, 2023):
            ds = HourlyDataset(f"data/merged_hourly_{year}.csv", verbose=False)
            for i, d in enumerate(ds.dates):
                solar, price = (np.asarray(a, float) for a in ds.get_day(d))
                fc = base_forecast(scen, solar, year, i, d)
                if fc is None:
                    continue
                b = dict(scenario=scen, year=year, date=str(d))
                rows.append(dict(b, method="Oracle", seed=-1,
                                 cost=space.oracle(solar, price)[0]))
                rows.append(dict(b, method="Latest start", seed=-1,
                                 cost=space.fixed(FIXED_RULES["Latest start (11-13 h)"],
                                                  solar, price)[0]))
                rows.append(dict(b, method="Best block", seed=-1,
                                 cost=space.block(fc, price, solar)[0]))
                rows.append(dict(b, method="EMPC", seed=-1,
                                 cost=empc(space, fc, solar, price, intraday)))
                for (v, s), m in models.items():
                    c, viol = rollout(m, v, solar, price, fc, intraday)
                    rows.append(dict(b, method=VARIANT_LABEL[v], seed=s, cost=c,
                                     violations=viol))
            print(scen, year, flush=True)
    df = pd.DataFrame(rows)
    tag = "_".join(s.replace("+", "p") for s in scens)
    df.to_csv(RES / f"forecast_rev_{tag}.csv", index=False)


def summarise():
    df = pd.concat([pd.read_csv(f) for f in RES.glob("forecast_rev_*.csv")])
    out = {}
    for (sc, y), g in df.groupby(["scenario", "year"]):
        o = g[g.method == "Oracle"].cost.mean()
        out[f"{sc}|{y}"] = {m: dict(mean=float(h.cost.mean()),
                                    gap_pct=float(100 * (h.cost.mean() / o - 1)),
                                    n=int(len(h)))
                            for m, h in g.groupby("method")}
    (RES / "forecast_rev.json").write_text(json.dumps(out, indent=2))
    for k, t in out.items():
        print(k, {m[:14]: round(e["gap_pct"], 2) for m, e in t.items()})


def errstats():
    """Statistics of the empirical day-ahead errors applied to the site."""
    rel, ac, nmae = [], [], []
    for y in (2020, 2023):
        d = pd.read_csv(f"data/fr_solar_forecast_{y}.csv")
        nmae.append(float((d.forecast_mw - d.actual_mw).abs().sum() / d.actual_mw.sum()))
        ds = HourlyDataset(f"data/merged_hourly_{y}.csv", verbose=False)
        for dt in ds.dates:
            s, _ = (np.asarray(a, float) for a in ds.get_day(dt))
            f = real_forecast(s, y, dt)
            if f is None:
                continue
            m = s > 0.05 * max(s.max(), 1.0)
            r = f[m] / s[m] - 1
            rel += list(r)
            if m.sum() > 3:
                ac.append(np.corrcoef(r[:-1], r[1:])[0, 1])
        ac_y = float(np.nanmean(ac))
        ac = [] if y == 2020 else ac
        if y == 2020:
            ac20 = ac_y
    out = dict(rel_std=float(np.std(rel)), nmae_min=min(nmae), nmae_max=max(nmae),
               ac_min=min(ac20, ac_y), ac_max=max(ac20, ac_y))
    (RES / "forecast_errstats.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    if sys.argv[1] == "errstats":
        errstats()
    elif sys.argv[1] == "summary":
        summarise()
    else:
        seeds = [int(x) for x in sys.argv[1].split(",")]
        main(seeds, sys.argv[2:])

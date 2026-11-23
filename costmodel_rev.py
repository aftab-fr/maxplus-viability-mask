"""
Sensitivity to the energy-cost model (revision: R2-M6, R4-9).

Scenarios (illustrative values of the order of French C4/HTA industrial
tariffs; all costs per day):
  E    energy only (paper):            sum (L-S)^+ * pi
  N    + network energy charge:        + sum (L-S)^+ * tau(t),
                                         tau = 0.030 EUR/kWh 07-23 h, 0.015 otherwise
  ND   + demand charge:                + lambda * max_t (L-S)^+,
                                         lambda = 0.40 EUR/kW/day (~12 EUR/kW/month)
  NDX  + export remuneration at the day-ahead price (negative prices are paid):
                                       - sum (S-L)^+ * pi
  B    energy only + battery 1 MWh / 0.5 MW, round-trip efficiency 0.9,
       self-consumption rule (charge from surplus PV, discharge to cover the
       residual load), empty at 00 h
  ALL  NDX + battery

For every scenario: fixed rules and the agents' schedules (trained on E,
not retrained) are re-priced; best block, EMPC and oracle are re-optimized
with the scenario cost (EMPC on the forecast, oracle on the actual PV).
The mask and its guarantees do not depend on the cost.

Output: results/costmodel.json, results/costmodel_days.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import FIXED_RULES, ScheduleSpace
from data_loader import HourlyDataset
from evaluate import day_forecast
from factory_env import load_profile

RES = Path("results")
SIGMA = 0.15
HOURS = np.arange(24)
TAU = np.where((HOURS >= 7) & (HOURS < 23), 0.030, 0.015)
LAMBDA = 0.40
BAT_E, BAT_P, ETA = 1000.0, 500.0, 0.9 ** 0.5   # kWh, kW, one-way efficiency

SCEN = {"E": dict(), "N": dict(net=True), "ND": dict(net=True, dem=True),
        "NDX": dict(net=True, dem=True, exp=True), "B": dict(bat=True),
        "ALL": dict(net=True, dem=True, exp=True, bat=True)}


def net_import(L, S, bat):
    """Grid import and export (K,24) after an optional battery."""
    r = L - S[None, :]
    if not bat:
        return np.maximum(r, 0), np.maximum(-r, 0)
    K = L.shape[0]
    soc = np.zeros(K)
    imp, exp = np.zeros_like(L), np.zeros_like(L)
    for t in range(24):
        rt = r[:, t]
        ch = np.minimum(np.minimum(np.maximum(-rt, 0), BAT_P), (BAT_E - soc) / ETA)
        dis = np.minimum(np.minimum(np.maximum(rt, 0), BAT_P), soc * ETA)
        soc = soc + ch * ETA - dis / ETA
        imp[:, t] = np.maximum(rt, 0) - dis
        exp[:, t] = np.maximum(-rt, 0) - ch
    return imp, exp


def cost(L, S, price, net=False, dem=False, exp=False, bat=False):
    L = np.atleast_2d(L)
    imp, ex = net_import(L, S, bat)
    c = (imp * price).sum(1)
    if net:
        c += (imp * TAU).sum(1)
    if dem:
        c += LAMBDA * imp.max(1)
    if exp:
        c -= (ex * price).sum(1)
    return c


def main():
    df = pd.read_csv(RES / "daily_costs.csv")
    df = df[(df.sigma == SIGMA) & df.year.isin([2020, 2023])]
    rl = [m for m in df.method.unique() if df[df.method == m].seed.max() >= 0]
    space = ScheduleSpace()
    rows = []
    for year in (2020, 2023):
        ds = HourlyDataset(f"data/merged_hourly_{year}.csv", verbose=False)
        g = df[df.year == year]
        for i, d in enumerate(ds.dates):
            solar, price = (np.asarray(a, float) for a in ds.get_day(d))
            fc = day_forecast(solar, SIGMA, year, i)
            gd = g[g.date == str(d)]
            for sc, kw in SCEN.items():
                base = dict(year=year, date=str(d), scenario=sc)
                ca = cost(space.L, solar, price, **kw)
                cf = cost(space.L, fc, price, **kw)
                rows.append(dict(base, method="Oracle", seed=-1, cost=float(ca.min())))
                rows.append(dict(base, method="EMPC", seed=-1,
                                 cost=float(ca[int(np.argmin(cf))])))
                # EMPC that ignores the scenario (optimizes energy only)
                ce = cost(space.L, fc, price)
                rows.append(dict(base, method="EMPC (energy-only model)", seed=-1,
                                 cost=float(ca[int(np.argmin(ce))])))
                blocks = [space.index[tuple(range(s, s + 3))] for s in range(24)
                          if tuple(range(s, s + 3)) in space.index]
                rows.append(dict(base, method="Best block", seed=-1,
                                 cost=float(ca[blocks[int(np.argmin(cf[blocks]))]])))
                for name, r in FIXED_RULES.items():
                    rows.append(dict(base, method=name, seed=-1,
                                     cost=float(ca[space.index[r]])))
                for m in rl:
                    for _, r in gd[gd.method == m].iterrows():
                        sch = tuple(int(s) for s in str(r.schedule).split())
                        if sch in space.index:
                            c = float(ca[space.index[sch]])
                        else:
                            c = float(cost(load_profile(space.line, sch)[0][None],
                                           solar, price, **kw)[0])
                        rows.append(dict(base, method=m, seed=int(r.seed), cost=c))
        print("done", year, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(RES / "costmodel_days.csv", index=False)
    out = {}
    for (sc, year), g in R.groupby(["scenario", "year"]):
        orc = g[g.method == "Oracle"].cost.mean()
        lat = g[g.method == "Latest start (11-13 h)"].cost.mean()
        out[f"{sc}|{year}"] = {
            m: dict(mean=float(h.cost.mean()),
                    gap_to_oracle_pct=float(100 * (h.cost.mean() / orc - 1)),
                    saving_vs_latest_pct=float(100 * (1 - h.cost.mean() / lat)))
            for m, h in g.groupby("method")}
    (RES / "costmodel.json").write_text(json.dumps(out, indent=2))
    for k, tab in out.items():
        print("==", k)
        for m, e in tab.items():
            print(f"   {m[:40]:40s} {e['mean']:8.2f}  gap {e['gap_to_oracle_pct']:6.1f}%  sav {e['saving_vs_latest_pct']:6.1f}%")


if __name__ == "__main__":
    main()

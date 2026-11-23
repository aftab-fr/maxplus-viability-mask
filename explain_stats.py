"""
Quantify the decision explanations (RQ3) on the held-out years.

For every release of the proposed agent (all seeds) and of the max-plus EMPC:
share of forced releases, mean slack, and the local-consistency rate
(the chosen hour has the lowest forecast marginal cost among the admissible
hours t-1, t, t+1). Also writes one operator report for the Gantt day.

Output: results/explain_stats.json, results/explanation_example.txt
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import HourlyDataset
from evaluate import day_forecast
from explain import explain_schedule, render
from factory_env import FactoryMicrogridEnv
from figures import M_EMPC, M_MASK, pick_gantt_day

RES = Path("results")


def main(sigma=0.15):
    df = pd.read_csv(RES / "daily_costs.csv")
    factory = lambda: FactoryMicrogridEnv(stochastic=False, feasibility="mask")
    stats = {}
    for year in (2020, 2023):
        ds = HourlyDataset(f"data/merged_hourly_{year}.csv", verbose=False)
        idx = {str(d): i for i, d in enumerate(ds.dates)}
        g = df[(df.year == year) & (df.sigma == sigma)]
        for m in (M_MASK, M_EMPC):
            rec = []
            for _, r in g[g.method == m].iterrows():
                i = idx[r.date]
                solar, price = (np.asarray(a, float) for a in ds.get_day(ds.dates[i]))
                fc = day_forecast(solar, sigma, year, i)
                sched = tuple(int(s) for s in str(r.schedule).split())
                for e in explain_schedule(factory, sched, solar, price, fc):
                    rec.append(dict(forced=e["forced"], slack=e["slack_h"],
                                    consistent=e["locally_consistent"],
                                    pv=e["pv_share_forecast"],
                                    release_by=[c["driver"] for c in e["critical_path"]]))
            n = len(rec)
            stats[f"{year}|{m}"] = dict(
                releases=n,
                forced_pct=100 * np.mean([x["forced"] for x in rec]),
                mean_slack_h=float(np.mean([x["slack"] for x in rec])),
                locally_consistent_pct=100 * np.mean([x["consistent"] for x in rec]),
                mean_pv_share_pct=100 * np.mean([x["pv"] for x in rec]),
                oven_driven_by_previous_job_pct=100 * np.mean(
                    [x["release_by"][-1] == "previous job" for x in rec]))
            print(year, m, {k: round(v, 2) for k, v in stats[f"{year}|{m}"].items()})
    (RES / "explain_stats.json").write_text(json.dumps(stats, indent=2))

    # operator report for the Gantt day (seed 0)
    date = pick_gantt_day(df)
    ds = HourlyDataset("data/merged_hourly_2023.csv", verbose=False)
    i = [str(d) for d in ds.dates].index(date)
    solar, price = (np.asarray(a, float) for a in ds.get_day(ds.dates[i]))
    fc = day_forecast(solar, sigma, 2023, i)
    r = df[(df.year == 2023) & (df.sigma == sigma) & (df.date == date)
           & (df.method == M_MASK) & (df.seed == 0)].iloc[0]
    sched = tuple(int(s) for s in r.schedule.split())
    txt = f"Day {date}, MaskablePPO seed 0, schedule {sched}\n" + \
        render(explain_schedule(factory, sched, solar, price, fc))
    (RES / "explanation_example.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()

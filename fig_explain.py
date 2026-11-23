"""
Decision-level visualization of the model-grounded explanations
(revision: R4-8, R2-M7). Day of Fig. 7 (July 2023), masked agent seed 0.

(a) slack of the remaining jobs over the day with the agent's releases and
    the epochs at which the mask forbids waiting (forced) -- what the
    operator sees as "how long can I still wait?";
(b) for each job, the re-optimized forecast day cost as a function of its
    release hour (all later jobs re-optimized), with the admissible window,
    the agent's choice and the model's best hour -- the full counterfactual
    curve behind the +-1 h contrast in the text report.

Output: figures/fig8_explain.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figures  # noqa: F401  (style)
from figures import M_MASK, COL, INK2, pick_gantt_day, FIG, RES
from baselines import ScheduleSpace
from data_loader import HourlyDataset
from evaluate import day_forecast
from explain import best_completion
from factory_env import FactoryMicrogridEnv, T_HORIZON


def main(sigma=0.15, seed=0):
    df = pd.read_csv(RES / "daily_costs.csv")
    date = pick_gantt_day(df)
    ds = HourlyDataset("data/merged_hourly_2023.csv", verbose=False)
    i = [str(d) for d in ds.dates].index(date)
    solar, price = (np.asarray(a, float) for a in ds.get_day(ds.dates[i]))
    fc = day_forecast(solar, sigma, 2023, i)
    r = df[(df.year == 2023) & (df.sigma == sigma) & (df.date == date)
           & (df.method == M_MASK) & (df.seed == seed)].iloc[0]
    sched = tuple(int(s) for s in r.schedule.split())
    space = ScheduleSpace()

    # (a) slack trajectory
    env = FactoryMicrogridEnv(stochastic=False, feasibility="mask")
    env.set_day(solar, price, fc)
    env.reset()
    slack, forced = [], []
    for t in range(T_HORIZON):
        slack.append(env.slack() if env.remaining else np.nan)
        forced.append(env.remaining > 0 and not env.viable_wait())
        env.step(1 if t in sched else 0)

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = axs[0]
    ax.step(range(T_HORIZON), slack, where="post", color=COL[M_MASK])
    for k, t in enumerate(sched):
        ax.axvline(t, color=INK2, lw=0.6, ls=":")
        ax.text(t + 0.2, max(np.nanmax(slack), 1) * 0.95, f"J{k + 1}", fontsize=7, color=INK2)
    ft = [t for t, f in enumerate(forced) if f]
    if ft:
        ax.scatter(ft, [0] * len(ft), marker="x", color="#d03b3b", label="waiting forbidden")
        ax.legend(loc="lower left")
    ax.set_xlabel("hour")
    ax.set_ylabel("slack of remaining jobs [h]")
    ax.set_title("(a) admissibility: slack and mask", loc="left")
    ax.set_xlim(0, 23)

    # (b) counterfactual cost curves
    ax = axs[1]
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    for k, t in enumerate(sched):
        prefix = list(sched[:k])
        hrs, cost = [], []
        for u in range(T_HORIZON):
            if prefix and u <= prefix[-1]:
                continue
            c = best_completion(space, prefix, u, fc, price)
            if c is not None:
                hrs.append(u)
                cost.append(c)
        cost = np.array(cost)
        ax.plot(hrs, cost, color=cols[k], marker=".", ms=3, label=f"job {k + 1}")
        j = hrs.index(t)
        ax.scatter([t], [cost[j]], s=40, facecolor="none", edgecolor=cols[k], lw=1.4, zorder=5)
        b = int(np.argmin(cost))
        ax.scatter([hrs[b]], [cost[b]], marker="*", s=40, color=cols[k], zorder=6)
    ax.set_xlabel("release hour of the job (later jobs re-optimized)")
    ax.set_ylabel("forecast day cost [EUR]")
    ax.set_title("(b) contrast: counterfactual cost of each release", loc="left")
    ax.legend(loc="upper right", ncol=3)
    ax.text(0.01, 0.93, "o agent's choice    * model optimum", transform=ax.transAxes,
            fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "fig8_explain.pdf")
    print("saved", date, sched)


if __name__ == "__main__":
    main()

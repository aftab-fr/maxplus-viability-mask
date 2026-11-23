"""
Generate the figures of the JCAES manuscript from results/ and runs/.

    python figures.py

fig1_profiles.pdf        PV and day-ahead price profiles (real data)
fig3_learning.pdf        training cost of the three feasibility variants
fig4_infeasible.pdf      shield interventions / deadline violations in training
fig5_gap.pdf             gap to the perfect-information optimum, all years
fig6_robustness.pdf      gap vs PV forecast error (held-out years)
fig7_gantt.pdf           schedules on a July 2023 day
(fig2 is a TikZ block diagram inside the manuscript.)
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from baselines import ScheduleSpace  # noqa: E402
from data_loader import HourlyDataset  # noqa: E402
from evaluate import (VARIANT_LABEL, day_forecast, load_model,  # noqa: E402
                      rollout)
from factory_env import MACHINE_KW, load_profile, make_line  # noqa: E402

FIG = Path("figures")
FIG.mkdir(exist_ok=True)
RES = Path("results")

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "savefig.bbox": "tight", "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "grid.color": "#e4e3de",
    "grid.linewidth": 0.5, "lines.linewidth": 1.5, "legend.frameon": False,
})
INK2 = "#52514e"

# fixed method -> colour mapping (categorical order of the reference palette)
M_MASK, M_SHIELD, M_PEN = (VARIANT_LABEL[k] for k in ("mask", "shield", "penalty"))
M_DAY, M_LATE = "Day shift (08-10 h)", "Latest start (11-13 h)"
M_BLOCK, M_EMPC = "Best block (forecast)", "Max-plus EMPC (forecast)"
M_OR = "Oracle (perfect information)"
COL = {M_MASK: "#2a78d6", M_SHIELD: "#eb6834", M_PEN: "#1baf7a",
       M_DAY: "#eda100", M_LATE: "#e87ba4", M_BLOCK: "#008300",
       M_EMPC: "#4a3aa7", M_OR: INK2}
SHORT = {M_MASK: "MaskablePPO+mask", M_SHIELD: "PPO+shield",
         M_PEN: "PPO+penalty", M_DAY: "Day shift", M_LATE: "Latest start",
         M_BLOCK: "Best block", M_EMPC: "Max-plus EMPC", M_OR: "Oracle"}
YEAR_COL = {2018: "#2a78d6", 2019: "#eb6834", 2020: "#1baf7a", 2023: "#4a3aa7"}


def _grid(ax, axis="y"):
    ax.grid(True, axis=axis)
    ax.set_axisbelow(True)


# --------------------------------------------------------------- fig 1 ----
def fig1_profiles():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ds = {y: HourlyDataset(f"data/merged_hourly_{y}.csv", verbose=False)
          for y in (2018, 2019, 2020, 2023)}
    h = np.arange(24)
    for mo, ls, lab in ((7, "-", "July"), (1, "--", "January")):
        pv, _ = ds[2019].monthly_avg_profile(mo)
        a.plot(h, pv, ls, color="#eda100", label=lab)
    a.set(xlabel="Hour of day (CET)", ylabel="Mean PV output (kW)",
          title="(a) PV output, Belfort 2 MWp (2019)", xlim=(0, 23))
    a.legend(loc="upper left")
    _grid(a)
    for y, d in ds.items():
        _, pr = d.monthly_avg_profile(7)
        b.plot(h, pr / pr.mean(), color=YEAR_COL[y], label=str(y))
    b.axhline(1, color="#8a8984", lw=0.6)
    b.set(xlabel="Hour of day (CET)", ylabel="Price / daily mean",
          title="(b) French day-ahead price shape, July", xlim=(0, 23))
    b.legend(loc="upper left", ncol=2)
    _grid(b)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_profiles.pdf")
    plt.close(fig)


# ------------------------------------------------------------- fig 3/4 ----
def _episodes(variant):
    out = []
    for p in sorted(Path("runs").glob(f"{variant}_s[0-9]")):
        out.append(pd.read_csv(p / "episodes.csv"))
    return out


def _binned(dfs, col, bins):
    ys = []
    for d in dfs:
        idx = np.digitize(d.timestep, bins)
        ys.append([d[col][idx == k].mean() for k in range(1, len(bins))])
    ys = np.array(ys)
    return ys.mean(0), ys.std(0)


def fig3_learning(oracle_train_mean):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    bins = np.linspace(0, 2_000_000, 41)
    xc = 0.5 * (bins[1:] + bins[:-1]) / 1e6
    for v, lab in VARIANT_LABEL.items():
        dfs = _episodes(v)
        if not dfs:
            continue
        m, s = _binned(dfs, "cost", bins)
        ax.plot(xc, m, color=COL[lab], label=SHORT[lab])
        ax.fill_between(xc, m - s, m + s, color=COL[lab], alpha=0.15, lw=0)
    ax.axhline(oracle_train_mean, color=INK2, ls="--", lw=1)
    ax.text(2.0, oracle_train_mean, "oracle", ha="right", va="bottom",
            color=INK2, fontsize=7)
    ax.set(xlabel="Training steps (millions)",
           ylabel="Grid cost per episode (EUR)", xlim=(0, 2.0))
    ax.legend(loc="upper right")
    _grid(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_learning.pdf")
    plt.close(fig)


def fig4_infeasible():
    """Cumulative deadline-relevant events during training, per run."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    grid = np.linspace(0, 2_000_000, 201)
    for ax, v, col, ttl, yl in (
            (a, "shield", "safety_interventions", "(a) PPO + post-hoc shield",
             "Cumulative shield overrides"),
            (b, "penalty", "violations", "(b) PPO + deadline penalty",
             "Cumulative late/unreleased jobs")):
        lab = VARIANT_LABEL[v]
        eps = _episodes(v)
        for k, e in enumerate(eps):
            cum = np.interp(grid, e.timestep, e[col].cumsum())
            ax.plot(grid / 1e6, cum, color=COL[lab], lw=1.0, alpha=0.85,
                    label=f"{len(eps)} seeds" if k == 0 else None)
        ax.plot([0, 2], [0, 0], color=COL[M_MASK], lw=1.5, ls="--",
                label="MaskablePPO+mask (0 by construction)")
        ax.set(xlabel="Training steps (millions)", ylabel=yl, title=ttl,
               xlim=(0, 2.0))
        top = ax.get_ylim()[1]
        ax.set_ylim(-0.02 * top, 1.35 * top)          # headroom for the legend
        ax.legend(loc="upper left", fontsize=7)
        _grid(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_infeasible.pdf")
    plt.close(fig)


# --------------------------------------------------------------- fig 5 ----
def fig5_gap(summary, sigma=0.15):
    methods = [M_BLOCK, M_PEN, M_SHIELD, M_MASK, M_EMPC]
    years = [2018, 2019, 2020, 2023]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        xs, ys, es = [], [], []
        for k, y in enumerate(years):
            e = summary[f"{y}_sigma{sigma}"].get(m)
            if e is None:
                continue
            orc = summary[f"{y}_sigma{sigma}"][M_OR]["mean"]
            xs.append(k + (j - (len(methods) - 1) / 2) * w)
            ys.append(e["gap_to_oracle_pct"])
            es.append(100 * e["seed_std"] / orc)
        ax.bar(xs, ys, width=w * 0.9, color=COL[m], label=SHORT[m],
               yerr=es if any(es) else None,
               error_kw=dict(lw=0.6, capsize=1.5, ecolor="#0b0b0b"))
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([f"{y}\n({'train' if y < 2020 else 'test'})" for y in years])
    ax.set_ylabel("Gap to oracle (%)")
    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, 1.15))
    _grid(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_gap.pdf")
    plt.close(fig)


# --------------------------------------------------------------- fig 6 ----
def fig6_robustness(summary, sigmas=(0.0, 0.15, 0.3)):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=False)
    for ax, y in zip(axes, (2020, 2023)):
        for m in (M_BLOCK, M_MASK, M_EMPC):
            g = [summary[f"{y}_sigma{s}"][m]["gap_to_oracle_pct"] for s in sigmas]
            ax.plot(sigmas, g, "o-", color=COL[m], label=SHORT[m], ms=4)
        ax.set(xlabel=r"PV forecast error $\sigma$", ylabel="Gap to oracle (%)",
               title=f"({'a' if y == 2020 else 'b'}) {y} (held-out)",
               xticks=sigmas)
        _grid(ax)
    axes[0].legend(loc="center left")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_robustness.pdf")
    plt.close(fig)


# --------------------------------------------------------------- fig 7 ----
def pick_gantt_day(df):
    g = df[(df.year == 2023) & (df.sigma == 0.15) & (df.month == 7)]
    p = g.groupby(["date", "method"]).cost.mean().unstack()
    return (p[M_LATE] - p[M_OR]).idxmax()


def fig7_gantt(df, seed=0):
    date = pick_gantt_day(df)
    ds = HourlyDataset("data/merged_hourly_2023.csv", verbose=False)
    d = [x for x in ds.dates if str(x) == date][0]
    i = ds.dates.index(d)
    solar, price = (np.asarray(a, float) for a in ds.get_day(d))
    fc = day_forecast(solar, 0.15, 2023, i)
    space = ScheduleSpace()
    model = load_model("mask", seed)
    rl_cost, rl_s, *_ = rollout(model, "mask", solar, price, fc)
    or_cost, or_s = space.oracle(solar, price)
    la_cost, la_s = space.fixed((11, 12, 13), solar, price)
    rows = [(M_LATE, la_s, la_cost), (M_MASK, rl_s, rl_cost), (M_OR, or_s, or_cost)]

    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 0.8, 1.6], hspace=0.6)
    a, b, c = (fig.add_subplot(gs[k]) for k in range(3))
    h = np.arange(24)
    a.bar(h + 0.5, solar, width=0.9, color="#eda100", label="PV (actual)")
    a.step(np.r_[h, 24], np.r_[fc, fc[-1]], where="post", color=INK2, lw=1,
           ls="--", label="PV forecast")
    L_rl, _ = load_profile(make_line(), rl_s)
    a.step(np.r_[h, 24], np.r_[L_rl, L_rl[-1]], where="post",
           color=COL[M_MASK], lw=1.5, label="Load (MaskablePPO)")
    a.set(ylabel="kW", xlim=(0, 24), title=f"(a) PV and load, {date}")
    a.legend(ncol=1, loc="upper left", fontsize=7)
    _grid(a)
    b.step(np.r_[h, 24], np.r_[price, price[-1]] * 1000, where="post",
           color="#4a3aa7")
    b.set(ylabel="EUR/MWh", xlim=(0, 24), title="(b) Day-ahead price")
    _grid(b)

    line = make_line()
    names = ["prep", "treat.", "oven"]
    shades = [0.35, 0.65, 1.0]
    yt, yl = [], []
    for r, (m, sched, cost) in enumerate(rows):
        from maxplus import EPS
        x = np.full(line.n, EPS)
        for u in sched:
            xn = line.next_state(x, float(u))
            for mi, (s, e) in enumerate(line.busy_windows(x, xn)):
                y0 = r * 4 + (2 - mi)
                c.barh(y0, e - s, left=s, height=0.8, color=COL[m],
                       alpha=shades[mi], edgecolor="white", lw=1)
            x = xn
        for mi in range(3):
            yt.append(r * 4 + (2 - mi))
            yl.append(f"{SHORT[m]}: {names[mi]}" if mi == 1 else names[mi])
        c.text(24.3, r * 4 + 1, f"{cost:.0f} EUR", va="center", fontsize=8)
    c.axvline(23, color="#e34948", lw=1, ls=":")
    c.set_yticks(yt)
    c.set_yticklabels(yl, fontsize=7)
    c.set(xlim=(0, 24), xlabel="Hour of day (CET)", title="(c) Machine schedules")
    c.invert_yaxis()
    _grid(c, "x")
    fig.savefig(FIG / "fig7_gantt.pdf")
    plt.close(fig)
    return date, rows


def main():
    df = pd.read_csv(RES / "daily_costs.csv")
    summary = json.loads((RES / "summary.json").read_text())
    space = ScheduleSpace()
    tr = HourlyDataset([f"data/merged_hourly_{y}.csv" for y in (2018, 2019)],
                       verbose=False)
    orc = np.mean([space.oracle(*(np.asarray(a, float) for a in tr.get_day(d)))[0]
                   for d in tr.dates])
    fig1_profiles()
    fig3_learning(orc)
    fig4_infeasible()
    fig5_gap(summary)
    fig6_robustness(summary)
    date, rows = fig7_gantt(df)
    print("gantt day", date, [(SHORT[m], s, round(c, 1)) for m, s, c in rows])
    print("figures written to", FIG)


if __name__ == "__main__":
    main()

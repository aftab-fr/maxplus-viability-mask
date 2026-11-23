"""
Figures for the experiments added in the revision (journal PDF, static).

fig9_deploy.pdf      (a) per-seed gap of the three RL variants, (b) late seed-days
                     under the tightened deadline for every unmasked variant
fig10_robust.pdf     (a) gap under synthetic and real forecast errors,
                     (b) late days when delays exceed the assumed bound
fig11_agnostic.pdf   (a) mask on five policies (late days with/without mask),
                     (b) gap under extended tariffs
fig12_audit.pdf      audit of the explanations (consistency, gap located,
                     ex-post validity, surrogate fidelity)
fig13_scale.pdf      larger instance: (a) gap to the oracle, (b) decision time

Palette: the categorical colours of figures.py (validated for CVD separation);
every bar/line is labelled directly because three hues have < 3:1 contrast.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figures  # noqa: F401  (rcParams)
from figures import COL, INK2, M_EMPC, M_MASK, M_PEN, M_SHIELD, M_BLOCK, FIG

RES = Path("results")
C_MASK, C_PEN, C_SHIELD, C_EMPC, C_BLOCK = (COL[M_MASK], COL[M_PEN], COL[M_SHIELD],
                                            COL[M_EMPC], COL[M_BLOCK])
GRID = "#e4e3de"


def _clean(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def _hbar_labels(ax, bars, vals, fmt="{:.1f}", pad=0.4):
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + pad, b.get_y() + b.get_height() / 2, fmt.format(v),
                va="center", ha="left", fontsize=7, color="#0b0b0b")


# ------------------------------------------------------------------ fig 9
def fig9():
    dc = pd.read_csv(RES / "daily_costs.csv")
    g = dc[(dc.sigma == 0.15) & dc.year.isin([2020, 2023])]
    orc = g[g.method.str.startswith("Oracle")].groupby("year").cost.mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1, 1.5]))
    rng = np.random.default_rng(0)
    for i, (m, c, lab) in enumerate(((M_MASK, C_MASK, "Mask"), (M_SHIELD, C_SHIELD, "Shield"),
                                     (M_PEN, C_PEN, "Penalty"))):
        per = g[g.method == m].groupby(["seed", "year"]).cost.mean().unstack()
        gap = (100 * (per / orc - 1)).mean(1).to_numpy()
        x = i + rng.uniform(-0.12, 0.12, len(gap))
        a.scatter(x, gap, s=22, color=c, edgecolor="white", lw=0.6, zorder=3)
        a.hlines(gap.mean(), i - 0.28, i + 0.28, color="#0b0b0b", lw=1.6, zorder=4)
        a.text(i + 0.31, gap.mean(), f"{gap.mean():.1f}", va="center", fontsize=7)
    a.set_xticks(range(3), ["Mask", "Shield", "Penalty"])
    a.set_xlim(-0.5, 2.8)
    a.set_ylabel("gap to oracle, test years (%)")
    a.set_title("(a) one dot per seed, bar = mean", loc="left")
    _clean(a, "y")

    ab = pd.read_csv(RES / "rev_ablation.csv")
    if (RES / "rev_cfga.csv").exists():
        ab = pd.concat([ab, pd.read_csv(RES / "rev_cfga.csv")])
    ab["group"] = ab.run.str.replace(r"_s\d+$", "", regex=True)
    sh = pd.read_csv(RES / "deadline_shift.csv")
    rows = []
    for grp, lab in (("penalty_p150", "Penalty 150 EUR"), ("penalty", "Penalty 500 EUR"),
                     ("penalty_p2000", "Penalty 2000 EUR"), ("penalty_cfgA", "Penalty, own tuning"),
                     ("penalty_noslack", "Penalty, no slack")):
        h = ab[(ab.D == 19.0) & (ab.group == grp)]
        if len(h):
            rows.append((lab, 100 * (h.viol > 0).mean(), C_PEN))
    hm = sh[(sh.deadline == 19.0) & (sh.method == M_MASK)]
    rows.append(("Mask (ours)", 100 * (hm.violations > 0).mean(), C_MASK))
    labs, vals, cols = zip(*rows)
    y = np.arange(len(rows))[::-1]
    bars = b.barh(y, vals, color=cols, height=0.62)
    _hbar_labels(b, bars, vals, "{:.1f}%", pad=0.15)
    b.text(0.9, y[-1], "(guaranteed by Theorem 3)", va="center", fontsize=7, color=INK2)
    b.set_yticks(y, labs)
    b.set_xlabel("seed-days with a late job, $D=19$ h (%)")
    b.set_title("(b) tighter deadline, agents not retrained", loc="left")
    b.set_xlim(0, max(vals) * 1.25 + 0.5)
    _clean(b)
    fig.tight_layout()
    fig.savefig(FIG / "fig9_deploy.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ fig 10
def fig10():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    js = json.loads((RES / "forecast_rev.json").read_text())
    sc = [("ar1_rho0.3", "AR(1), rho 0.3"), ("ar1_rho0.7", "AR(1), rho 0.7"),
          ("ar1_rho0.95", "AR(1), rho 0.95"), ("ar1_rho0.7+intraday", "AR(1) + intraday"),
          ("real", "Real TSO errors"), ("real+intraday", "Real + intraday")]
    meth = [(M_MASK, C_MASK, "Mask", "o"), (M_PEN, C_PEN, "Penalty", "s"),
            (M_SHIELD, C_SHIELD, "Shield", "D"), ("EMPC", C_EMPC, "EMPC", "^")]
    y = np.arange(len(sc))[::-1]
    for m, c, lab, mk in meth:
        v = [np.mean([js[f"{k}|{yy}"][m]["gap_pct"] for yy in (2020, 2023)]) for k, _ in sc]
        a.scatter(v, y, color=c, marker=mk, s=30, edgecolor="white", lw=0.6, label=lab, zorder=3)
    a.set_yticks(y, [t for _, t in sc])
    a.set_xlabel("gap to oracle, mean of test years (%)")
    a.set_title("(a) forecast error model", loc="left")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, handletextpad=0.2,
             columnspacing=0.8)
    a.axhspan(-0.5, 1.5, color="#f1f0ec", zorder=0)
    _clean(a)

    df = pd.DataFrame(json.loads((RES / "robust_ext.json").read_text()))
    for D, ls in ((23.0, "-"), (19.0, "--")):
        for meth_, c, lab in (("Masked agent (robust mask)", C_MASK, "Mask"),
                              ("EMPC (robust model)", C_EMPC, "EMPC")):
            q = df[(df["mode"] == "random") & (df.noise == 0) & (df.D == D) & (df.method == meth_)].sort_values("kappa")
            b.plot(q.kappa, q.late_pct, ls=ls, color=c, marker="o", ms=4,
                   label=f"{lab}, D={D:.0f} h")
    b.axvline(1.0, color=INK2, lw=0.6, ls=":")
    b.text(1.03, 0.5 * b.get_ylim()[1], "assumed\nbound", fontsize=7, color=INK2, va="center")
    b.set_xlabel("delays up to kappa x assumed bound")
    b.set_ylabel("days with a late job (%)")
    b.set_title("(b) beyond the bound (robust mask)", loc="left")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7)
    _clean(b, "y")
    fig.tight_layout()
    fig.savefig(FIG / "fig10_robust.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ fig 11
def fig11():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
    pa = pd.DataFrame(json.loads((RES / "policy_agnostic.json").read_text()))
    pols = [("stale plan", "Stale plan"), ("threshold", "Price threshold"),
            ("RL agent", "RL policy"), ("random", "Random"), ("EMPC", "EMPC")]
    y = np.arange(len(pols))[::-1]
    off = 0.2
    v0 = [float(pa[(pa.D == 19.0) & (pa.policy == p) & (~pa.masked)].late_pct.iloc[0]) for p, _ in pols]
    v1 = [float(pa[(pa.D == 19.0) & (pa.policy == p) & (pa.masked)].late_pct.iloc[0]) for p, _ in pols]
    b0 = a.barh(y + off, v0, height=0.36, color="#b9b8b2", label="without mask")
    b1 = a.barh(y - off, v1, height=0.36, color=C_MASK, label="with mask")
    _hbar_labels(a, b0, v0, "{:.1f}%", pad=0.3)
    _hbar_labels(a, b1, v1, "{:.1f}%", pad=0.3)
    a.set_yticks(y, [t for _, t in pols])
    a.set_xlabel("days with a late job, $D=19$ h (%)")
    a.set_title("(a) one mask, five policies", loc="left")
    a.set_xlim(0, max(v0) * 1.3)
    a.legend(loc="lower right")
    _clean(a)

    js = json.loads((RES / "costmodel.json").read_text())
    tr = json.loads((RES / "tariff_retrain.json").read_text()) if (RES / "tariff_retrain.json").exists() else {}
    sc = [("E", "Energy\nonly"), ("N", "+network\ncharge"), ("ND", "+demand\ncharge"),
          ("NDX", "+export"), ("B", "energy +\nbattery"), ("ALL", "all +\nbattery")]
    series = [(M_MASK, C_MASK, "Mask, trained on energy cost"),
              ("EMPC (energy-only model)", "#b9b8b2", "EMPC, energy-only model"),
              ("EMPC", C_EMPC, "EMPC, actual tariff")]
    x = np.arange(len(sc))
    w = 0.22
    for i, (m, c, lab) in enumerate(series):
        v = [np.mean([js[f"{k}|{yy}"][m]["gap_to_oracle_pct"] for yy in (2020, 2023)]) for k, _ in sc]
        b.bar(x + (i - 1) * w, v, width=w * 0.92, color=c, label=lab)
    if "ALL" in tr:
        vr = np.mean([tr["ALL"]["2020"], tr["ALL"]["2023"]])
        bb = b.bar([x[-1] + 2 * w], [vr], width=w * 0.92, color=C_MASK, hatch="////",
                   edgecolor="white", label="Mask, retrained on full tariff")
        b.text(x[-1] + 2 * w, vr + 1, f"{vr:.0f}", ha="center", fontsize=7)
    b.set_xticks(x, [t for _, t in sc], fontsize=6.3)
    b.set_ylabel("gap to oracle, test years (%)")
    b.set_title("(b) extended tariffs", loc="left")
    b.legend(loc="upper left", fontsize=6.5)
    _clean(b, "y")
    fig.tight_layout()
    fig.savefig(FIG / "fig11_agnostic.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ fig 12
def fig12():
    js = json.loads((RES / "explain_audit.json").read_text())
    e20, e23 = js[f"2020|{M_MASK}"], js[f"2023|{M_MASK}"]
    fid = js["surrogate_fidelity"]
    acc = np.mean([fid[k]["accuracy"] for k in fid if k.endswith("depth3")])
    maj = np.mean([fid[k]["majority"] for k in fid if k.endswith("depth3")])
    items = [("Consistent releases\n(no cheaper option within 1 h)",
              e20["consistent_pct_w1"], e23["consistent_pct_w1"]),
             ("Agent's gap to EMPC located\nby flagged releases (3 h window)",
              e20["share_of_gap_explained_w3"], e23["share_of_gap_explained_w3"]),
             ("Flagged advice also right\non actual PV (ex post)",
              e20["flagged_expost_valid_pct_w1"], e23["flagged_expost_valid_pct_w1"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.7, 1]))
    y = np.arange(len(items))[::-1]
    for j, (yr, c, off) in enumerate(((2020, "#8fb8ea", 0.18), (2023, C_MASK, -0.18))):
        v = [it[1 + j] for it in items]
        bars = a.barh(y + off, v, height=0.34, color=c, label=str(yr))
        _hbar_labels(a, bars, v, "{:.0f}%", pad=1)
    a.set_yticks(y, [it[0] for it in items], fontsize=7)
    a.set_xlim(0, 130)
    a.set_xticks([0, 20, 40, 60, 80, 100])
    a.set_xlabel("%")
    a.set_title("(a) audit of the learned policy", loc="left")
    a.legend(loc="center right")
    _clean(a)
    bars = b.bar([0, 1], [maj, acc], color=["#b9b8b2", C_MASK], width=0.55)
    for bb, v in zip(bars, (maj, acc)):
        b.text(bb.get_x() + bb.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=7)
    b.set_xticks([0, 1], ["majority\naction", "depth-3 tree on\nexplanation terms"], fontsize=7)
    b.set_ylim(50, 100)
    b.set_ylabel("decisions reproduced (%)")
    b.set_title("(b) fidelity, other test year", loc="left")
    _clean(b, "y")
    fig.tight_layout()
    fig.savefig(FIG / "fig12_audit.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ fig 13
def fig13():
    if not (RES / "scale_eval.json").exists():
        return
    js = json.loads((RES / "scale_eval.json").read_text())
    meth = [("Latest start", "Latest start", "#e87ba4"), ("Best block", "Best block", C_BLOCK),
            ("MaskablePPO + mask", "Masked PPO", C_MASK), ("Max-plus EMPC (CP-SAT)", "EMPC (CP-SAT)", C_EMPC)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    x = np.arange(len(meth))
    v = [np.mean([js[y][m]["gap_pct"] for y in ("2020", "2023")]) for m, _, _ in meth]
    bars = a.bar(x, v, color=[c for *_, c in meth], width=0.6)
    for bb, vv in zip(bars, v):
        a.text(bb.get_x() + bb.get_width() / 2, vv + max(v) * 0.02, f"{vv:.1f}", ha="center", fontsize=7)
    a.set_xticks(x, [t for _, t, _ in meth], fontsize=7)
    a.set_ylabel("gap to oracle, test years (%)")
    a.set_title("(a) 5 jobs, 15-min epochs: cost", loc="left")
    _clean(a, "y")
    tm = [("MaskablePPO + mask", "Masked PPO\n(mask + policy)", C_MASK),
          ("Max-plus EMPC (CP-SAT)", "EMPC (CP-SAT)\nper solve", C_EMPC)]
    tv = [1e3 * np.mean([js[y][m]["mean_time_s"] for y in ("2020", "2023")]) for m, _, _ in tm]
    bars = b.bar([0, 1], tv, color=[c for *_, c in tm], width=0.5)
    b.set_yscale("log")
    for bb, vv in zip(bars, tv):
        b.text(bb.get_x() + bb.get_width() / 2, vv * 1.3, f"{vv:.2f} ms" if vv < 10 else f"{vv / 1e3:.2f} s",
               ha="center", fontsize=7)
    b.set_xticks([0, 1], [t for _, t, _ in tm], fontsize=7)
    b.set_ylabel("computation time (ms, log scale)")
    b.set_title("(b) decision time", loc="left")
    b.set_ylim(min(tv) / 3, max(tv) * 6)
    _clean(b, "y")
    fig.tight_layout()
    fig.savefig(FIG / "fig13_scale.pdf")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig9, fig10, fig11, fig12, fig13):
        try:
            f()
            print("ok", f.__name__)
        except Exception as ex:
            print("FAIL", f.__name__, type(ex).__name__, ex)

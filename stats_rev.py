"""
Statistics for the revision (reviewer comments R2-M8, R3-Results, R4-6, R4-7).

For a pair of methods (A, B) evaluated on the same days with common random
numbers, the unit of analysis is the *day* (independent evaluation unit);
RL methods have several seeds per day (seed-days are repeated measurements of
the same day, not independent observations).

Reported per comparison:
  * mean paired difference  Delta = mean_day( cost_A - cost_B )  [EUR/day]
    and relative difference 100 * Delta / mean(cost_B)
  * 95% CI of Delta by a two-level (hierarchical) bootstrap:
      - seeds of each RL method are resampled with replacement
        (keeps the between-seed variance, R2-M8),
      - days are resampled with a circular moving-block bootstrap with
        7-day blocks (keeps the autocorrelation of PV and prices, R3)
  * effect sizes: paired Cohen's d_z = mean(d)/sd(d) on seed-averaged daily
    differences, and the matched-pairs rank-biserial correlation r_rb
  * Wilcoxon signed-rank p (kept for continuity with the first submission)
  * lag-1 autocorrelation of the daily differences (justifies the blocks)
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

BLOCK = 7
N_BOOT = 4000


def _matrix(df, method):
    """days x seeds matrix of daily costs (deterministic methods: 1 column)."""
    h = df[df.method == method]
    return h.pivot_table(index="date", columns="seed", values="cost").sort_index()


def _block_indices(n, rng, block=BLOCK):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, nb)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
    return idx[:n]


def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def compare(df, a, b, n_boot=N_BOOT, seed=0, block=BLOCK):
    """Paired comparison of method a against method b (a - b)."""
    A, B = _matrix(df, a), _matrix(df, b)
    days = A.index.intersection(B.index)
    A, B = A.loc[days].to_numpy(), B.loc[days].to_numpy()
    n = len(days)
    d_day = A.mean(1) - B.mean(1)                      # seed-averaged daily diff
    delta = float(d_day.mean())
    base = float(B.mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        ia = rng.integers(0, A.shape[1], A.shape[1])
        ib = rng.integers(0, B.shape[1], B.shape[1])
        idx = _block_indices(n, rng, block)
        boots[k] = A[idx][:, ia].mean() - B[idx][:, ib].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sd = d_day.std(ddof=1)
    ac1 = float(np.corrcoef(d_day[:-1], d_day[1:])[0, 1]) if n > 2 else np.nan
    p = float(wilcoxon(d_day).pvalue) if (d_day != 0).any() else 1.0
    return dict(a=a, b=b, n_days=int(n), seeds_a=int(A.shape[1]),
                seeds_b=int(B.shape[1]), delta=delta, ci_lo=float(lo),
                ci_hi=float(hi), rel_pct=100 * delta / base,
                rel_ci_lo=100 * lo / base, rel_ci_hi=100 * hi / base,
                cohen_dz=float(delta / sd) if sd > 0 else 0.0,
                r_rb=rank_biserial(d_day), wilcoxon_p=p, lag1_ac=ac1,
                frac_days_a_better=float((d_day < 0).mean()))


def seed_level(df, method, oracle="Oracle (perfect information)"):
    """Gap to the oracle of every seed (annual mean)."""
    o = df[df.method == oracle].cost.mean()
    s = df[df.method == method].groupby("seed").cost.mean()
    return 100 * (s / o - 1)

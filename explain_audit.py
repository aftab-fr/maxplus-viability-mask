"""
Audit of the model-grounded explanations (revision: R2-M7, R2-m3, R3, R4-8).

For every release of the masked agent on the held-out years (sigma = 0.15,
all seeds) and of the EMPC:

  * counterfactual regret within a window of +-w hours (w = 1, 2, 3):
      regret_w = c(t) - min_{|du|<=w, admissible} c(t+du),
    c(.) = re-optimized forecast day cost (all later releases re-optimized);
    the release is "model-consistent at w" if regret_w <= 0 (ties allowed).
  * size of the inconsistencies (how many EUR the flagged decisions lose)
    and which share of the agent's gap to the EMPC they account for;
  * ex-post validity: the same counterfactual recomputed on the *actual* PV;
    share of flagged decisions for which the recommended alternative is also
    cheaper ex post (the explanation's advice would have paid off).

Policy fidelity of the explanation features: a depth-3 decision tree is
fitted to the agent's decisions at every non-forced epoch with the
explanation quantities (slack, hour, remaining jobs, forecast PV now, price
ratio now, price now relative to the cheapest of the next 6 h, forecast PV
share if released now) as features; its accuracy on the other test year is
the fidelity of a model-grounded surrogate of the policy.

Output: results/explain_audit.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from baselines import ScheduleSpace
from data_loader import HourlyDataset
from evaluate import day_forecast, load_model
from explain import best_completion
from factory_env import FactoryMicrogridEnv, grid_cost, T_HORIZON

RES = Path("results")
M_MASK = "MaskablePPO + max-plus mask (proposed)"
M_EMPC = "Max-plus EMPC (forecast)"
SIGMA = 0.15
WINDOWS = (1, 2, 3)


def release_audit(space, sched, solar, price, fc):
    out = []
    for k, t in enumerate(sched):
        prefix = list(sched[:k])
        nxt_lim = None  # later releases are re-optimized, so only u > prev
        c = {}
        ca = {}
        for du in range(-max(WINDOWS), max(WINDOWS) + 1):
            u = t + du
            if prefix and u <= prefix[-1]:
                continue
            if not 0 <= u < T_HORIZON:
                continue
            v = best_completion(space, prefix, u, fc, price)
            if v is not None:
                c[du] = v
                ca[du] = best_completion(space, prefix, u, solar, price)
        rec = dict(k=k, t=t)
        for w in WINDOWS:
            alt = {du: v for du, v in c.items() if abs(du) <= w}
            best_du = min(alt, key=alt.get)
            rec[f"regret{w}"] = c[0] - alt[best_du]
            rec[f"best{w}"] = best_du
            # ex-post: is the recommended alternative also cheaper on actual PV?
            rec[f"expost_ok{w}"] = (ca[best_du] < ca[0] - 1e-9) if best_du != 0 else None
            rec[f"expost_gain{w}"] = ca[0] - ca[best_du]
        out.append(rec)
    return out


def features(env):
    t = env.hour
    pr = env.price
    nxt = pr[t:min(T_HORIZON, t + 6)]
    return dict(hour=t, remaining=env.remaining, slack=env.slack(),
                pv_now=env.forecast[t] / 2000.0,
                price_ratio=pr[t] / env.price_scale,
                price_vs_min6=(pr[t] - nxt.min()) / env.price_scale)


def policy_trace(model, solar, price, fc):
    env = FactoryMicrogridEnv(stochastic=False, feasibility="mask")
    env.set_day(solar, price, fc)
    obs, _ = env.reset()
    rows, done = [], False
    while not done:
        mk = env.action_mask()
        a, _ = model.predict(obs, action_masks=mk, deterministic=True)
        if mk.all() and env.remaining > 0:      # a real choice
            f = features(env)
            f["action"] = int(a)
            rows.append(f)
        obs, _, done, _, _ = env.step(int(a))
    return rows


def main(seeds=(0, 1, 2, 3, 4)):
    df = pd.read_csv(RES / "daily_costs.csv")
    space = ScheduleSpace()
    models = {s: load_model("mask", s) for s in seeds}
    out, traces = {}, {}
    for year in (2020, 2023):
        ds = HourlyDataset(f"data/merged_hourly_{year}.csv", verbose=False)
        idx = {str(d): i for i, d in enumerate(ds.dates)}
        g = df[(df.year == year) & (df.sigma == SIGMA)]
        empc_mean = g[g.method == M_EMPC].cost.mean()
        for m in (M_MASK, M_EMPC):
            recs = []
            h = g[g.method == m]
            if m == M_MASK:
                h = h[h.seed.isin(seeds)]
            for _, r in h.iterrows():
                i = idx[r.date]
                solar, price = (np.asarray(a, float) for a in ds.get_day(ds.dates[i]))
                fc = day_forecast(solar, SIGMA, year, i)
                sched = tuple(int(s) for s in str(r.schedule).split())
                for e in release_audit(space, sched, solar, price, fc):
                    e.update(seed=int(r.seed), date=r.date)
                    recs.append(e)
            R = pd.DataFrame(recs)
            n_days = h.groupby(["seed", "date"]).ngroups
            ent = dict(releases=len(R))
            for w in WINDOWS:
                bad = R[R[f"regret{w}"] > 1e-6]
                ent[f"consistent_pct_w{w}"] = 100 * (1 - len(bad) / len(R))
                ent[f"regret_per_day_w{w}"] = float(R[f"regret{w}"].sum() / n_days)
                if len(bad):
                    ent[f"flagged_median_regret_w{w}"] = float(bad[f"regret{w}"].median())
                    ent[f"flagged_regret_lt_0.5eur_pct_w{w}"] = 100 * float((bad[f"regret{w}"] < 0.5).mean())
                    ent[f"flagged_regret_lt_2eur_pct_w{w}"] = 100 * float((bad[f"regret{w}"] < 2.0).mean())
                    ok = bad[f"expost_ok{w}"].astype(bool)
                    ent[f"flagged_expost_valid_pct_w{w}"] = 100 * float(ok.mean())
                    ent[f"flagged_expost_gain_per_day_w{w}"] = float(bad[f"expost_gain{w}"].sum() / n_days)
            if m == M_MASK:
                gap = g[(g.method == M_MASK) & g.seed.isin(seeds)].cost.mean() - empc_mean
                ent["gap_to_empc_eur_day"] = float(gap)
                for w in WINDOWS:
                    ent[f"share_of_gap_explained_w{w}"] = 100 * ent[f"regret_per_day_w{w}"] / gap
            out[f"{year}|{m}"] = ent
            print(year, m[:20], {k: round(v, 2) for k, v in ent.items()}, flush=True)

        # policy traces for the surrogate
        rows = []
        for s, mdl in models.items():
            for i, d in enumerate(ds.dates):
                solar, price = (np.asarray(a, float) for a in ds.get_day(d))
                fc = day_forecast(solar, SIGMA, year, i)
                for f in policy_trace(mdl, solar, price, fc):
                    f.update(seed=s)
                    rows.append(f)
        traces[year] = pd.DataFrame(rows)

    feats = ["hour", "remaining", "slack", "pv_now", "price_ratio", "price_vs_min6"]
    fid = {}
    for tr, te in ((2020, 2023), (2023, 2020)):
        A, B = traces[tr], traces[te]
        for depth in (2, 3, 4):
            clf = DecisionTreeClassifier(max_depth=depth, random_state=0,
                                         class_weight=None).fit(A[feats], A.action)
            acc = float((clf.predict(B[feats]) == B.action).mean())
            base = float(max(B.action.mean(), 1 - B.action.mean()))
            fid[f"train{tr}_test{te}_depth{depth}"] = dict(accuracy=100 * acc,
                                                           majority=100 * base,
                                                           n=len(B))
            if depth == 3 and tr == 2020:
                fid["tree_depth3_text"] = export_text(clf, feature_names=feats)
    out["surrogate_fidelity"] = fid
    print(json.dumps(fid, indent=1))
    (RES / "explain_audit.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()

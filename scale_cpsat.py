"""
Exact max-plus EMPC for the larger instance with CP-SAT (OR-Tools).

The max-plus recursion x_i(k) = p_i + max(x_i(k-1), x_{i-1}(k)) is imposed
exactly with max-equality constraints (no idle time can be inserted, as in
the plant), machine occupancy by time-indexed start indicators, and the grid
import by import = max(0, load - PV). Times are in epochs of dt hours, energy
in 0.5 kWh units, prices in 1e-4 EUR/kWh, so the model is integer and exact
up to this rounding.

  solve(forecast_h, price_h, N, dt, D, released=(), t0=0, x0=None)
returns the optimal release epochs (fixed prefix `released`), the solve time
and the solver status.
"""

import time

import numpy as np
from ortools.sat.python import cp_model

from factory_env import MACHINE_KW, PROC_H


def solve(pv_h, price_h, N=5, dt=0.25, D=23.0, released=(), t0=0,
          time_limit=30.0, workers=1):
    q = int(round(1 / dt))
    T = 24 * q
    P = [int(round(p / dt)) for p in PROC_H]
    W = [int(round(2 * kw * dt)) for kw in MACHINE_KW]          # 0.5 kWh / epoch
    pv = np.repeat(np.round(2 * np.asarray(pv_h) * dt).astype(int), q)
    pr = np.repeat(np.round(np.asarray(price_h) * 1e4).astype(int), q)
    Dq = int(np.floor(D / dt + 1e-9))
    H = T + sum(P) * N + 1

    m = cp_model.CpModel()
    r = []
    for k in range(N):
        if k < len(released):
            r.append(m.NewConstant(int(released[k])))
        else:
            lo = t0 if k == len(released) else 0
            r.append(m.NewIntVar(lo, T - 1, f"r{k}"))
        if k:
            m.Add(r[k] >= r[k - 1] + 1)
    c = [[None] * N for _ in range(3)]
    for k in range(N):
        for i in range(3):
            up = r[k] if i == 0 else c[i - 1][k]
            terms = [up] + ([c[i][k - 1]] if k else [])
            mx = m.NewIntVar(0, H, f"m{i}{k}")
            m.AddMaxEquality(mx, terms)
            c[i][k] = m.NewIntVar(0, H, f"c{i}{k}")
            m.Add(c[i][k] == mx + P[i])
    m.Add(c[2][N - 1] <= Dq)
    # start indicators
    busy = [[] for _ in range(T)]
    for i in range(3):
        for k in range(N):
            xs = []
            for s in range(0, Dq - P[i] + 1):
                b = m.NewBoolVar(f"x{i}{k}{s}")
                xs.append((s, b))
                for tt in range(s, min(s + P[i], T)):
                    busy[tt].append((W[i], b))
            m.AddExactlyOne([b for _, b in xs])
            m.Add(sum(s * b for s, b in xs) == c[i][k] - P[i])
    obj = []
    for tt in range(T):
        if not busy[tt]:
            if pr[tt] < 0:
                pass
            continue
        L = sum(w * b for w, b in busy[tt])
        Lmax = sum(w for w, _ in busy[tt])
        diff = m.NewIntVar(-int(pv[tt]), Lmax, f"d{tt}")
        m.Add(diff == L - int(pv[tt]))
        imp = m.NewIntVar(0, Lmax, f"i{tt}")
        m.AddMaxEquality(imp, [diff, 0])
        obj.append(int(pr[tt]) * imp)
    m.Minimize(sum(obj))
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = time_limit
    sol.parameters.num_workers = workers
    t1 = time.perf_counter()
    st = sol.Solve(m)
    el = time.perf_counter() - t1
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, el, sol.StatusName(st)
    return tuple(int(sol.Value(v)) for v in r), el, sol.StatusName(st)


def empc_receding(fc_h, price_h, solar_h, N, dt, D=23.0, update=None, **kw):
    """Receding-horizon EMPC (re-solve every epoch while jobs remain).
    Without forecast updates it equals the open-loop optimum."""
    T = int(round(24 / dt))
    rel, times = [], []
    plan = None
    for t in range(T):
        if len(rel) == N:
            break
        f = update(fc_h, solar_h, t) if update else fc_h
        if plan is None or update is not None:
            plan, el, st = solve(f, price_h, N, dt, D, tuple(rel), t, **kw)
            times.append(el)
        if plan[len(rel)] == t:
            rel.append(t)
    return tuple(rel), times

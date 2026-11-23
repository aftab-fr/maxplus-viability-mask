"""
Non-learning schedulers used as benchmarks in the paper.

All of them are evaluated with the same max-plus plant model and the same
cost function as the RL environment (`factory_env.load_profile`,
`factory_env.grid_cost`), so the numbers are directly comparable.

    fixed(...)   conventional fixed-shift release rules
    block(...)   best block of consecutive releases w.r.t. the PV forecast
    empc(...)    receding-horizon economic MPC on the max-plus model
                 (exhaustive search over the feasible release set, forecast PV)
    oracle(...)  perfect-information optimum (actual PV) -> lower bound
"""

import itertools
import time

import numpy as np

from factory_env import (T_HORIZON, grid_cost, load_profile, make_line)


class ScheduleSpace:
    """All feasible release sequences (one release per hourly epoch) and
    their hourly load profiles, pre-computed once with the max-plus model."""

    def __init__(self, n_jobs=3, deadline=23.0):
        self.line = make_line()
        self.n_jobs = n_jobs
        self.deadline = deadline
        scheds, loads = [], []
        for r in itertools.combinations(range(T_HORIZON), n_jobs):
            L, comp = load_profile(self.line, r)
            if max(comp) <= deadline + 1e-9:
                scheds.append(r)
                loads.append(L)
        self.S = np.array(scheds, dtype=int)          # (n_feasible, N)
        self.L = np.array(loads)                      # (n_feasible, 24)
        self.index = {tuple(s): i for i, s in enumerate(scheds)}

    def __len__(self):
        return len(self.S)

    def cost(self, sched, solar, price):
        return float(grid_cost(self.L[self.index[tuple(sched)]], solar, price))

    # ---- benchmarks ------------------------------------------------------
    def oracle(self, solar, price):
        c = grid_cost(self.L, solar, price)
        i = int(np.argmin(c))
        return float(c[i]), tuple(self.S[i])

    def open_loop(self, forecast, price):
        """Certainty-equivalent optimum on the forecast (one-shot)."""
        c = grid_cost(self.L, forecast, price)
        return tuple(self.S[int(np.argmin(c))])

    def empc(self, forecast, price, solar, return_timing=False):
        """Receding-horizon economic MPC.

        At every epoch t the controller enumerates all feasible completions
        of the current (partially released) schedule, evaluates them on the
        forecast, and applies only the first decision (release now or wait).
        """
        released = []
        times = []
        c_fore = grid_cost(self.L, forecast, price)
        for t in range(T_HORIZON):
            t0 = time.perf_counter()
            k = len(released)
            if k == self.n_jobs:
                times.append(time.perf_counter() - t0)
                continue
            ok = np.ones(len(self.S), dtype=bool)
            if k:
                ok &= (self.S[:, :k] == np.array(released)).all(axis=1)
            ok &= self.S[:, k] >= t
            i = np.flatnonzero(ok)[np.argmin(c_fore[ok])]
            if self.S[i, k] == t:
                released.append(t)
            times.append(time.perf_counter() - t0)
        sched = tuple(released)
        out = (self.cost(sched, solar, price), sched)
        return out + (np.array(times),) if return_timing else out

    def block(self, forecast, price, solar):
        """Best start hour for back-to-back releases (s, s+1, ..., s+N-1)."""
        best, best_s = np.inf, None
        for s in range(T_HORIZON):
            r = tuple(range(s, s + self.n_jobs))
            if r not in self.index:
                continue
            c = self.cost(r, forecast, price)
            if c < best:
                best, best_s = c, r
        return self.cost(best_s, solar, price), best_s

    def fixed(self, releases, solar, price):
        r = tuple(releases)
        if r not in self.index:
            raise ValueError(f"fixed schedule {r} is infeasible")
        return self.cost(r, solar, price), r


FIXED_RULES = {
    "Day shift (08-10 h)": (8, 9, 10),
    "Latest start (11-13 h)": (11, 12, 13),
}


# ------------------------------------------------------------------------
#  EMPC with measured state and an arbitrary planning model (used in the
#  processing-delay experiment). Vectorised over all candidate completions.
# ------------------------------------------------------------------------
from maxplus import EPS, SerialLine  # noqa: E402
from factory_env import MACHINE_KW, PROC_H  # noqa: E402

_COMBOS = {}


def _combos(t, m):
    key = (t, m)
    if key not in _COMBOS:
        _COMBOS[key] = np.array(list(itertools.combinations(range(t, T_HORIZON), m)),
                                dtype=float).reshape(-1, m)
    return _COMBOS[key]


def _vec_simulate(x0, R, line):
    """Hourly loads (K,24) and last completion (K,) of release sets R (K,m)."""
    K, m = R.shape
    x = np.tile(x0, (K, 1))
    L = np.zeros((K, T_HORIZON))
    hours = np.arange(T_HORIZON)
    for j in range(m):
        xn = np.max(line.A[None, :, :] + x[:, None, :], axis=2)
        xn = np.maximum(xn, line.B[None, :, 0] + R[:, j:j + 1])
        s, e = xn - line.proc[None, :], xn
        ov = np.clip(np.minimum(e[:, :, None], hours + 1) -
                     np.maximum(s[:, :, None], hours), 0.0, None)
        L += (ov * MACHINE_KW[None, :, None]).sum(axis=1)
        x = xn
    return L, x[:, -1]


def empc_measured(forecast, price, solar, delays, plan_proc, deadline=23.0,
                  n_jobs=3):
    """Receding-horizon EMPC that re-plans every hour from the measured line
    state with the planning model `plan_proc` (nominal or upper-bound times);
    the plant uses the actual times PROC_H + delays[k].

    Returns (actual cost, schedule, late jobs, epochs without a feasible plan).
    """
    plan = SerialLine(plan_proc)
    x = np.full(plan.n, EPS)
    L_act = np.zeros(T_HORIZON)
    released, late, no_plan = [], 0, 0
    for t in range(T_HORIZON):
        m = n_jobs - len(released)
        if m == 0:
            break
        R = _combos(t, m)
        Lp, y = _vec_simulate(x, R, plan)
        ok = y <= deadline + 1e-9
        if ok.any():
            c = grid_cost(L_act[None, :] + Lp, forecast, price)
            c[~ok] = np.inf
            release_now = R[int(np.argmin(c)), 0] == t
        else:
            no_plan += 1
            release_now = True                       # best effort: release ASAP
        if release_now:
            act = SerialLine(np.array(PROC_H) + delays[len(released)])
            La, ya = _vec_simulate(x, np.array([[float(t)]]), act)
            L_act += La[0]
            x = act.next_state(x, float(t))
            late += int(ya[0] > deadline + 1e-9)
            released.append(t)
    late += n_jobs - len(released)
    return float(grid_cost(L_act, solar, price)), tuple(released), late, no_plan

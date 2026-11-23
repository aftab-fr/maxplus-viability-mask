"""
Decision-level explanations derived from the max-plus model.

For every release decision taken at epoch t the explanation reports

  1. admissibility  - mask status and slack (how many hours the remaining
                      jobs could still have been postponed), Prop. 1;
  2. timing         - the max-plus critical path of the released job
                      (which term of x(k+1) = A x(k) (+) B u attains the max);
  3. energy         - the oven window, PV coverage and grid cost of the job;
  4. contrast       - the counterfactual day cost (forecast PV) of releasing
                      the job one hour earlier / later, with all *later*
                      releases re-optimized on the max-plus model, relative
                      to releasing it now (also re-optimized). A decision is
                      "model-consistent" if no admissible neighbour is cheaper.

All quantities are recomputed from the plant model, so every statement in an
explanation can be checked independently (it is not a saliency estimate).
"""

import numpy as np

from baselines import ScheduleSpace
from factory_env import MACHINE_KW, T_HORIZON, grid_cost

_SPACE = {}


def _space(n_jobs, deadline):
    key = (n_jobs, deadline)
    if key not in _SPACE:
        _SPACE[key] = ScheduleSpace(n_jobs, deadline)
    return _SPACE[key]


def best_completion(space, prefix, u, pv, price):
    """Min forecast day cost over feasible schedules starting with prefix+(u,)."""
    k = len(prefix)
    ok = space.S[:, k] == u
    if k:
        ok &= (space.S[:, :k] == np.array(prefix)).all(axis=1)
    if not ok.any():
        return None
    return float(grid_cost(space.L[ok], pv, price).min())


def _job_load(line, x, u):
    """Hourly load of one job released at u after state x, and the new state."""
    x_new = line.next_state(x, float(u))
    L = np.zeros(T_HORIZON)
    for i, (s, e) in enumerate(line.busy_windows(x, x_new)):
        for t in range(int(np.floor(s)), int(np.ceil(e))):
            if 0 <= t < T_HORIZON:
                L[t] += MACHINE_KW[i] * max(0.0, min(e, t + 1) - max(s, t))
    return L, x_new


def marginal_cost(line, x, base_load, u, pv, price):
    """Extra grid cost caused by adding one job released at u."""
    L, x_new = _job_load(line, x, u)
    return float(grid_cost(base_load + L, pv, price) - grid_cost(base_load, pv, price)), L, x_new


def explain_schedule(env_factory, schedule, solar, price, forecast):
    """Replay `schedule` in a fresh env and explain each release.

    Returns a list of dicts, one per released job.
    """
    env = env_factory()
    env.set_day(solar, price, forecast)
    env.reset()
    line = env.line
    out = []
    for t in range(T_HORIZON):
        if t in schedule:
            m = env.remaining
            slack = env.slack()
            forced = not env.viable_wait()
            x, base = env.x.copy(), env.load.copy()
            space = _space(env.n_jobs, env.deadline)
            prefix = list(schedule[:len(out)])
            cands = {}
            for du in (-1, 0, 1):
                u = t + du
                if prefix and u <= prefix[-1]:
                    continue
                if 0 <= u < T_HORIZON:
                    c = best_completion(space, prefix, u, forecast, price)
                    if c is not None:
                        cands[du] = c
            c_now, L, x_new = marginal_cost(line, x, base, t, forecast, price)
            c_act = marginal_cost(line, x, base, t, solar, price)[0]
            e_job = L.sum()
            pv_share = float(np.minimum(L, np.maximum(0.0, forecast - base)).sum() / e_job)
            oven = (x_new[-1] - line.proc[-1], x_new[-1])
            cp = line.critical_path(x, float(t))
            out.append(dict(
                job=len(out) + 1, t=t, remaining_before=m, slack_h=slack,
                forced=forced, critical_path=cp,
                critical_text=line.critical_path_text(x, float(t)),
                oven_window=oven, completion=float(x_new[-1]),
                energy_kwh=float(e_job), pv_share_forecast=pv_share,
                marginal_cost_forecast=c_now, marginal_cost_actual=c_act,
                contrast={du: c - cands[0] for du, c in cands.items() if du != 0},
                locally_consistent=all(c >= cands[0] - 1e-6 for c in cands.values())))
            env.step(1)
        else:
            env.step(0)
    return out


def render(expl, price_unit="EUR"):
    """Operator-facing text report."""
    lines = []
    for e in expl:
        adm = ("FORCED (waiting would violate the deadline)" if e["forced"]
               else f"optional (slack {e['slack_h']:.0f} h)")
        lines.append(f"Job {e['job']} released at {e['t']:02d}:00 -- {adm}")
        lines += ["   " + s for s in e["critical_text"]]
        lines.append(f"   oven {e['oven_window'][0]:.0f}-{e['oven_window'][1]:.0f} h, "
                     f"PV covers {100 * e['pv_share_forecast']:.0f}% of "
                     f"{e['energy_kwh']:.0f} kWh (forecast), marginal cost "
                     f"{e['marginal_cost_forecast']:.2f} {price_unit}")
        for du, d in sorted(e["contrast"].items()):
            lines.append(f"   releasing {'1 h earlier' if du < 0 else '1 h later'} "
                         f"(later jobs re-optimized): {d:+.2f} {price_unit}")
    return "\n".join(lines)

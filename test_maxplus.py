"""
Sanity checks backing the claims of the paper.

    python -m pytest -q test_maxplus.py      (or:  python test_maxplus.py)

1. The explicit max-plus matrices equal the closed form of Eq. (4).
2. The max-plus recursion reproduces an independent event-by-event simulation.
3. Proposition 1 (soundness): every mask-respecting trajectory releases all
   jobs and meets the deadline -- checked by exhaustive depth-first search.
4. Completeness: the mask excludes no feasible schedule (the set of
   reachable schedules equals the enumerated feasible set).
5. The environment and the enumeration baselines compute identical costs.
"""

import itertools

import numpy as np

from baselines import ScheduleSpace
from factory_env import (FactoryMicrogridEnv, grid_cost, load_profile,
                         make_line, price_profile, solar_profile)
from maxplus import EPS, SerialLine


def test_matrices_closed_form():
    line = make_line()
    p = line.proc
    for i in range(line.n):
        for j in range(line.n):
            want = p[j:i + 1].sum() if j <= i else EPS
            assert line.A[i, j] == want
        assert line.B[i, 0] == p[:i + 1].sum()
    np.testing.assert_array_equal(line.A, [[1, EPS, EPS], [3, 2, EPS], [6, 5, 3]])


def _event_sim(proc, releases):
    """Direct FIFO simulation with unlimited buffers (no max-plus)."""
    n = len(proc)
    free = [-np.inf] * n
    out = []
    for u in releases:
        t = u
        for i in range(n):
            t = max(t, free[i]) + proc[i]
            free[i] = t
        out.append(list(free))
    return np.array(out)


def test_recursion_matches_event_simulation():
    rng = np.random.default_rng(1)
    for _ in range(200):
        proc = rng.integers(1, 5, size=4).astype(float)
        line = SerialLine(proc)
        rel = np.sort(rng.integers(0, 20, size=5)).astype(float)
        x = np.full(line.n, EPS)
        xs = []
        for u in rel:
            x = line.next_state(x, u)
            xs.append(x)
        np.testing.assert_allclose(np.array(xs), _event_sim(proc, rel))


def _all_masked_trajectories(env):
    """DFS over every action sequence admitted by the mask."""
    leaves = []

    def rec(state):
        env.hour, env.released, env.x, env.schedule, env.completions = (
            state[0], state[1], state[2].copy(), list(state[3]), list(state[4]))
        if env.hour == 24:
            leaves.append((tuple(env.schedule), list(env.completions), env.released))
            return
        m = env.action_mask()
        assert m.any(), "empty mask reached (Proposition 1 violated)"
        for a in (0, 1):
            if not m[a]:
                continue
            env.hour, env.released, env.x, env.schedule, env.completions = (
                state[0], state[1], state[2].copy(), list(state[3]), list(state[4]))
            if a == 1:
                u = float(env.hour)
                env.x = env.line.next_state(env.x, u)
                env.released += 1
                env.schedule.append(int(u))
                env.completions.append(float(env.x[-1]))
            env.hour += 1
            rec((env.hour, env.released, env.x.copy(), list(env.schedule),
                 list(env.completions)))

    env.reset(seed=0)
    rec((0, 0, np.full(env.line.n, EPS), [], []))
    return leaves


def check_mask_sound_and_complete():
    env = FactoryMicrogridEnv(feasibility="mask", stochastic=False)
    leaves = _all_masked_trajectories(env)
    space = ScheduleSpace()
    for sched, comp, released in leaves:
        assert released == env.n_jobs                         # all jobs released
        assert max(comp) <= env.deadline + 1e-9               # all on time
    reach = {s for s, _, _ in leaves}
    assert reach == set(map(tuple, space.S))                  # exact feasible set
    assert len(reach) == len(leaves)                          # no duplicates
    return len(leaves)


def test_mask_sound_and_complete():
    assert check_mask_sound_and_complete() == 760


def test_env_cost_equals_enumeration():
    space = ScheduleSpace()
    rng = np.random.default_rng(3)
    env = FactoryMicrogridEnv(feasibility="mask", stochastic=False)
    for _ in range(100):
        solar, price = solar_profile(rng=rng), price_profile(rng=rng)
        sched = tuple(space.S[rng.integers(len(space))])
        env.set_day(solar, price)
        env.reset()
        for t in range(24):
            env.step(1 if t in sched else 0)
        assert tuple(env.schedule) == sched
        assert abs(env.total_cost - space.cost(sched, solar, price)) < 1e-6
        L, _ = load_profile(make_line(), sched)
        assert abs(grid_cost(L, solar, price) - env.total_cost) < 1e-6


def test_shield_never_violates_and_penalty_can():
    rng = np.random.default_rng(5)
    for mode in ("shield", "penalty"):
        env = FactoryMicrogridEnv(feasibility=mode, seed=0)
        viol = 0
        for _ in range(300):
            env.reset()
            done = False
            while not done:
                _, _, done, _, info = env.step(int(rng.random() < 0.15))
            viol += info["ep_violations"]
        if mode == "shield":
            assert viol == 0
        else:
            assert viol > 0


if __name__ == "__main__":
    test_matrices_closed_form()
    test_recursion_matches_event_simulation()
    n = check_mask_sound_and_complete()
    test_env_cost_equals_enumeration()
    test_shield_never_violates_and_penalty_can()
    print(f"all checks passed; {n} mask-admissible trajectories = feasible set")

"""
Scalability of exhaustive EMPC versus the max-plus viability mask.

For N jobs and decision epochs of length dt (T = 24/dt epochs, one release
per epoch, D = 23 h) we report the number of candidate release sets
C(T, N), the number of feasible ones (counted by vectorised max-plus
simulation when C(T, N) <= LIMIT), the time to evaluate the cost of all
feasible schedules for one day (the work of one exhaustive EMPC solve), and
the time of one mask evaluation.

Output: results/scalability.json
"""

import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from factory_env import MACHINE_KW, PROC_H, solar_profile, price_profile
from maxplus import EPS, SerialLine

LIMIT = 13_000_000
D = 23.0


def feasible_loads(N, dt, line, chunk=200_000):
    T = int(round(24 / dt))
    hours = np.arange(24)
    feas, loads = 0, []
    it = itertools.combinations(range(T), N)
    while True:
        block = list(itertools.islice(it, chunk))
        if not block:
            break
        R = np.array(block, dtype=float) * dt
        K = len(R)
        x = np.full((K, line.n), EPS)
        L = np.zeros((K, 24))
        for j in range(N):
            xn = np.max(line.A[None] + x[:, None, :], axis=2)
            xn = np.maximum(xn, line.B[None, :, 0] + R[:, j:j + 1])
            s, e = xn - line.proc[None], xn
            ov = np.clip(np.minimum(e[:, :, None], hours + 1) -
                         np.maximum(s[:, :, None], hours), 0, None)
            L += (ov * MACHINE_KW[None, :, None]).sum(1)
            x = xn
        ok = x[:, -1] <= D + 1e-9
        feas += int(ok.sum())
        loads.append(L[ok].astype(np.float32))
    return feas, np.concatenate(loads)


def main():
    line = SerialLine(PROC_H)
    solar, price = solar_profile(), price_profile()
    out = []
    for dt in (1.0, 0.5, 0.25):
        T = int(round(24 / dt))
        for N in (3, 4, 5, 6):
            bound = math.comb(T, N)
            ent = dict(dt_h=dt, epochs=T, n_jobs=N, candidates=bound)
            # one mask evaluation (two viability tests) at t = 0
            t0 = time.perf_counter()
            for _ in range(200):
                line.viable(np.full(3, EPS), 1.0, N, D, dt)
                line.viable(np.full(3, EPS), 0.0, N, D, dt)
            ent["mask_ms"] = (time.perf_counter() - t0) / 200 * 1e3
            if bound <= LIMIT:
                t0 = time.perf_counter()
                feas, L = feasible_loads(N, dt, line)
                ent["enumeration_s"] = time.perf_counter() - t0
                ent["feasible"] = feas
                t0 = time.perf_counter()
                (np.maximum(0, L - solar) * price).sum(1).min()
                ent["empc_solve_ms"] = (time.perf_counter() - t0) * 1e3
                ent["memory_mb"] = L.nbytes / 1e6
            out.append(ent)
            print(ent, flush=True)
    Path("results/scalability.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

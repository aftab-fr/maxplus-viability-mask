"""
Max-plus linear algebra for discrete-event manufacturing systems.

In max-plus algebra:
    a (+) b  =  max(a, b)      (addition  -> synchronisation: wait for the latest)
    a (x) b  =  a + b          (multiplication -> sequencing: durations add up)
    epsilon  = -inf            (neutral element of (+), i.e. "no connection")
    e        = 0               (neutral element of (x))

A serial line with unlimited intermediate buffers obeys the implicit recursion

    x(k+1) = A0 (x) x(k+1)  (+)  A1 (x) x(k)  (+)  B0 (x) u(k+1)

whose explicit form, obtained with the Kleene star A0* = E (+) A0 (+) A0^2 ...
(finite because A0 is strictly lower triangular), is

    x(k+1) = A (x) x(k)  (+)  B (x) u(k+1),     A = A0* A1,  B = A0* B0,

where
    x(k) : completion times of job k on each machine,
    u(k) : release time of job k into the line (the scheduler's decision).

On top of the plant model this module provides the *viability* test used by
the action mask (Proposition 1 in the paper) and the critical-path trace used
for explanations.
"""

import numpy as np

EPS = -np.inf  # max-plus "zero"
E = 0.0        # max-plus "one"


def mp_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Max-plus matrix product: C[i,j] = max_k ( A[i,k] + B[k,j] )."""
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, "dimension mismatch"
    C = np.full((n, p), EPS)
    for i in range(n):
        for j in range(p):
            C[i, j] = np.max(A[i, :] + B[:, j])
    return C


def mp_matvec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Max-plus matrix-vector product."""
    return np.array([np.max(A[i, :] + x) for i in range(A.shape[0])])


def mp_add(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Max-plus addition = elementwise max."""
    return np.maximum(A, B)


def mp_identity(n: int) -> np.ndarray:
    I = np.full((n, n), EPS)
    np.fill_diagonal(I, E)
    return I


def mp_star(A0: np.ndarray) -> np.ndarray:
    """Kleene star A0* = E (+) A0 (+) A0^2 (+) ... for a nilpotent A0."""
    n = A0.shape[0]
    S, P = mp_identity(n), mp_identity(n)
    for _ in range(n):
        P = mp_matmul(P, A0)
        S = mp_add(S, P)
    return S


class SerialLine:
    """A serial production line with `n` machines and unlimited buffers.

    Machine i takes `proc[i]` hours; transfer from machine i to i+1 takes
    `transfer[i]` hours. Each machine processes one job at a time, jobs are
    processed in release order (FIFO), and a machine draws power only while
    processing.
    """

    def __init__(self, proc, transfer=None):
        self.proc = np.asarray(proc, dtype=float)
        self.n = len(self.proc)
        self.transfer = (np.zeros(self.n - 1) if transfer is None
                         else np.asarray(transfer, dtype=float))
        n = self.n

        # implicit model: x_i(k+1) = p_i + max( x_i(k), x_{i-1}(k+1) + tr_{i-1} )
        A0 = np.full((n, n), EPS)            # same-job coupling (upstream machine)
        A1 = np.full((n, n), EPS)            # previous-job coupling (same machine)
        B0 = np.full((n, 1), EPS)            # release enters machine 0
        for i in range(n):
            A1[i, i] = self.proc[i]
            if i > 0:
                A0[i, i - 1] = self.transfer[i - 1] + self.proc[i]
        B0[0, 0] = self.proc[0]

        S = mp_star(A0)
        self.A0, self.A1, self.B0 = A0, A1, B0
        self.A = mp_matmul(S, A1)
        self.B = mp_matmul(S, B0)
        self.cum = self.B[:, 0].copy()       # release-to-completion time per machine

    # ---- plant dynamics ---------------------------------------------------
    def next_state(self, x: np.ndarray, u: float) -> np.ndarray:
        """x(k+1) = A (x) x(k)  (+)  B (x) u."""
        return mp_add(mp_matvec(self.A, x), (self.B[:, 0] + u))

    def earliest_completion(self, x: np.ndarray, u: float) -> float:
        """Finishing time on the last machine of the next job if released at u."""
        return self.next_state(x, u)[-1]

    def busy_windows(self, x_prev: np.ndarray, x_new: np.ndarray):
        """Processing interval [start, end) of the new job on every machine."""
        return [(x_new[i] - self.proc[i], x_new[i]) for i in range(self.n)]

    # ---- viability (used by the action mask) ------------------------------
    def completion_of_sequence(self, x: np.ndarray, releases) -> float:
        """Last-machine completion time of the last job of a release sequence."""
        y = x
        for u in releases:
            y = self.next_state(y, float(u))
        return y[-1]

    def viable(self, x: np.ndarray, t: float, m: int, deadline: float,
               dt: float = 1.0) -> bool:
        """V(x, t, m): can the m remaining jobs still all meet the deadline if
        they are released at the earliest admissible epochs t, t+dt, ...?

        Because the max-plus map is monotone in u, the earliest release
        sequence minimises every completion time, so V is an exact test
        (necessary and sufficient) for the existence of a feasible completion.
        """
        if m <= 0:
            return True
        return self.completion_of_sequence(
            x, [t + j * dt for j in range(m)]) <= deadline + 1e-9

    def latest_viable_epoch(self, x: np.ndarray, t: float, m: int,
                            deadline: float, horizon: float,
                            dt: float = 1.0) -> float:
        """Largest epoch t' >= t at which V(x, t', m) still holds (slack)."""
        if m <= 0:
            return horizon
        if not self.viable(x, t, m, deadline, dt):
            return t - dt
        tt = t
        while tt + dt <= horizon and self.viable(x, tt + dt, m, deadline, dt):
            tt += dt
        return tt

    # ---- explainability ---------------------------------------------------
    def critical_path(self, x: np.ndarray, u: float):
        """For each machine, which max-plus term attains the maximum in
        x_i(k+1) = max( max_j A_ij + x_j(k),  B_i + u ).

        Returns a list of dicts (machine, completion, driver, source, bound).
        driver is 'release' or 'previous job'; source is the machine index j*
        of the dominant previous-job term.
        """
        x_new = self.next_state(x, u)
        out = []
        for i in range(self.n):
            release_bound = self.B[i, 0] + u
            prev_bounds = self.A[i, :] + x
            j = int(np.argmax(prev_bounds))
            if release_bound >= prev_bounds[j]:
                out.append(dict(machine=i, completion=float(x_new[i]),
                                driver="release", source=None,
                                bound=float(release_bound)))
            else:
                out.append(dict(machine=i, completion=float(x_new[i]),
                                driver="previous job", source=j,
                                bound=float(prev_bounds[j])))
        return out

    def critical_path_text(self, x: np.ndarray, u: float):
        """Human-readable version of `critical_path`."""
        lines = []
        for c in self.critical_path(x, u):
            i = c["machine"]
            if c["driver"] == "release":
                lines.append(f"M{i}: completes {c['completion']:.1f} h, driven by "
                             f"RELEASE at u={u:.1f} h (+{self.B[i, 0]:.1f} h)")
            else:
                j = c["source"]
                lines.append(f"M{i}: completes {c['completion']:.1f} h, driven by "
                             f"PREVIOUS JOB on M{j} (x_{j}={x[j]:.1f} h, "
                             f"+A[{i},{j}]={self.A[i, j]:.1f} h)")
        return lines

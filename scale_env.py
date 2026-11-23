"""
Larger instance (revision: R2-M1, R4-3, R4-10): N jobs, epochs of dt hours.

Same serial line, PV array, price data and cost as the case study; hourly PV
and prices are held constant within the hour. Decision every dt hours
(release the next job now or wait); the viability mask is the one of
Theorem 3 with epoch length dt (Algorithm 1 with tau + j*dt).

Observation (30 entries, as in the case study): t/T, m/N, slack/T,
backlogs/6 h, forecast PV and price ratio for the next 12 hours sampled every
hour from the current epoch.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from factory_env import MACHINE_KW, P_PEAK_KW, PROC_H, solar_forecast
from maxplus import EPS, SerialLine


class ScaleEnv(gym.Env):
    def __init__(self, n_jobs=5, dt=0.25, deadline=23.0, dataset=None, seed=None,
                 forecast_sigma=0.0, forecast_sigma_range=None, reward_scale=100.0):
        super().__init__()
        self.N, self.dt, self.D = n_jobs, dt, float(deadline)
        self.T = int(round(24 / dt))
        self.q = int(round(1 / dt))                      # epochs per hour
        self.line = SerialLine(list(PROC_H))
        self.dataset = dataset
        self.sigma, self.sigma_range = forecast_sigma, forecast_sigma_range
        self.reward_scale = reward_scale
        self._rng = np.random.default_rng(seed)
        self._fixed = None
        self.observation_space = spaces.Box(-3.0, 3.0, (30,), np.float32)
        self.action_space = spaces.Discrete(2)

    def set_day(self, solar, price, forecast):
        self._fixed = tuple(np.asarray(a, float) for a in (solar, price, forecast))

    # hourly -> epoch arrays
    def _rep(self, a):
        return np.repeat(np.asarray(a, float), self.q)

    def viable(self, tau_epoch, m):
        return self.line.viable(self.x, tau_epoch * self.dt, m, self.D, self.dt)

    def action_mask(self):
        m = self.N - self.released
        w = self.viable(self.t + 1, m) and not (m > 0 and self.t + 1 >= self.T)
        r = m > 0 and self.viable(self.t, m)
        return np.array([w, r], bool)

    def slack(self):
        m = self.N - self.released
        if m == 0:
            return (self.T - self.t) * self.dt
        tt = self.t
        if not self.viable(tt, m):
            return -self.dt
        while tt + 1 < self.T and self.viable(tt + 1, m):
            tt += 1
        return (tt - self.t) * self.dt

    def _obs(self):
        h0 = self.t * self.dt
        idx = [int(h0) + h for h in range(12)]
        pv = [self.fc_h[i] / P_PEAK_KW if i < 24 else 0.0 for i in idx]
        pr = [self.price_h[i] / self.pscale if i < 24 else 0.0 for i in idx]
        bl = [max(0.0, xi - h0) / 6.0 if np.isfinite(xi) else 0.0 for xi in self.x]
        o = np.concatenate(([self.t / self.T, (self.N - self.released) / self.N,
                             max(self.slack(), -1.0) / 24.0], bl, pv, pr))
        return np.clip(o, -3, 3).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self._rng
        if self._fixed is not None:
            s, p, f = self._fixed
        else:
            s, p = (np.asarray(a, float) for a in self.dataset.random_day(rng))
            sig = rng.uniform(*self.sigma_range) if self.sigma_range else self.sigma
            f = solar_forecast(s, sig, rng)
        self.solar_h, self.price_h, self.fc_h = s, p, f
        self.solar, self.price = self._rep(s), self._rep(p)
        self.pscale = max(float(np.mean(np.abs(p))), 1e-3)
        self.t, self.released = 0, 0
        self.x = np.full(3, EPS)
        self.load = np.zeros(self.T)                    # kWh per epoch
        self.cost, self.sched = 0.0, []
        return self._obs(), {}

    def _release(self):
        u = self.t * self.dt
        xn = self.line.next_state(self.x, u)
        for i in range(3):
            s, e = xn[i] - PROC_H[i], xn[i]
            a, b = int(round(s / self.dt)), int(round(e / self.dt))
            self.load[a:min(b, self.T)] += MACHINE_KW[i] * self.dt
        self.x = xn
        self.released += 1
        self.sched.append(self.t)

    def step(self, a):
        mk = self.action_mask()
        if not mk[int(a)]:
            raise RuntimeError("masked action executed")
        if int(a) == 1:
            self._release()
        t = self.t
        c = max(0.0, self.load[t] - self.solar[t] * self.dt) * self.price[t]
        self.cost += c
        self.t += 1
        done = self.t >= self.T
        obs = self._obs() if not done else np.zeros(30, np.float32)
        return obs, -c / self.reward_scale, done, False, {}


def schedule_cost(sched, solar_h, price_h, n_jobs, dt, deadline=23.0):
    """Cost and lateness of a release schedule (epoch indices) on actual PV."""
    env = ScaleEnv(n_jobs, dt, deadline)
    env.set_day(solar_h, price_h, solar_h)
    env.reset()
    line, x = env.line, np.full(3, EPS)
    T = env.T
    load = np.zeros(T)
    for k in sched:
        xn = line.next_state(x, k * dt)
        for i in range(3):
            a, b = int(round((xn[i] - PROC_H[i]) / dt)), int(round(xn[i] / dt))
            load[a:min(b, T)] += MACHINE_KW[i] * dt
        x = xn
    sol, pr = np.repeat(solar_h, env.q), np.repeat(price_h, env.q)
    return float((np.maximum(0, load - sol * dt) * pr).sum()), float(x[-1])

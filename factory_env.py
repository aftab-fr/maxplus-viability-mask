"""
Gymnasium environment: day-ahead job-release scheduling in a PV factory microgrid.

One episode = one day, decision epochs t = 0..23 (hourly).
N jobs must be released into a 3-machine serial line
(prep -> treatment -> drying oven) and leave the last machine by the
shipping deadline D. Each machine draws constant power while processing.

Action at every epoch:  0 = wait,  1 = release the next job now.

Feasibility handling (`feasibility` argument) -- the ablation of the paper:
    "mask"    max-plus viability mask exposed to the policy (MaskablePPO).
              By Proposition 1 every trajectory is feasible; no correction.
    "shield"  the policy sees no mask; an external shield overrides any
              non-viable action after the policy has chosen it (post-hoc
              correction). Interventions are counted.
    "penalty" no mask, no correction: every requested release is executed;
              late or unreleased jobs incur a penalty (reward shaping only).
              Deadline violations are counted.

Reward per hour: -(grid energy [kWh] * day-ahead price [EUR/kWh]) / reward_scale.
PV covers the load first; surplus PV is not remunerated (self-consumption).

Information structure: day-ahead prices for the whole day are known at t=0
(they are cleared the day before). PV output is only known through a
day-ahead forecast with multiplicative AR(1) error of level `forecast_sigma`;
the reward is always computed on the *actual* PV output.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from maxplus import SerialLine, EPS

T_HORIZON = 24
PROC_H = (1.0, 2.0, 3.0)                 # processing hours per machine
MACHINE_KW = np.array([150.0, 250.0, 500.0])
P_PEAK_KW = 2000.0
# extended tariff (revision): network energy charge, demand charge, battery
NET_TAU = np.where((np.arange(24) >= 7) & (np.arange(24) < 23), 0.030, 0.015)
DEM_LAMBDA = 0.40
BAT_E, BAT_P, BAT_ETA = 1000.0, 500.0, 0.9 ** 0.5


# ------------------------------------------------------------ plant helpers --
def make_line():
    return SerialLine(proc=list(PROC_H), transfer=[0.0, 0.0])


def load_profile(line, releases, kw=MACHINE_KW, T=T_HORIZON):
    """Hourly energy drawn [kWh per hour slot] by a release sequence.

    Uses exactly the max-plus recursion of the environment, so costs from
    enumeration-based baselines and from the RL environment are identical.
    Also returns the last-machine completion time of every job.
    """
    x = np.full(line.n, EPS)
    L = np.zeros(T)
    completions = []
    for u in releases:
        x_new = line.next_state(x, float(u))
        for i, (s, e) in enumerate(line.busy_windows(x, x_new)):
            for t in range(int(np.floor(s)), int(np.ceil(e))):
                if 0 <= t < T:
                    L[t] += kw[i] * max(0.0, min(e, t + 1) - max(s, t))
        completions.append(float(x_new[-1]))
        x = x_new
    return L, completions


def grid_cost(L, solar, price):
    """Daily grid cost [EUR] of an hourly load profile (vectorised over rows)."""
    return (np.maximum(0.0, L - solar) * price).sum(axis=-1)


def solar_forecast(solar, sigma, rng, rho=0.7):
    """Day-ahead PV forecast with AR(1) multiplicative error (std `sigma`)."""
    if sigma <= 0 or rng is None:
        return np.asarray(solar, dtype=np.float64).copy()
    e = np.zeros(len(solar))
    e[0] = rng.normal(0.0, sigma)
    for t in range(1, len(solar)):
        e[t] = rho * e[t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0.0, sigma)
    return np.clip(solar * (1.0 + e), 0.0, P_PEAK_KW)


# ------------------------------------------------------------- synthetic data --
def solar_profile(peak_kw=1800.0, rng=None, noise=0.10):
    """Synthetic clear-sky bell curve (for unit tests without real data)."""
    t = np.arange(24)
    p = peak_kw * np.exp(-0.5 * ((t - 13.0) / 3.0) ** 2)
    p[(t < 6) | (t > 20)] = 0.0
    if rng is not None:
        p = p * (1 + rng.normal(0, noise, 24)).clip(0.3, 1.3)
    return p


def price_profile(rng=None, noise=0.08):
    """Synthetic duck-curve price in EUR/kWh (for unit tests only)."""
    base = np.array([
        .085, .080, .078, .075, .078, .090, .110, .130,
        .120, .095, .070, .055, .045, .040, .045, .055,
        .075, .100, .135, .150, .140, .120, .100, .090,
    ])
    if rng is not None:
        base = base * (1 + rng.normal(0, noise, 24)).clip(0.5, 1.5)
    return base


# ------------------------------------------------------------- environment --
class FactoryMicrogridEnv(gym.Env):
    metadata = {"render_modes": []}
    MACHINE_KW = MACHINE_KW
    LOOKAHEAD = 12

    def __init__(self, n_jobs=3, deadline=23.0, stochastic=True, seed=None,
                 dataset=None, feasibility="mask", forecast_sigma=0.0,
                 forecast_sigma_range=None, reward_scale=100.0,
                 violation_penalty=500.0, proc_delay_max=None,
                 mask_model="nominal", observe_slack=True,
                 meas_noise_max=0.0, meas_margin=0.0, tariff=None):
        """
        dataset : HourlyDataset | None
            Real days (PVGIS PV + day-ahead prices). None -> synthetic curves.
        forecast_sigma_range : (lo, hi) | None
            If given, sigma is drawn uniformly per episode (training).
        violation_penalty : EUR per late / unreleased job ("penalty" mode).
        proc_delay_max : per-machine upper bound of a random processing delay
            [h]; the actual time of job k on machine i is p_i + U[0, d_i].
        mask_model : "nominal" -> mask/slack use the nominal times p;
            "robust" -> they use the upper bounds p + d (Corollary 1).
        """
        super().__init__()
        assert feasibility in ("mask", "shield", "penalty")
        assert mask_model in ("nominal", "robust")
        self.proc_delay_max = (np.zeros(len(PROC_H)) if proc_delay_max is None
                               else np.asarray(proc_delay_max, float))
        self.mask_model = mask_model
        # False -> the slack entry of the observation is zeroed (model-free
        # unmasked baselines; revision experiment R2-m2)
        self.observe_slack = observe_slack
        # measurement error of the completion times used by the mask:
        # x_meas = x_true + U[-meas_noise_max, meas_noise_max] (finite entries);
        # meas_margin is added to x_meas before the viability test (revision
        # experiment R4-5; margin >= noise bound restores Corollary 4)
        # extended tariff (revision R4-9): None = energy only (paper), or a
        # dict with keys net (bool), dem (bool), exp (bool), bat (bool);
        # values as in costmodel_rev.py
        self.tariff = tariff
        self.meas_noise_max = float(meas_noise_max)
        self.meas_margin = float(meas_margin)
        self.plant_nominal = make_line()
        # planning model used by the mask, the slack and the observation
        self.line = (SerialLine(np.array(PROC_H) + self.proc_delay_max)
                     if mask_model == "robust" else make_line())
        self.n_jobs = n_jobs
        self.deadline = float(deadline)
        self.stochastic = stochastic
        self.dataset = dataset
        self.feasibility = feasibility
        self.forecast_sigma = forecast_sigma
        self.forecast_sigma_range = forecast_sigma_range
        self.reward_scale = reward_scale
        self.violation_penalty = violation_penalty
        self._rng = np.random.default_rng(seed)
        self._fixed_day = None
        self._fixed_delays = None

        H = self.LOOKAHEAD
        # [t/24, remaining/N, slack/24, backlog(3), PV forecast(H), price ratio(H)]
        self.obs_dim = 6 + 2 * H + (2 if tariff else 0)
        self.observation_space = spaces.Box(-3.0, 3.0, shape=(self.obs_dim,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        # cumulative counters (survive resets; read by training callbacks)
        self.total_interventions = 0
        self.total_violations = 0
        self.total_episodes = 0

    # ---- day selection ------------------------------------------------------
    def set_day(self, solar, price, forecast=None, delays=None):
        """Fix the next episode(s) to a given day (evaluation).
        delays : (n_jobs, n_machines) actual processing delays [h] or None."""
        self._fixed_day = (np.asarray(solar, float), np.asarray(price, float),
                           None if forecast is None else np.asarray(forecast, float))
        self._fixed_delays = None if delays is None else np.asarray(delays, float)

    # ---- max-plus quantities -----------------------------------------------
    @property
    def remaining(self):
        return self.n_jobs - self.released

    @property
    def x_mask(self):
        """Completion vector seen by the mask (measured, plus margin)."""
        if self.meas_noise_max <= 0 and self.meas_margin <= 0:
            return self.x
        if getattr(self, "_xm_key", None) != (self.hour, self.released):
            noise = self._rng.uniform(-self.meas_noise_max, self.meas_noise_max,
                                      len(self.x)) if self.meas_noise_max > 0 else 0.0
            self._xm = self.x + noise + self.meas_margin
            self._xm_key = (self.hour, self.released)
        return self._xm

    def viable_release(self):
        return self.remaining > 0 and self.line.viable(
            self.x_mask, float(self.hour), self.remaining, self.deadline)

    def viable_wait(self):
        return self.line.viable(self.x_mask, float(self.hour + 1), self.remaining,
                                self.deadline)

    def action_mask(self):
        """Max-plus viability mask  m(t) = [ V(x,t+1,m),  m>0 and V(x,t,m) ].

        Under Theorem 1 the mask is never empty. It can become empty only if
        the plant deviates from the planning model (nominal mask under
        processing delays); then the release is forced as early as possible
        (best effort) and the event is counted.
        """
        m = np.array([self.viable_wait(), self.viable_release()], dtype=bool)
        if not m.any():
            self._mask_empty_now = True
            m = np.array([self.remaining == 0, self.remaining > 0])
        return m

    def slack(self):
        """Hours the remaining jobs can still be postponed (max-plus)."""
        if self.remaining == 0:
            return float(T_HORIZON - self.hour)
        lv = self.line.latest_viable_epoch(self.x_mask, float(self.hour),
                                           self.remaining, self.deadline,
                                           horizon=T_HORIZON - 1)
        return lv - self.hour

    # ---- observation --------------------------------------------------------
    def _obs(self):
        t, H = self.hour, self.LOOKAHEAD
        idx = [t + h for h in range(H)]
        pv = np.array([self.forecast[i] / P_PEAK_KW if i < T_HORIZON else 0.0
                       for i in idx])
        pr = np.array([self.price[i] / self.price_scale if i < T_HORIZON else 0.0
                       for i in idx])
        backlog = [max(0.0, xi - t) / 6.0 if np.isfinite(xi) else 0.0
                   for xi in self.x]
        sl = max(self.slack(), -1.0) / T_HORIZON if self.observe_slack else 0.0
        o = np.concatenate(([t / T_HORIZON, self.remaining / self.n_jobs, sl],
                            backlog, pv, pr))
        if self.tariff:
            o = np.concatenate((o, [self.soc / BAT_E, self.peak_imp / P_PEAK_KW]))
        return np.clip(o, -3.0, 3.0).astype(np.float32)

    # ---- gym API ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng = self._rng
        if self._fixed_day is not None:
            self.solar, self.price, fc = self._fixed_day
        elif self.dataset is not None:
            self.solar, self.price = (np.asarray(a, float) for a in
                                      self.dataset.random_day(rng))
            fc = None
        else:
            self.solar = solar_profile(rng=rng)
            self.price = price_profile(rng=rng)
            fc = None

        if fc is not None:
            self.forecast = fc
        else:
            sigma = (rng.uniform(*self.forecast_sigma_range)
                     if self.forecast_sigma_range else self.forecast_sigma)
            self.forecast = solar_forecast(self.solar, sigma, rng)

        # price normalisation by the mean absolute day-ahead price (known at t=0)
        self.price_scale = max(float(np.mean(np.abs(self.price))), 1e-3)

        if getattr(self, "_fixed_delays", None) is not None:
            self.delays = self._fixed_delays
        elif self.proc_delay_max.any():
            self.delays = rng.uniform(0.0, 1.0, (self.n_jobs, self.line.n)) * self.proc_delay_max
        else:
            self.delays = np.zeros((self.n_jobs, self.line.n))

        self.hour = 0
        self.released = 0
        self.ep_late = 0
        self.ep_lateness_h = 0.0
        self._xm_key = None
        self.ep_mask_empty = 0
        self._mask_empty_now = False
        self.x = np.full(self.line.n, EPS)
        self.load = np.zeros(T_HORIZON)
        self.total_cost = 0.0
        self.soc = 0.0
        self.peak_imp = 0.0
        self.schedule = []
        self.completions = []
        self.explanations = []
        self.ep_interventions = 0
        self.ep_safety_interventions = 0
        self.ep_violations = 0
        return self._obs(), {"action_mask": self.action_mask()}

    def _release(self):
        u = float(self.hour)
        d = self.delays[self.released]
        plant = (SerialLine(np.array(PROC_H) + d) if d.any() else self.plant_nominal)
        x_new = plant.next_state(self.x, u)
        for i, (s, e) in enumerate(plant.busy_windows(self.x, x_new)):
            for t in range(int(np.floor(s)), int(np.ceil(e))):
                if 0 <= t < T_HORIZON:
                    self.load[t] += MACHINE_KW[i] * max(0.0, min(e, t + 1) - max(s, t))
        self.explanations.append(plant.critical_path_text(self.x, u))
        if x_new[-1] > self.deadline + 1e-9:
            self.ep_late += 1
            self.ep_lateness_h += float(x_new[-1] - self.deadline)
        self.x = x_new
        self.released += 1
        self.schedule.append(int(u))
        self.completions.append(float(x_new[-1]))
        return float(x_new[-1])

    def _tariff_cost(self, t):
        tf = self.tariff
        r = self.load[t] - self.solar[t]
        imp, ex = max(r, 0.0), max(-r, 0.0)
        if tf.get("bat"):
            ch = min(ex, BAT_P, (BAT_E - self.soc) / BAT_ETA)
            dis = min(imp, BAT_P, self.soc * BAT_ETA)
            self.soc += ch * BAT_ETA - dis / BAT_ETA
            imp, ex = imp - dis, ex - ch
        c = imp * self.price[t]
        if tf.get("net"):
            c += imp * NET_TAU[t]
        if tf.get("dem") and imp > self.peak_imp:
            c += DEM_LAMBDA * (imp - self.peak_imp)
        self.peak_imp = max(self.peak_imp, imp)
        if tf.get("exp"):
            c -= ex * self.price[t]
        return c

    def step(self, action):
        action = int(action)
        penalty = 0.0
        self._mask_empty_now = False
        mask = self.action_mask()
        if self._mask_empty_now:
            self.ep_mask_empty += 1

        if self.feasibility == "mask":
            if not mask[action]:
                raise RuntimeError("masked action executed: mask not applied")
        elif self.feasibility == "shield":
            if not mask[action]:
                # a request to release when no job is left is a harmless no-op;
                # every other override prevents a deadline violation
                if not (action == 1 and self.remaining == 0):
                    self.ep_safety_interventions += 1
                action = 1 - action           # Thm. 1(a): the other action is viable
                self.ep_interventions += 1
                self.total_interventions += 1
        # "penalty": no correction at all

        if action == 1 and self.remaining > 0:
            y = self._release()
            if self.feasibility == "penalty" and y > self.deadline + 1e-9:
                self.ep_violations += 1
                penalty += self.violation_penalty

        t = self.hour
        if self.tariff:
            cost = self._tariff_cost(t)
        else:
            cost = max(0.0, self.load[t] - self.solar[t]) * self.price[t]
        self.total_cost += cost
        self.hour += 1
        terminated = self.hour >= T_HORIZON

        if terminated:
            if self.remaining > 0:
                # only reachable without the mask / shield
                self.ep_late += self.remaining
                self.ep_violations += self.remaining
                penalty += self.violation_penalty * self.remaining
            self.total_violations += self.ep_violations
            self.total_episodes += 1

        reward = -(cost + penalty) / self.reward_scale
        info = {"action_mask": (self.action_mask() if not terminated
                                else np.array([True, False])),
                "cost": cost}
        if terminated:
            info.update(ep_cost=self.total_cost,
                        ep_interventions=self.ep_interventions,
                        ep_safety_interventions=self.ep_safety_interventions,
                        ep_violations=self.ep_violations,
                        ep_late=self.ep_late,
                        ep_lateness_h=self.ep_lateness_h,
                        ep_mask_empty=self.ep_mask_empty,
                        schedule=list(self.schedule))
        obs = self._obs() if not terminated else np.zeros(self.obs_dim, np.float32)
        return obs, reward, terminated, False, info

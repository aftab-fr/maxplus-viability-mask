"""
Train one RL scheduler (one feasibility variant, one seed).

Variants (see factory_env.py):
    mask     MaskablePPO with the max-plus viability mask   (proposed)
    shield   PPO + post-hoc max-plus shield                 (ablation)
    penalty  PPO + deadline penalty, no correction          (ablation)

Training data: real days of the training years (default 2018-2019), PV
forecast error sigma ~ U[0, 0.3] per episode.

Per-episode training statistics (cost, shield interventions, deadline
violations) are written to runs/<variant>_s<seed>/episodes.csv.

Usage
-----
  python train_ppo.py --variant mask --seed 0
  python train_ppo.py --variant shield --seed 3 --timesteps 500000
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from data_loader import HourlyDataset
from factory_env import FactoryMicrogridEnv

RUNS_DIR = Path("runs")

HYPERPARAMS = dict(
    learning_rate=3e-4,
    n_steps=240,            # 10 episodes per environment per rollout
    batch_size=240,
    n_epochs=10,
    gamma=1.0,              # finite 24-step horizon, undiscounted cost
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,
    policy_kwargs=dict(net_arch=[128, 128], activation_fn=torch.nn.Tanh),
)
N_ENVS = 4
FORECAST_SIGMA_RANGE = (0.0, 0.3)


def year_files(years):
    return [f"data/merged_hourly_{y}.csv" for y in years]


TARIFFS = {"ALL": dict(net=True, dem=True, exp=True, bat=True),
           "NDX": dict(net=True, dem=True, exp=True)}


def make_env_fn(dataset, seed, variant, penalty=500.0, observe_slack=True,
                tariff=None):
    def _make():
        env = FactoryMicrogridEnv(stochastic=True, seed=seed, dataset=dataset,
                                  feasibility=variant,
                                  forecast_sigma_range=FORECAST_SIGMA_RANGE,
                                  violation_penalty=penalty,
                                  observe_slack=observe_slack,
                                  tariff=TARIFFS.get(tariff))
        if variant == "mask":
            env = ActionMasker(env, lambda e: e.action_mask())
        return env
    return _make


class EpisodeLogger(BaseCallback):
    """Write one CSV row per finished training episode."""

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.n = 0

    def _on_training_start(self):
        self.f = open(self.path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["episode", "timestep", "cost", "interventions",
                         "safety_interventions", "violations"])

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "ep_cost" in info:
                self.n += 1
                self.w.writerow([self.n, self.num_timesteps,
                                 round(info["ep_cost"], 4),
                                 info["ep_interventions"],
                                 info["ep_safety_interventions"],
                                 info["ep_violations"]])
        return True

    def _on_training_end(self):
        self.f.close()


def train(variant, seed, timesteps, train_years=(2018, 2019), tag="",
          penalty=500.0, observe_slack=True, tariff=None, **over):
    torch.set_num_threads(1)
    hp = dict(HYPERPARAMS, **{k: v for k, v in over.items() if v is not None})
    out = RUNS_DIR / f"{variant}{tag}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    dataset = HourlyDataset(year_files(train_years), verbose=False)

    vec_env = DummyVecEnv([make_env_fn(dataset, seed * 1000 + i, variant,
                                       penalty, observe_slack, tariff)
                           for i in range(N_ENVS)])
    algo = MaskablePPO if variant == "mask" else PPO
    model = algo("MlpPolicy", vec_env, seed=seed, verbose=0, **hp)

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=EpisodeLogger(out / "episodes.csv"))
    wall = time.time() - t0
    model.save(out / "model")
    meta = dict(variant=variant, seed=seed, timesteps=timesteps,
                train_years=list(train_years), wall_clock_s=round(wall, 1),
                violation_penalty=penalty, observe_slack=observe_slack,
                tariff=tariff,
                n_envs=N_ENVS, forecast_sigma_range=FORECAST_SIGMA_RANGE,
                hyperparams={k: v for k, v in hp.items() if k != "policy_kwargs"},
                net_arch=hp["policy_kwargs"]["net_arch"])
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{variant} s{seed}] done in {wall:.0f} s -> {out}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["mask", "shield", "penalty"], default="mask")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--train-years", type=int, nargs="+", default=[2018, 2019])
    ap.add_argument("--tag", default="", help="suffix for tuning runs")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--ent", type=float)
    ap.add_argument("--net", type=int, nargs="+")
    ap.add_argument("--penalty", type=float, default=500.0)
    ap.add_argument("--tariff", choices=list(TARIFFS))
    ap.add_argument("--no-slack", action="store_true",
                    help="zero the model-derived slack in the observation")
    a = ap.parse_args()
    over = dict(learning_rate=a.lr, ent_coef=a.ent)
    if a.net:
        over["policy_kwargs"] = dict(net_arch=a.net, activation_fn=torch.nn.Tanh)
    train(a.variant, a.seed, a.timesteps, tuple(a.train_years), a.tag,
          a.penalty, not a.no_slack, a.tariff, **over)


if __name__ == "__main__":
    main()

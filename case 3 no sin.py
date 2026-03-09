"""Case 3: RL + Fluent suction/jet control without sinusoidal modulation (refactored)."""

from dataclasses import dataclass
from pathlib import Path
import logging
import os

import ansys.fluent.core as pyfluent
import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


@dataclass
class FluentEnvConfig:
    cas_path: str
    data_path: str = "70 angle unsteady_jet0_dt=0.03s.dat.h5"
    workdir: str = r"D:\LYZ\A case3 with hole jet ag70"
    show_gui: bool = True
    processor_count: int = 56
    slice_len: int = 10
    max_decisions: int = 80

    amplitude_range: tuple[float, float] = (0.0, 200.0)
    baseline_tploss: float = 0.07

    inlet_name: str = "hole_inlet"
    report_file_prefix: str = "report-def-0-rfile"
    max_iter_per_step: int = 30


class CompressorEnv(gym.Env):
    def __init__(self, config: FluentEnvConfig):
        super().__init__()
        self.config = config

        os.chdir(Path(self.config.workdir))
        self._setup_logger()

        self.a_min, self.a_max = self.config.amplitude_range

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf, 0.0], dtype=np.float32),
            high=np.array([1.0, np.inf, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.session = self._launch_fluent_session()
        self._reset_internal_state()

    def _setup_logger(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)

    def _launch_fluent_session(self):
        session = pyfluent.launch_fluent(
            mode="solver",
            dimension=3,
            show_gui=self.config.show_gui,
            processor_count=self.config.processor_count,
        )
        session.settings.file.read_case(file_name=self.config.cas_path)
        session.solution.run_calculation.reporting_interval = self.config.slice_len
        session.solution.run_calculation.profile_update_interval = self.config.slice_len
        return session

    def _reset_internal_state(self) -> None:
        self.a_current = 0.0
        self.current_velocity_expr = "0[m s^-1]"
        self.last_tploss = None
        self.last_action = np.zeros(1, dtype=np.float32)

        self.t_idx = 0
        self.decision_count = 0

        self._amplitude_hist: list[float] = []
        self._tploss_hist: list[float] = []
        self._velocity_expr_hist: list[str] = []

    def _normalized_to_physical(self, a_norm: float) -> float:
        return float(self.a_min + np.clip(a_norm, 0.0, 1.0) * (self.a_max - self.a_min))

    @staticmethod
    def _make_velocity_expression(amplitude: float) -> str:
        return f"{amplitude:.2f}[m s^-1]"

    def _set_inlet_velocity(self, velocity_expression: str) -> None:
        self.session.setup.boundary_conditions.velocity_inlet[
            self.config.inlet_name
        ].momentum.velocity.value = velocity_expression

    def _advance_simulation(self) -> None:
        self.session.settings.solution.run_calculation.dual_time_iterate(
            time_step_count=self.config.slice_len,
            max_iter_per_step=self.config.max_iter_per_step,
        )

    def _find_latest_report_file(self) -> Path:
        base = self.config.report_file_prefix
        candidates = set()
        for pattern in [base, f"{base}_*", f"{base}_*_*"]:
            candidates.update(Path().glob(pattern))

        valid_files = [path for path in candidates if path.is_file()]
        if not valid_files:
            raise FileNotFoundError(f"No report files found for prefix '{base}' in {Path().resolve()}")
        return max(valid_files, key=lambda file_path: file_path.stat().st_mtime)

    def _read_latest_tploss(self) -> float:
        latest_file = self._find_latest_report_file()
        tploss_value = float(pd.read_csv(latest_file, sep=r"\s+", skiprows=2).iloc[-1, 2])
        self.logger.info("Read tploss %.6f from %s", tploss_value, latest_file)
        return tploss_value

    def _compute_reward(self, tploss_scaled: float) -> float:
        return self.config.baseline_tploss * 10.0 - tploss_scaled

    def _build_observation(self, a_norm: float, tploss_scaled: float) -> np.ndarray:
        return np.array(
            [a_norm, tploss_scaled, self.t_idx / self.config.max_decisions],
            dtype=np.float32,
        )

    def seed(self, seed: int = 2025):
        return seed

    def reset(self):
        self.session.settings.file.read_data(file_name=self.config.data_path)
        self._reset_internal_state()

        a_norm_init = 0.0
        self.a_current = self._normalized_to_physical(a_norm_init)
        self.current_velocity_expr = self._make_velocity_expression(self.a_current)
        self._set_inlet_velocity(self.current_velocity_expr)

        tploss_scaled = self.config.baseline_tploss * 10.0
        obs = self._build_observation(a_norm_init, tploss_scaled)

        self._amplitude_hist.append(self.a_current)
        self._tploss_hist.append(tploss_scaled)
        self._velocity_expr_hist.append(self.current_velocity_expr)

        return obs

    def step(self, action):
        self.decision_count += 1

        a_norm = float(np.clip(action[0], 0.0, 1.0))
        amplitude = self._normalized_to_physical(a_norm)

        velocity_expression = self._make_velocity_expression(amplitude)
        self.current_velocity_expr = velocity_expression
        self._set_inlet_velocity(velocity_expression)

        self._advance_simulation()
        tploss_scaled = self._read_latest_tploss() * 10.0
        reward = self._compute_reward(tploss_scaled)

        self.a_current = amplitude
        self.last_tploss = tploss_scaled
        self.last_action = np.array([a_norm], dtype=np.float32)
        self.t_idx += 1

        obs = self._build_observation(a_norm, tploss_scaled)

        self._amplitude_hist.append(amplitude)
        self._tploss_hist.append(tploss_scaled)
        self._velocity_expr_hist.append(velocity_expression)

        done = self.decision_count >= self.config.max_decisions
        info = {
            "tploss_now": tploss_scaled,
            "A": amplitude,
            "expr": velocity_expression,
            "t_idx": self.t_idx,
            "reward_now": reward,
        }
        self.logger.info("Step info: %s", info)

        return obs, float(reward), done, info

    def close(self):
        if hasattr(self, "session") and self.session is not None:
            try:
                self.session.exit()
            except Exception as error:
                self.logger.warning("Failed to close Fluent session cleanly: %s", error)


def build_model(env: DummyVecEnv) -> PPO:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs={"net_arch": [64, 128, 64]},
        learning_rate=5e-4,
        n_steps=80,
        batch_size=20,
        n_epochs=5,
        clip_range=0.2,
        gae_lambda=0.95,
        ent_coef=0.01,
    )
    return model


def train_and_save(model: PPO, total_timesteps: int, save_path: str) -> None:
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)


def main() -> None:
    config = FluentEnvConfig(cas_path="70 angle unsteady_jet0_dt=0.03s.cas.h5")
    env = DummyVecEnv([lambda: CompressorEnv(config)])

    model = build_model(env)
    model = PPO.load("my_model_nosin4", env=env)

    train_and_save(model=model, total_timesteps=100 * 10, save_path="my_model_nosin5")


if __name__ == "__main__":
    main()

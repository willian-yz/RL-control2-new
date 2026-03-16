"""Unified RL + Fluent framework for case3/case4 compressor control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import csv
from datetime import datetime
import json
import logging
import os

import ansys.fluent.core as pyfluent
import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


@dataclass
class EnvConfig:
    case_id: int
    cas_path: str
    data_path: str
    workdir: str
    show_gui: bool = True
    processor_count: int = 56
    slice_len: int = 10
    max_decisions: int = 80
    inlet_name: str = "hole_inlet"
    report_file_prefix: str = "report-def-0-rfile"
    max_iter_per_step: int = 30
    initial_tploss: float = 0.7


@dataclass
class RLConfig:
    action_dim: int = 1
    amplitude_range: tuple[float, float] = (0.0, 200.0)
    frequency_range: tuple[float, float] = (200.0, 2000.0)
    expr_mode: str = "constant"  # "constant" or "sin"
    pi: float = 3.1415926
    t0: float = 0.210084
    baseline_tploss: float = 0.07

    learning_rate: float = 5e-4
    n_steps: int = 80
    batch_size: int = 40
    n_epochs: int = 5
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    ent_coef: float = 0.01


class HistoryRecorder:
    def __init__(self, root: Path, case_id: int, mode: str):
        self.root = root
        self.mode = mode
        self.case_id = case_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.root / f"case{case_id}_{mode}_history.csv"

    def _fallback_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.csv_path.with_name(f"{self.csv_path.stem}_{timestamp}{self.csv_path.suffix}")

    def _write_row(self, path: Path, row: dict) -> None:
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def append(self, row: dict) -> None:
        try:
            self._write_row(self.csv_path, row)
        except PermissionError:
            fallback = self._fallback_path()
            self._write_row(fallback, row)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.csv_path)


class CompressorEnv(gym.Env):
    def __init__(self, env_config: EnvConfig, rl_config: RLConfig, mode: str = "train", history_dir: str = "history"):
        super().__init__()
        self.env_config = env_config
        self.rl_config = rl_config
        self.mode = mode

        os.chdir(Path(self.env_config.workdir))
        self._setup_logger()

        self.a_min, self.a_max = self.rl_config.amplitude_range
        self.f_min, self.f_max = self.rl_config.frequency_range

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.rl_config.action_dim,), dtype=np.float32)
        if self.rl_config.action_dim == 1:
            self.observation_space = spaces.Box(
                low=np.array([0.0, -np.inf, 0.0], dtype=np.float32), # A, total_loss, steps_of_decision
                high=np.array([1.0, np.inf, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, -np.inf, 0.0], dtype=np.float32), # A, f, total_loss, steps_of_decision
                high=np.array([1.0, 1.0, np.inf, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self.history = HistoryRecorder(Path(history_dir), self.env_config.case_id, mode)
        self.session = self._launch_fluent_session()
        self._reset_internal_state()

    def _setup_logger(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.env_config.case_id}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)

    def _launch_fluent_session(self):
        session = pyfluent.launch_fluent(
            mode="solver",
            dimension=2,
            show_gui=self.env_config.show_gui,
            processor_count=self.env_config.processor_count,
        )
        session.settings.file.read_case(file_name=self.env_config.cas_path)
        session.solution.run_calculation.reporting_interval = self.env_config.slice_len
        session.solution.run_calculation.profile_update_interval = self.env_config.slice_len
        return session

    def _reset_internal_state(self) -> None:
        self.a_current = 0.0
        self.f_current = 0.0
        self.current_velocity_expr = "0[m s^-1]"
        self.last_tploss = float(self.env_config.initial_tploss)
        self.t_idx = 0
        self.decision_count = 0

    def _normalized_to_amplitude(self, action_a: float) -> float:
        a_norm = float(np.clip(action_a, 0.0, 1.0))
        return float(self.a_min + a_norm * (self.a_max - self.a_min))

    def _normalized_to_frequency(self, action_f: float) -> float:
        f_norm = float(np.clip(action_f, 0.0, 1.0))
        return float(self.f_min + f_norm * (self.a_max - self.a_min))

    def _make_velocity_expression(self, amplitude: float, frequency: Optional[float] = None) -> str:
        if self.rl_config.action_dim == 1:
            return f"{amplitude:.2f}[m s^-1]"

        if self.rl_config.action_dim == 2:
            return (
                f"{amplitude:.2f}[m s^-1]*sin(2*{self.rl_config.pi}*{frequency:.2f} [s^-1]"
                f"*(t - {self.rl_config.t0}[s]))"
            )

    def _set_inlet_velocity(self, velocity_expression: str) -> None:
        self.session.setup.boundary_conditions.velocity_inlet[
            self.env_config.inlet_name
        ].momentum.velocity.value = velocity_expression

    def _advance_simulation(self) -> None:
        self.session.settings.solution.run_calculation.dual_time_iterate(
            time_step_count=self.env_config.slice_len,
            max_iter_per_step=self.env_config.max_iter_per_step,
        )

    def _find_latest_report_file(self) -> Optional[Path]:
        base = self.env_config.report_file_prefix
        candidates = set()
        for pattern in [base, f"{base}_*", f"{base}_*_*"]:
            candidates.update(Path().glob(pattern))
        valid_files = [path for path in candidates if path.is_file()]
        if not valid_files:
            return None
        return max(valid_files, key=lambda file_path: file_path.stat().st_mtime)

    def _read_latest_tploss_scaled(self) -> float:
        latest_file = self._find_latest_report_file()
        if latest_file is None:
            self.logger.warning("No report file found, fallback to last_tploss=%.6f", self.last_tploss)
            return float(self.last_tploss)
        tploss_value = float(pd.read_csv(latest_file, sep=r"\s+", skiprows=2).iloc[-1, 1]) * 10.0 #区分瞬时值和平均值
        self.logger.info("Read tploss %.6f from %s", tploss_value, latest_file)
        return tploss_value

    def _compute_reward(self, tploss_scaled: float) -> float:
        return self.rl_config.baseline_tploss * 10.0 - tploss_scaled

    def _build_observation(self, action: np.ndarray, tploss_scaled: float) -> np.ndarray:
        progress = self.t_idx / self.env_config.max_decisions
        if self.rl_config.action_dim == 1:
            return np.array([action[0], tploss_scaled, progress], dtype=np.float32)
        return np.array([action[0], action[1], tploss_scaled, progress], dtype=np.float32)

    def reset(self):
        self.session.settings.file.read_data(file_name=self.env_config.data_path)
        self._reset_internal_state()
        action_init = np.zeros(self.rl_config.action_dim, dtype=np.float32)
        if self.rl_config.action_dim == 1:
            amplitue = self._normalized_to_amplitude(action_init[0])
            self.current_velocity_expr = self._make_velocity_expression(amplitue)
        else:
            amplitude = self._normalized_to_amplitude(action_init[0])
            frequency = self._normalized_to_frequency(action_init[1])
            self.current_velocity_expr = self._make_velocity_expression(amplitude, frequency)
        self._set_inlet_velocity(self.current_velocity_expr)

        self.last_tploss = float(self.env_config.initial_tploss)
        return self._build_observation(action_init, self.last_tploss)

    def step(self, action):
        self.decision_count += 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.rl_config.action_dim == 1:
            amplitude = self._normalized_to_amplitude(action[0])
            frequency = None
            self.current_velocity_expr = self._make_velocity_expression(amplitude)
        else:
            amplitude = self._normalized_to_amplitude(action[0])
            frequency = self._normalized_to_frequency(action[1])
            self.current_velocity_expr = self._make_velocity_expression(amplitude, frequency)

        self._set_inlet_velocity(self.current_velocity_expr)

        self._advance_simulation()
        tploss_scaled = self._read_latest_tploss_scaled()
        reward = self._compute_reward(tploss_scaled)

        self.a_current = amplitude
        self.f_current = float(frequency) if frequency is not None else 0.0
        self.last_tploss = tploss_scaled
        self.t_idx += 1

        done = self.decision_count >= self.env_config.max_decisions
        obs = self._build_observation(action, tploss_scaled)

        info = {
            "tploss_now": tploss_scaled,
            "A": amplitude,
            "f": frequency if frequency is not None else "",
            "expr": self.current_velocity_expr,
            "t_idx": self.t_idx,
            "reward_now": reward,
            "mode": self.mode,
        }

        row = {
            "step": self.t_idx,
            "reward": reward,
            "tploss": tploss_scaled,
            "action_a": float(action[0]),
            "action_f": float(action[1]) if self.rl_config.action_dim > 1 else "",
            "A": amplitude,
            "f": frequency if self.rl_config.action_dim > 1 else "",
            "expr": self.current_velocity_expr,
        }
        self.history.append(row)

        return obs, float(reward), done, info

    def close(self):
        if hasattr(self, "session") and self.session is not None:
            try:
                self.session.exit()
            except Exception as error:
                self.logger.warning("Failed to close Fluent session cleanly: %s", error)


class ExperimentManager:
    def __init__(self, env_config: EnvConfig, rl_config: RLConfig, artifact_dir: str = "artifacts"):
        self.env_config = env_config
        self.rl_config = rl_config
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.artifact_dir / f"case{self.env_config.case_id}_model_meta.json"

    def _build_model(self, env: DummyVecEnv) -> PPO:
        return PPO(
            "MlpPolicy",
            env,
            verbose=1,
            policy_kwargs={"net_arch": [64, 128, 64]},
            learning_rate=self.rl_config.learning_rate,
            n_steps=self.rl_config.n_steps,
            batch_size=self.rl_config.batch_size,
            n_epochs=self.rl_config.n_epochs,
            clip_range=self.rl_config.clip_range,
            gae_lambda=self.rl_config.gae_lambda,
            ent_coef=self.rl_config.ent_coef,
        )

    def _write_text_with_fallback(self, path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(content, encoding="utf-8")
            return path
        except (PermissionError, FileNotFoundError):
            path.parent.mkdir(parents=True, exist_ok=True)
            fallback = path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
            fallback.write_text(content, encoding="utf-8")
            return fallback

    def _save_figure_with_fallback(self, fig, output: Path) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(output, dpi=180)
            return output
        except (PermissionError, FileNotFoundError):
            output.parent.mkdir(parents=True, exist_ok=True)
            fallback = output.with_name(f"{output.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{output.suffix}")
            fig.savefig(fallback, dpi=180)
            return fallback

    def train(self, train_steps: int, load_model_path: Optional[str] = None) -> Path:
        env = DummyVecEnv([lambda: CompressorEnv(self.env_config, self.rl_config, mode="train")])
        if load_model_path:
            model = PPO.load(load_model_path, env=env)
            existing_steps = int(getattr(model, "num_timesteps", 0))
        else:
            model = self._build_model(env)
            existing_steps = 0

        model.learn(total_timesteps=train_steps)
        total_steps = existing_steps + train_steps
        save_name = f"my_model_case{self.env_config.case_id}_step{total_steps}"
        save_path = self.artifact_dir / save_name
        try:
            model.save(str(save_path))
        except PermissionError:
            save_path = save_path.with_name(f"{save_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{save_path.suffix}")
            model.save(str(save_path))

        meta = {
            "case_id": self.env_config.case_id,
            "total_steps": total_steps,
            "model_path": str(save_path),
            "env_config": asdict(self.env_config),
            "rl_config": asdict(self.rl_config),
        }
        self._write_text_with_fallback(self.meta_path, json.dumps(meta, ensure_ascii=False, indent=2))
        return save_path

    def test(self, model_path: str, tot_steps: int = 1) -> list[float]:
        env = CompressorEnv(self.env_config, self.rl_config, mode="test")
        model = PPO.load(model_path)
        episode_rewards: list[float] = []


        obs = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        while not done and step < tot_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += float(reward)
            step += 1
        episode_rewards.append(total_reward)
        env.close()
        return episode_rewards

    def plot_history(self, mode: str = "train") -> Path:
        recorder = HistoryRecorder(Path("history"), self.env_config.case_id, mode)
        df = recorder.to_dataframe()
        if df.empty:
            raise FileNotFoundError(f"No {mode} history for case {self.env_config.case_id}")

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].plot(df["step"], df["reward"])
        axes[0, 0].set_title("Reward")
        axes[0, 1].plot(df["step"], df["tploss"])
        axes[0, 1].set_title("Tploss")
        axes[1, 0].plot(df["step"], df["A"])
        axes[1, 0].set_title("Amplitude A")
        if "f" in df.columns and df["f"].replace("", np.nan).notna().any():
            axes[1, 1].plot(df["step"], pd.to_numeric(df["f"], errors="coerce"))
            axes[1, 1].set_title("Frequency f")
        else:
            axes[1, 1].axis("off")
        fig.tight_layout()

        output = self.artifact_dir / f"case{self.env_config.case_id}_{mode}_history.png"
        output = self._save_figure_with_fallback(fig, output)
        plt.close(fig)
        return output

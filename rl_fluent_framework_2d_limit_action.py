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
from collections.abc import Iterable

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
    inlet_names_dim2: tuple[str, str] = ("hole_inlet_1", "hole_inlet_2")
    report_file_prefix: str = "report-def-0-rfile"
    max_iter_per_step: int = 30
    initial_tploss: float = 0.7
    rake_names: tuple[str, ...] = (
        "rake-1",
        "rake-2",
        "rake-3",
        "rake-4",
        "rake-5",
        "rake-6",
        "rake-7",
        "rake-8",
        "rake-9",
    )
    rake_point_count: int = 60


@dataclass
class RLConfig:
    action_dim: int = 1
    amplitude_range: tuple[float, float] = (0.0, 200.0)
    frequency_range: tuple[float, float] = (200.0, 2000.0)
    max_delta_amplitude: float = 20.0
    use_delta_action: bool = False
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

        # 下面这一段表示了是否使用delta_action，如果使用了那么速度[-1,1]阶梯递增，反之则是[0,1]
        if self.rl_config.use_delta_action:
            if self.rl_config.action_dim == 1:
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            else:
                self.action_space = spaces.Box(
                    low=np.array([-1.0, -1.0], dtype=np.float32),
                    high=np.array([1.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                )
        else:
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.rl_config.action_dim,), dtype=np.float32)
        action_obs_low = -1.0 if self.rl_config.use_delta_action else 0.0

        velocity_feature_dim = self.env_config.rake_point_count * 2  # Vx+Vy
        if self.rl_config.action_dim == 1:
            self.observation_space = spaces.Box(
                low=np.concatenate(
                    [
                        np.array([action_obs_low, -np.inf], dtype=np.float32),  # A, tploss
                        np.full(velocity_feature_dim, -np.inf, dtype=np.float32),
                    ]
                ),
                high=np.concatenate(
                    [
                        np.array([1.0, np.inf], dtype=np.float32),
                        np.full(velocity_feature_dim, np.inf, dtype=np.float32),
                    ]
                ),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=np.concatenate(
                    [
                        np.array([action_obs_low, -1.0, -np.inf], dtype=np.float32),  # A, f, tploss
                        np.full(velocity_feature_dim, -np.inf, dtype=np.float32),
                    ]
                ),
                high=np.concatenate(
                    [
                        np.array([1.0, 1.0, np.inf], dtype=np.float32),
                        np.full(velocity_feature_dim, np.inf, dtype=np.float32),
                    ]
                ),
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
        self.a_current_2 = 0.0
        self.f_current = 0.0
        self.current_velocity_expr = "0[m s^-1]"
        self.last_tploss = float(self.env_config.initial_tploss)
        self.t_idx = 0
        self.decision_count = 0
        self._needs_session_restart = False

    def _normalized_to_amplitude(self, action_a: float) -> float:
        a_norm = float(np.clip(action_a, 0.0, 1.0))
        return float(self.a_min + a_norm * (self.a_max - self.a_min))

    def _limit_amplitude_step(self, target_amplitude: float) -> float:
        max_delta = abs(float(self.rl_config.max_delta_amplitude))
        if max_delta <= 0:
            return float(np.clip(target_amplitude, self.a_min, self.a_max))
        lower = self.a_current - max_delta
        upper = self.a_current + max_delta
        return float(np.clip(target_amplitude, lower, upper))


    def _action_to_amplitude(self, action_a: float, current_amplitude: Optional[float] = None) -> float:
        amp_now = self.a_current if current_amplitude is None else float(current_amplitude)
        if self.rl_config.use_delta_action:
            delta_norm = float(np.clip(action_a, -1.0, 1.0))
            delta = delta_norm * abs(float(self.rl_config.max_delta_amplitude))
            target_amplitude = amp_now + delta
            return float(np.clip(target_amplitude, self.a_min, self.a_max))

        target_amplitude = self._normalized_to_amplitude(action_a)
        max_delta = abs(float(self.rl_config.max_delta_amplitude))
        if max_delta <= 0:
            return float(np.clip(target_amplitude, self.a_min, self.a_max))
        lower = amp_now - max_delta
        upper = amp_now + max_delta
        return float(np.clip(target_amplitude, lower, upper))

    def _normalized_to_frequency(self, action_f: float) -> float:
        f_norm = float(np.clip(action_f, 0.0, 1.0))
        return float(self.f_min + f_norm * (self.f_max - self.f_min))

    def _make_velocity_expression(self, amplitude: float, frequency: Optional[float] = None) -> str:
        if self.rl_config.action_dim == 1:
            return f"{amplitude:.2f}[m s^-1]"

        if self.rl_config.action_dim == 2:
            return f"{amplitude:.2f}[m s^-1]"

    def _set_inlet_velocity(self, velocity_expression: str, velocity_expression_dim2: Optional[str] = None) -> None:
        if self.rl_config.action_dim == 2:
            inlet_1, inlet_2 = self.env_config.inlet_names_dim2
            expr_2 = velocity_expression if velocity_expression_dim2 is None else velocity_expression_dim2
            self.session.setup.boundary_conditions.velocity_inlet[inlet_1].momentum.velocity.value = velocity_expression
            self.session.setup.boundary_conditions.velocity_inlet[inlet_2].momentum.velocity.value = expr_2
            return

        self.session.setup.boundary_conditions.velocity_inlet[self.env_config.inlet_name].momentum.velocity.value = (
            velocity_expression
        )

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

        try:
            with latest_file.open("rb") as file_obj:
                file_obj.seek(0, os.SEEK_END)
                position = file_obj.tell() - 1
                last_line = b""
                while position >= 0:
                    file_obj.seek(position)
                    char = file_obj.read(1)
                    if char == b"\n" and last_line:
                        break
                    if char != b"\n":
                        last_line = char + last_line
                    position -= 1
            values = np.fromstring(last_line.decode("utf-8", errors="ignore"), sep=" ")
            if values.size >= 2:
                tploss_value = float(values[1]) * 10.0  # 区分瞬时值和平均值
                self.logger.info("Read tploss %.6f from %s", tploss_value, latest_file)
                return tploss_value
            raise ValueError("No tploss numeric column found in last line.")
        except Exception as error:
            self.logger.warning("Failed to parse %s, fallback to pandas parser: %s", latest_file, error)
            tploss_value = float(pd.read_csv(latest_file, sep=r"\s+", skiprows=2).iloc[-1, 1]) * 10.0
            self.logger.info("Read tploss %.6f from %s", tploss_value, latest_file)
            return tploss_value

    def _compute_reward(self, tploss_scaled: float) -> float:
        return self.rl_config.baseline_tploss * 10.0 - tploss_scaled

    def _to_1d_numeric_array(self, data_obj) -> np.ndarray:
        def _try_cast(candidate) -> Optional[np.ndarray]:
            if candidate is None:
                return None
            try:
                arr = np.asarray(candidate, dtype=np.float64).reshape(-1)
                if arr.size:
                    return arr
            except Exception:
                return None
            return None

        if isinstance(data_obj, dict):
            for value in data_obj.values():
                result = _try_cast(value)
                if result is not None:
                    return result

        for attr in ("scalar_data", "values", "data", "field_data"):
            if hasattr(data_obj, attr):
                result = _try_cast(getattr(data_obj, attr))
                if result is not None:
                    return result

        result = _try_cast(data_obj)
        return result if result is not None else np.zeros(0, dtype=np.float64)

    def _flatten_surface_data(self, raw_data) -> np.ndarray:
        if isinstance(raw_data, dict):
            chunks: list[np.ndarray] = []
            for surface_name in self.env_config.rake_names:
                if surface_name in raw_data:
                    chunks.append(self._to_1d_numeric_array(raw_data[surface_name]))
            if chunks:
                return np.concatenate(chunks)
            return np.concatenate([self._to_1d_numeric_array(v) for v in raw_data.values() if v is not None])

        if isinstance(raw_data, Iterable) and not isinstance(raw_data, (str, bytes, np.ndarray)):
            chunks = [self._to_1d_numeric_array(v) for v in raw_data]
            chunks = [chunk for chunk in chunks if chunk.size > 0]
            if chunks:
                return np.concatenate(chunks)

        return self._to_1d_numeric_array(raw_data)

    def _read_rake_velocity_features(self) -> np.ndarray:
        target_points = int(self.env_config.rake_point_count)

        try:
            field_data = self.session.fields.field_data
            surfaces = list(self.env_config.rake_names)
            vx_raw = field_data.get_scalar_field_data(field_name="x-velocity", surfaces=surfaces)
            vy_raw = field_data.get_scalar_field_data(field_name="y-velocity", surfaces=surfaces)
            vx = self._flatten_surface_data(vx_raw)
            vy = self._flatten_surface_data(vy_raw)
        except Exception as error:
            self.logger.warning("Failed to read rake velocities, fallback to zeros: %s", error)
            return np.zeros(target_points * 2, dtype=np.float32)

        if vx.size < target_points:
            vx = np.pad(vx, (0, target_points - vx.size), mode="constant")
        if vy.size < target_points:
            vy = np.pad(vy, (0, target_points - vy.size), mode="constant")

        vx = vx[:target_points]
        vy = vy[:target_points]
        return np.concatenate([vx, vy]).astype(np.float32)

    def _build_observation(self, action: np.ndarray, tploss_scaled: float) -> np.ndarray:
        velocity_features = self._read_rake_velocity_features()
        if self.rl_config.action_dim == 1:
            return np.concatenate(
                [
                    np.array([action[0], tploss_scaled], dtype=np.float32),
                    velocity_features / 100, # 归一化
                ]
            )
        return np.concatenate(
            [
                np.array([action[0], action[1], tploss_scaled], dtype=np.float32),
                velocity_features / 100, # 归一化
            ]
        )

    def reset(self):
        if getattr(self, "_needs_session_restart", False):
            try:
                self.session.exit()
            except Exception as error:
                self.logger.warning("Failed to close Fluent session before restart: %s", error)
            self.session = self._launch_fluent_session()
            self._needs_session_restart = False

        self.session.settings.file.read_data(file_name=self.env_config.data_path)
        self._reset_internal_state()
        action_init = np.zeros(self.rl_config.action_dim, dtype=np.float32)
        if self.rl_config.action_dim == 1:
            amplitue = self._normalized_to_amplitude(action_init[0])
            self.current_velocity_expr = self._make_velocity_expression(amplitue)
            self._set_inlet_velocity(self.current_velocity_expr)
        else:
            amplitude_1 = self._normalized_to_amplitude(action_init[0])
            amplitude_2 = self._normalized_to_amplitude(action_init[1])
            expr_1 = self._make_velocity_expression(amplitude_1)
            expr_2 = self._make_velocity_expression(amplitude_2)
            self.current_velocity_expr = f"{expr_1} | {expr_2}"
            self._set_inlet_velocity(expr_1, expr_2)

        self.last_tploss = float(self.env_config.initial_tploss)
        return self._build_observation(action_init, self.last_tploss)

    def step(self, action):
        self.decision_count += 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.rl_config.action_dim == 1:
            amplitude = self._action_to_amplitude(action[0])
            frequency = None
            self.current_velocity_expr = self._make_velocity_expression(amplitude)
            self._set_inlet_velocity(self.current_velocity_expr)
        else:
            amplitude = self._action_to_amplitude(action[0], current_amplitude=self.a_current)
            amplitude_2 = self._action_to_amplitude(action[1], current_amplitude=self.a_current_2)
            frequency = None
            expr_1 = self._make_velocity_expression(amplitude)
            expr_2 = self._make_velocity_expression(amplitude_2)
            self.current_velocity_expr = f"{expr_1} | {expr_2}"
            self._set_inlet_velocity(expr_1, expr_2)

        self._advance_simulation()
        tploss_scaled = self._read_latest_tploss_scaled()
        reward = self._compute_reward(tploss_scaled)

        self.a_current = amplitude
        if self.rl_config.action_dim == 2:
            self.a_current_2 = amplitude_2
        self.f_current = float(frequency) if frequency is not None else 0.0
        self.last_tploss = tploss_scaled
        self.t_idx += 1

        done = self.decision_count >= self.env_config.max_decisions
        if done:
            self._needs_session_restart = True
        obs = self._build_observation(action, tploss_scaled)

        info = {
            "tploss_now": tploss_scaled,
            "A": amplitude,
            "f": self.a_current_2 if self.rl_config.action_dim > 1 else "",
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
            "f": self.a_current_2 if self.rl_config.action_dim > 1 else "",
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

        save_interval = int(self.env_config.max_decisions)
        if save_interval <= 0:
            save_interval = int(train_steps)

        trained_steps = 0
        checkpoints: list[str] = []
        save_path: Optional[Path] = None

        while trained_steps < train_steps:
            chunk_steps = min(save_interval, train_steps - trained_steps)
            model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False) #这里保证是False才能维持步数恒定训练
            trained_steps += chunk_steps

            total_steps = existing_steps + trained_steps
            save_name = f"my_model_case{self.env_config.case_id}_step{total_steps}"
            save_path = self.artifact_dir / save_name
            try:
                model.save(str(save_path))
            except PermissionError:
                save_path = save_path.with_name(
                    f"{save_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{save_path.suffix}"
                )
                model.save(str(save_path))
            checkpoints.append(str(save_path))

        if save_path is None:
            raise RuntimeError("Training did not run any step; no checkpoint was saved.")

        meta = {
            "case_id": self.env_config.case_id,
            "total_steps": existing_steps + trained_steps,
            "model_path": str(save_path),
            "checkpoints": checkpoints,
            "save_interval": save_interval,
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
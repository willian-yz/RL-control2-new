"""Unified RL + Fluent framework for case3/case4 compressor control."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    inlet_names_dim2: tuple[str, str] = ("inlet_hole_15", "inlet_hole_60")
    report_file_prefix: str = "report-def-0-rfile"
    max_iter_per_step: int = 30
    initial_tploss: np.ndarray = field(default_factory=lambda: np.array([0.12151944, 0.12023581, 0.11902861, 0.11849607, 0.11890123,
       0.12013197, 0.12149273, 0.12232298, 0.12237743, 0.12164915,
       0.12053154, 0.11961707, 0.11934691, 0.11983936, 0.12083819,
       0.12171283, 0.12201158, 0.12158998, 0.12060359, 0.11957217,
       0.11900271, 0.1191502 , 0.12000738, 0.1210556 , 0.12171178,
       0.12173892, 0.12109944, 0.12012228, 0.11932182, 0.11909269,
       0.11958641, 0.12058729, 0.12154518, 0.12199737, 0.12179437,
       0.12099092, 0.1200156 , 0.11934558, 0.11928197, 0.11988009,
       0.12081682, 0.12157555, 0.12182433, 0.12149242, 0.12073678,
       0.11998021, 0.11959072, 0.11973707, 0.12036271, 0.12107553,
       0.12146039, 0.12134983, 0.12074867, 0.11997565, 0.11942225,
       0.11934729, 0.11983309, 0.12062764, 0.12130672, 0.12154791]))

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
    max_delta_amplitude: float = 20.0
    use_delta_action: bool = False
    expr_mode: str = "constant"  # "constant" or "sin"
    pi: float = 3.1415926
    t0: float = 0.210084
    policy_kwargs: dict = field(default_factory=lambda: {"net_arch": [128, 256, 128]})
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

        velocity_feature_dim = self.env_config.rake_point_count * 3  # Vx+Vy+p
        if self.rl_config.action_dim == 1:
            self.observation_space = spaces.Box(
                low=np.concatenate(
                    [
                        np.array([self.a_min / 100, -np.inf], dtype=np.float32),  # A, tploss
                        np.full(velocity_feature_dim, -np.inf, dtype=np.float32),
                    ]
                ),
                high=np.concatenate(
                    [
                        np.array([self.a_max / 100, np.inf], dtype=np.float32),
                        np.full(velocity_feature_dim, np.inf, dtype=np.float32),
                    ]
                ),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=np.concatenate(
                    [
                        np.array([self.a_min / 100, self.a_min / 100, -np.inf], dtype=np.float32),  # A, f, tploss
                        np.full(velocity_feature_dim, -np.inf, dtype=np.float32),
                    ]
                ),
                high=np.concatenate(
                    [
                        np.array([self.a_max / 100, self.a_max / 100, np.inf], dtype=np.float32),
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
        self.current_velocity_expr = "0[m s^-1]"
        self.last_tploss = float(self.env_config.initial_tploss[0])
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
        pd_file = pd.read_csv(latest_file, sep=r"\s+", skiprows=2)
        tploss_value = float(pd_file.iloc[-1, 1]) * 10.0 #区分瞬时值和平均值
        wg_wf = float(pd_file.iloc[-1, 3])
        self.logger.info("Read tploss %.6f from %s", tploss_value, latest_file)
        return tploss_value,wg_wf

    def _compute_reward(self, tploss_scaled: float, wg_wf:float, t_idx: int) -> float:
        #print(t_idx,self.env_config.initial_tploss[t_idx] * 10.0,tploss_scaled)
        w_rl = tploss_scaled / 10
        w_ori = self.env_config.initial_tploss[t_idx]
        dw = w_ori - w_rl
        #相对下降[0,1],绝对下降[0,1],
        wg_wf_reward = 1 if np.isfinite(wg_wf) and 1 <= wg_wf <= 3 else -1
        reward = (dw / w_ori) + dw * 10 + wg_wf_reward
        return reward

    def _read_rake_velocity_features(self) -> np.ndarray:
        vx_values: list[np.ndarray] = []
        vy_values: list[np.ndarray] = []
        p_values: list[np.ndarray] = []
        field_data = self.session.fields.field_data
        for rake_name in self.env_config.rake_names:
            vx_raw = field_data.get_scalar_field_data(field_name="x-velocity", surfaces=[rake_name])
            vy_raw = field_data.get_scalar_field_data(field_name="y-velocity", surfaces=[rake_name])
            p_raw = field_data.get_scalar_field_data(field_name="pressure", surfaces=[rake_name])
            for i_raw in range(len(vx_raw[:])):
                vx_arr = vx_raw[i_raw].scalar_data
                vy_arr = vy_raw[i_raw].scalar_data
                p_arr = p_raw[i_raw].scalar_data
                vx_values.append(vx_arr)
                vy_values.append(vy_arr)
                p_values.append(p_arr) #1000

        # 测试是否会写出数据
        print(np.concatenate([vx_values, vy_values,np.array(p_values)/1000]).astype(np.float32))
        return np.concatenate([vx_values, vy_values,np.array(p_values)/1000]).astype(np.float32)

    def _build_observation(self, tploss_scaled: float) -> np.ndarray:
        if self.env_config.rake_point_count != 0:
            velocity_features = self._read_rake_velocity_features()
        else:
            velocity_features = np.array([], dtype=np.float32)
        if self.rl_config.action_dim == 1:
            return np.concatenate(
                [
                    np.array([self.a_current / 100, tploss_scaled*10], dtype=np.float32), #tploss再放大10倍
                    velocity_features / 100, # 归一化
                ]
            )
        return np.concatenate(
            [
                np.array([self.a_current / 100, self.a_current_2 / 100, tploss_scaled*10], dtype=np.float32),#tploss再放大10倍
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
        if self.rl_config.action_dim == 1:
            amplitue = 0.0
            self.current_velocity_expr = self._make_velocity_expression(amplitue)
            self._set_inlet_velocity(self.current_velocity_expr)
            self.a_current = amplitue
        else:
            amplitude_1 = 0.0
            amplitude_2 = 0.0
            expr_1 = self._make_velocity_expression(amplitude_1)
            expr_2 = self._make_velocity_expression(amplitude_2)
            self.current_velocity_expr = f"{expr_1} | {expr_2}"
            self._set_inlet_velocity(expr_1, expr_2)
            self.a_current = amplitude_1
            self.a_current_2 = amplitude_2

        self.last_tploss = float(self.env_config.initial_tploss[0])
        return self._build_observation(self.last_tploss * 10)

    def step(self, action):
        self.decision_count += 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.rl_config.action_dim == 1:
            amplitude = self._action_to_amplitude(action[0])
            self.current_velocity_expr = self._make_velocity_expression(amplitude)
            self._set_inlet_velocity(self.current_velocity_expr)
        else:
            amplitude = self._action_to_amplitude(action[0], current_amplitude=self.a_current)
            amplitude_2 = self._action_to_amplitude(action[1], current_amplitude=self.a_current_2)
            expr_1 = self._make_velocity_expression(amplitude)
            expr_2 = self._make_velocity_expression(amplitude_2)
            self.current_velocity_expr = f"{expr_1} | {expr_2}"
            self._set_inlet_velocity(expr_1, expr_2)

        self._advance_simulation()
        tploss_scaled, wg_wf = self._read_latest_tploss_scaled()
        reward = self._compute_reward(tploss_scaled,wg_wf,self.t_idx)

        self.a_current = amplitude
        if self.rl_config.action_dim == 2:
            self.a_current_2 = amplitude_2
        self.last_tploss = tploss_scaled

        done = self.decision_count >= self.env_config.max_decisions
        if done:
            self._needs_session_restart = True
        obs = self._build_observation(tploss_scaled)

        info = {
            "tploss_now": tploss_scaled/10,
            "A_1": amplitude,
            "A_2": self.a_current_2 if self.rl_config.action_dim > 1 else "",
            "expr": self.current_velocity_expr,
            "t_idx": self.t_idx,
            "reward_now": reward,
            "mode": self.mode,
        }

        row = {
            "step": self.t_idx+1,
            "reward": reward,
            "tploss_origin": self.env_config.initial_tploss[self.t_idx],
            "tploss": tploss_scaled/10,
            "action_a1": float(action[0]),
            "action_a2": float(action[1]) if self.rl_config.action_dim > 1 else "",
            "A_1": amplitude,
            "A_2": self.a_current_2 if self.rl_config.action_dim > 1 else "",
            "expr": self.current_velocity_expr,
        }
        self.history.append(row)
        self.t_idx += 1
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
            policy_kwargs=self.rl_config.policy_kwargs,
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
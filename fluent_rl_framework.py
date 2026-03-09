from __future__ import annotations

"""
Unified RL + Fluent framework for compressor/jet control.

Module layout:
1) Config layer: dataclasses for simulation and PPO parameters.
2) Fluent interaction layer: launch/read/set BC/advance steps.
3) IO layer: report file parsing and CSV export.
4) Reward layer: transparent reward calculation.
5) RL environment layer: reset/step/obs and logging.
6) Training/testing helpers: train model, test model, plot curves.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ansys.fluent.core as pyfluent
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


# =========================
# 1) Configuration Layer
# =========================


@dataclass
class SimulationConfig:
    """Simulation and environment configuration."""

    case_name: str
    work_dir: str
    cas_path: str
    data_path: str

    # RL action setup
    # action_dims = 1: only control A
    # action_dims = 2: control A and F
    action_dims: int = 2

    # Fluent runtime setup
    show_gui: bool = True
    processor_count: int = 56
    slice_len: int = 10
    max_iter_per_step: int = 30

    # Episode setup
    max_decisions: int = 80

    # Physical control ranges
    a_range: Tuple[float, float] = (0.0, 100.0)
    f_range: Tuple[float, float] = (200.0, 2000.0)

    # Reference and reward anchors
    t0: float = 0.210084
    baseline_tploss: float = 0.12
    tploss_scale: float = 10.0

    # Fluent boundary/report settings
    inlet_name: str = "hole_inlet"
    report_file_stem: str = "report-def-0-rfile"
    pi: float = np.pi

    # Model I/O
    model_load_path: Optional[str] = None
    model_save_path: str = "ppo_fluent_model"

    # Output and plotting
    output_dir: str = "./outputs"
    save_plots: bool = True
    show_plots: bool = False


@dataclass
class PPOConfig:
    """PPO hyper-parameters."""

    policy: str = "MlpPolicy"
    learning_rate: float = 5e-4
    n_steps: int = 80
    batch_size: int = 20
    n_epochs: int = 5
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    net_arch: List[int] = field(default_factory=lambda: [64, 128, 64])
    verbose: int = 1


# =========================
# 2) Fluent Interaction Layer
# =========================


class FluentSessionManager:
    """Encapsulate all direct Fluent calls."""

    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg
        self.work_dir = Path(cfg.work_dir)
        self.session = pyfluent.launch_fluent(
            mode="solver",
            dimension=3,
            show_gui=cfg.show_gui,
            processor_count=cfg.processor_count,
        )
        self.session.settings.file.read_case(file_name=str(self._resolve_path(cfg.cas_path)))
        self.session.solution.run_calculation.reporting_interval = cfg.slice_len
        self.session.solution.run_calculation.profile_update_interval = cfg.slice_len

    def _resolve_path(self, file_name: str) -> Path:
        path = Path(file_name)
        if path.is_absolute():
            return path
        return self.work_dir / path

    def read_data(self) -> None:
        self.session.settings.file.read_data(file_name=str(self._resolve_path(self.cfg.data_path)))

    def set_velocity_expression(self, velocity_expr: str) -> None:
        self.session.setup.boundary_conditions.velocity_inlet[
            self.cfg.inlet_name
        ].momentum.velocity.value = velocity_expr

    def run_transient_steps(self) -> None:
        self.session.settings.solution.run_calculation.dual_time_iterate(
            time_step_count=self.cfg.slice_len,
            max_iter_per_step=self.cfg.max_iter_per_step,
        )

    def close(self) -> None:
        try:
            self.session.exit()
        except Exception:
            pass


# =========================
# 3) Data IO Layer
# =========================


class TplossReportReader:
    """Read tploss from latest Fluent report file."""

    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg
        self.report_dir = Path(cfg.work_dir)

    def read_latest_tploss(self) -> float:
        base = self.cfg.report_file_stem
        patterns = [base, f"{base}_*", f"{base}_*_*"]
        candidate_files = set()
        for pattern in patterns:
            candidate_files.update(self.report_dir.glob(pattern))
        candidate_files = [f for f in candidate_files if f.is_file()]

        if not candidate_files:
            raise FileNotFoundError(
                f"No report file found in {self.report_dir.resolve()} with patterns: {patterns}"
            )

        latest_file = max(candidate_files, key=lambda f: f.stat().st_mtime)
        table = pd.read_csv(latest_file, sep=r"\s+", skiprows=2)
        if table.empty:
            raise ValueError(f"Report file is empty: {latest_file}")

        tploss_raw = float(table.iloc[-1, 2])
        return tploss_raw * self.cfg.tploss_scale


def save_dataframe(data: List[Dict], csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


def plot_training_curves(
    step_csv: Path,
    episode_csv: Path,
    output_dir: Path,
    show_plots: bool = False,
    title_prefix: str = "",
) -> List[Path]:
    """
    Plot reward/A/F/tploss trends for quick diagnostics.

    Outputs:
    - step_trends.png: step-wise reward, A, F, tploss
    - episode_trends.png: episode reward and mean tploss
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    step_df = pd.read_csv(step_csv)
    ep_df = pd.read_csv(episode_csv)

    # Step-wise figure
    fig1, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig1.suptitle(f"{title_prefix} Step-wise Trends")

    axes[0].plot(step_df.index, step_df["reward"], color="tab:blue")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(step_df.index, step_df["A"], color="tab:orange")
    axes[1].set_ylabel("A (m/s)")
    axes[1].grid(True, alpha=0.3)

    if "F" in step_df.columns:
        axes[2].plot(step_df.index, step_df["F"], color="tab:green")
    axes[2].set_ylabel("F (1/s)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(step_df.index, step_df["tploss"], color="tab:red")
    axes[3].set_ylabel("tploss")
    axes[3].set_xlabel("Global Step Index")
    axes[3].grid(True, alpha=0.3)

    step_plot = output_dir / "step_trends.png"
    fig1.tight_layout()
    fig1.savefig(step_plot, dpi=180)
    generated.append(step_plot)

    # Episode-wise figure
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig2.suptitle(f"{title_prefix} Episode-wise Trends")

    axes2[0].plot(ep_df["episode_id"], ep_df["episode_reward_sum"], marker="o", color="tab:blue")
    axes2[0].set_ylabel("Episode Reward Sum")
    axes2[0].grid(True, alpha=0.3)

    axes2[1].plot(ep_df["episode_id"], ep_df["episode_tploss_mean"], marker="o", color="tab:red")
    axes2[1].set_ylabel("Mean tploss")
    axes2[1].set_xlabel("Episode ID")
    axes2[1].grid(True, alpha=0.3)

    episode_plot = output_dir / "episode_trends.png"
    fig2.tight_layout()
    fig2.savefig(episode_plot, dpi=180)
    generated.append(episode_plot)

    if show_plots:
        plt.show()

    plt.close(fig1)
    plt.close(fig2)
    return generated


# =========================
# 4) Reward Layer
# =========================


class RewardCalculator:
    """Keep reward formula explicit and easy to modify."""

    def __init__(self, baseline_tploss: float, tploss_scale: float) -> None:
        self.baseline_scaled = baseline_tploss * tploss_scale

    def compute_reward(self, tploss_now: float) -> float:
        # Current logic: reward = baseline - current tploss
        # This keeps the original behavior and is intentionally simple.
        return self.baseline_scaled - tploss_now


# =========================
# 5) RL Environment Layer
# =========================


class FluentJetEnv(gym.Env):
    """Gym env for RL actuation with Fluent transient simulation."""

    metadata = {"render.modes": []}

    def __init__(self, cfg: SimulationConfig):
        super().__init__()
        self.cfg = cfg

        self.fluent = FluentSessionManager(cfg)
        self.report_reader = TplossReportReader(cfg)
        self.reward_calculator = RewardCalculator(cfg.baseline_tploss, cfg.tploss_scale)

        if cfg.action_dims not in (1, 2):
            raise ValueError("action_dims must be 1 or 2.")

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(cfg.action_dims,),
            dtype=np.float32,
        )

        if cfg.action_dims == 1:
            obs_low = np.array([0.0, -np.inf, 0.0], dtype=np.float32)
            obs_high = np.array([1.0, np.inf, 1.0], dtype=np.float32)
        else:
            obs_low = np.array([0.0, 0.0, -np.inf, 0.0], dtype=np.float32)
            obs_high = np.array([1.0, 1.0, np.inf, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Runtime states
        self.current_A = 0.0
        self.current_F = cfg.f_range[0]
        self.current_velocity_expr = "0[m s^-1]"
        self.last_action_norm = np.zeros(cfg.action_dims, dtype=np.float32)
        self.last_tploss = cfg.baseline_tploss * cfg.tploss_scale

        # Episode counters
        self.episode_id = -1
        self.decision_count = 0
        self.phase = "train"

        # Episode accumulators
        self.episode_reward_sum = 0.0
        self.episode_tploss_values: List[float] = []

        # Full logs (train + test)
        self.step_records: List[Dict] = []
        self.episode_records: List[Dict] = []

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def _map_action_to_physical(self, action: np.ndarray) -> Tuple[float, float, float, float]:
        a_norm = float(np.clip(action[0], 0.0, 1.0))
        A = self.cfg.a_range[0] + a_norm * (self.cfg.a_range[1] - self.cfg.a_range[0])

        if self.cfg.action_dims == 2:
            f_norm = float(np.clip(action[1], 0.0, 1.0))
            F = self.cfg.f_range[0] + f_norm * (self.cfg.f_range[1] - self.cfg.f_range[0])
        else:
            f_norm = 0.0
            F = self.cfg.f_range[0]

        return float(A), float(F), a_norm, f_norm

    def _build_velocity_expression(self, A: float, F: float) -> str:
        if self.cfg.action_dims == 2:
            return (
                f"{A:.2f}[m s^-1]*sin(2*{self.cfg.pi:.8f}*{F:.2f} [s^-1]"
                f"*(t - {self.cfg.t0:.6f}[s]))"
            )
        return f"{A:.2f}[m s^-1]"

    def _build_observation(self, a_norm: float, f_norm: float, tploss_now: float) -> np.ndarray:
        progress = self.decision_count / self.cfg.max_decisions
        if self.cfg.action_dims == 1:
            return np.array([a_norm, tploss_now, progress], dtype=np.float32)
        return np.array([a_norm, f_norm, tploss_now, progress], dtype=np.float32)

    def reset(self):
        self.fluent.read_data()

        self.episode_id += 1
        self.decision_count = 0
        self.episode_reward_sum = 0.0
        self.episode_tploss_values = []

        # Reset to minimum control values
        self.current_A = self.cfg.a_range[0]
        self.current_F = self.cfg.f_range[0]
        self.current_velocity_expr = self._build_velocity_expression(self.current_A, self.current_F)
        self.fluent.set_velocity_expression(self.current_velocity_expr)

        tploss_now = self.cfg.baseline_tploss * self.cfg.tploss_scale
        self.last_tploss = tploss_now
        self.last_action_norm = np.zeros(self.cfg.action_dims, dtype=np.float32)

        return self._build_observation(0.0, 0.0, tploss_now)

    def step(self, action):
        self.decision_count += 1

        # 1) Action mapping
        A, F, a_norm, f_norm = self._map_action_to_physical(np.asarray(action))

        # 2) Apply boundary condition
        velocity_expr = self._build_velocity_expression(A, F)
        self.fluent.set_velocity_expression(velocity_expr)

        # 3) Advance Fluent and read monitor
        self.fluent.run_transient_steps()
        tploss_now = self.report_reader.read_latest_tploss()

        # 4) Reward and termination
        reward = self.reward_calculator.compute_reward(tploss_now)
        done = self.decision_count >= self.cfg.max_decisions

        # 5) Update runtime state
        self.current_A = A
        self.current_F = F
        self.current_velocity_expr = velocity_expr
        self.last_action_norm = np.array(action, dtype=np.float32)
        self.last_tploss = tploss_now
        self.episode_reward_sum += reward
        self.episode_tploss_values.append(tploss_now)

        obs = self._build_observation(a_norm, f_norm, tploss_now)
        info = {
            "phase": self.phase,
            "episode_id": self.episode_id,
            "decision_step": self.decision_count,
            "A": A,
            "F": F,
            "a_norm": a_norm,
            "f_norm": f_norm,
            "tploss": tploss_now,
            "reward": reward,
            "velocity_expr": velocity_expr,
        }

        self.step_records.append(info.copy())

        if done:
            episode_info = {
                "phase": self.phase,
                "episode_id": self.episode_id,
                "decision_steps": self.decision_count,
                "episode_reward_sum": self.episode_reward_sum,
                "episode_reward_mean": self.episode_reward_sum / self.decision_count,
                "episode_tploss_mean": float(np.mean(self.episode_tploss_values)),
                "episode_tploss_min": float(np.min(self.episode_tploss_values)),
                "episode_tploss_max": float(np.max(self.episode_tploss_values)),
            }
            self.episode_records.append(episode_info)
            info.update(episode_info)

        print(
            f"[{self.phase}] ep={self.episode_id} step={self.decision_count} "
            f"A={A:.3f} F={F:.3f} tploss={tploss_now:.6f} reward={reward:.6f}"
        )
        return obs, float(reward), bool(done), info

    def save_logs(self, output_dir: str) -> Tuple[Path, Path]:
        output_path = Path(output_dir)
        step_file = save_dataframe(
            self.step_records,
            output_path / f"{self.cfg.case_name}_step_records.csv",
        )
        episode_file = save_dataframe(
            self.episode_records,
            output_path / f"{self.cfg.case_name}_episode_records.csv",
        )
        return step_file, episode_file

    def close(self):
        self.fluent.close()


# =========================
# 6) Training/Testing Helpers
# =========================


class TrainTraceCallback(BaseCallback):
    """Reserved callback hook for future custom metrics."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True


def build_model(env: DummyVecEnv, cfg: PPOConfig) -> PPO:
    return PPO(
        cfg.policy,
        env,
        verbose=cfg.verbose,
        policy_kwargs={"net_arch": cfg.net_arch},
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        clip_range=cfg.clip_range,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
    )


def run_test(model: PPO, env: DummyVecEnv, n_episodes: int) -> None:
    base_env: FluentJetEnv = env.envs[0]
    base_env.set_phase("test")

    for _ in range(n_episodes):
        obs = env.reset()
        done = [False]
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)


def train_and_test(
    sim_cfg: SimulationConfig,
    ppo_cfg: PPOConfig,
    train_timesteps: int,
    test_episodes: int = 1,
) -> Tuple[Path, Path, List[Path]]:
    """
    Main entry for one experiment.

    Returns:
    - step log csv path
    - episode log csv path
    - generated plot paths
    """
    vec_env = DummyVecEnv([lambda: FluentJetEnv(sim_cfg)])
    base_env: FluentJetEnv = vec_env.envs[0]

    if sim_cfg.model_load_path:
        model = PPO.load(sim_cfg.model_load_path, env=vec_env)
    else:
        model = build_model(vec_env, ppo_cfg)

    base_env.set_phase("train")
    model.learn(total_timesteps=train_timesteps, callback=TrainTraceCallback())
    model.save(sim_cfg.model_save_path)

    run_test(model, vec_env, n_episodes=test_episodes)

    step_file, episode_file = base_env.save_logs(sim_cfg.output_dir)

    plot_files: List[Path] = []
    if sim_cfg.save_plots:
        plot_files = plot_training_curves(
            step_csv=step_file,
            episode_csv=episode_file,
            output_dir=Path(sim_cfg.output_dir),
            show_plots=sim_cfg.show_plots,
            title_prefix=sim_cfg.case_name,
        )

    vec_env.close()
    return step_file, episode_file, plot_files


def evaluate_saved_model(
    sim_cfg: SimulationConfig,
    model_path: str,
    test_episodes: int = 1,
) -> Tuple[Path, Path, List[Path]]:
    """
    Independent evaluation utility.

    It only runs test inference with a pre-trained model and outputs
    step/episode logs + plots for reward, A/F, and tploss trends.
    """
    eval_cfg = SimulationConfig(**vars(sim_cfg))
    eval_cfg.model_load_path = model_path

    vec_env = DummyVecEnv([lambda: FluentJetEnv(eval_cfg)])
    base_env: FluentJetEnv = vec_env.envs[0]

    model = PPO.load(model_path, env=vec_env)
    run_test(model, vec_env, n_episodes=test_episodes)

    step_file, episode_file = base_env.save_logs(eval_cfg.output_dir)

    plot_files: List[Path] = []
    if eval_cfg.save_plots:
        plot_files = plot_training_curves(
            step_csv=step_file,
            episode_csv=episode_file,
            output_dir=Path(eval_cfg.output_dir),
            show_plots=eval_cfg.show_plots,
            title_prefix=f"{eval_cfg.case_name}_eval",
        )

    vec_env.close()
    return step_file, episode_file, plot_files


# =========================
# Case Presets
# =========================


def make_case3_config() -> SimulationConfig:
    return SimulationConfig(
        case_name="case3_no_sin",
        work_dir=r"D:\LYZ\A case3 with hole jet ag70",
        cas_path="70 angle unsteady_jet0_dt=0.03s.cas.h5",
        data_path="70 angle unsteady_jet0_dt=0.03s.dat.h5",
        action_dims=1,
        a_range=(0.0, 200.0),
        max_decisions=80,
        baseline_tploss=0.07,
        t0=0.210084,
        model_load_path="my_model_nosin4",
        model_save_path="my_model_nosin5",
        output_dir="./outputs_case3",
    )


def make_case4_config() -> SimulationConfig:
    return SimulationConfig(
        case_name="case4_no_sin",
        work_dir=r"D:\LYZ\A case4 with hole jet ag70",
        cas_path="double_sided 70 angle steady_jet0_dt=0.0115s.cas.h5",
        data_path="./double_sided 70 angle steady_jet0_dt=0.0115s.dat.h5",
        action_dims=2,
        a_range=(0.0, 100.0),
        f_range=(200.0, 2000.0),
        max_decisions=100,
        baseline_tploss=0.12,
        t0=0.210084,
        model_load_path="my_model_nosin4",
        model_save_path="my_model_nosin5",
        output_dir="./outputs_case4",
    )

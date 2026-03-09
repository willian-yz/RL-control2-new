"""
Post-processing utilities for RL + Fluent experiments.

This script is independent from training entry scripts and supports:
1) Analyze existing CSV logs and regenerate trend plots.
2) Load a trained RL model, run test episodes, and save logs + plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from fluent_rl_framework import (
    SimulationConfig,
    evaluate_saved_model,
    make_case3_config,
    make_case4_config,
    plot_training_curves,
)


# =========================
# 1) Config Selection
# =========================


def get_case_config(case_name: str) -> SimulationConfig:
    """Return case config preset for post-processing."""
    if case_name == "case3":
        return make_case3_config()
    if case_name == "case4":
        return make_case4_config()
    raise ValueError("case_name must be 'case3' or 'case4'.")


# =========================
# 2) Post-process from Existing Logs
# =========================


def postprocess_from_logs(
    step_csv: str,
    episode_csv: str,
    output_dir: str,
    show_plots: bool = False,
    title_prefix: str = "postprocess",
) -> List[Path]:
    """
    Build trend plots from existing logs.

    Required columns in step csv:
    reward, A, F, tploss
    Required columns in episode csv:
    episode_id, episode_reward_sum, episode_tploss_mean
    """
    return plot_training_curves(
        step_csv=Path(step_csv),
        episode_csv=Path(episode_csv),
        output_dir=Path(output_dir),
        show_plots=show_plots,
        title_prefix=title_prefix,
    )


# =========================
# 3) Post-process by Loading Trained Model
# =========================


def postprocess_from_model(
    case_name: str,
    model_path: str,
    test_episodes: int,
    output_dir: str,
    show_plots: bool,
) -> Tuple[Path, Path, List[Path]]:
    """
    Load a trained model, run deterministic test episodes, and export:
    - step log csv
    - episode log csv
    - trend plots for reward/A/F/tploss
    """
    sim_cfg = get_case_config(case_name)
    sim_cfg.output_dir = output_dir
    sim_cfg.show_plots = show_plots
    sim_cfg.save_plots = True

    return evaluate_saved_model(
        sim_cfg=sim_cfg,
        model_path=model_path,
        test_episodes=test_episodes,
    )


# =========================
# 4) CLI Entry
# =========================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-process RL + Fluent results")

    parser.add_argument(
        "--mode",
        choices=["from_logs", "from_model"],
        required=True,
        help="from_logs: only read CSV and plot; from_model: load RL model and run test",
    )

    parser.add_argument("--case", choices=["case3", "case4"], default="case4")
    parser.add_argument("--model-path", type=str, default="my_model_nosin5")
    parser.add_argument("--test-episodes", type=int, default=1)

    parser.add_argument("--step-csv", type=str, default="")
    parser.add_argument("--episode-csv", type=str, default="")

    parser.add_argument("--output-dir", type=str, default="./postprocess_outputs")
    parser.add_argument("--show-plots", action="store_true")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "from_logs":
        if not args.step_csv or not args.episode_csv:
            raise ValueError("from_logs mode requires --step-csv and --episode-csv")

        plot_files = postprocess_from_logs(
            step_csv=args.step_csv,
            episode_csv=args.episode_csv,
            output_dir=args.output_dir,
            show_plots=args.show_plots,
            title_prefix=f"{args.case}_from_logs",
        )
        for file_path in plot_files:
            print(f"Saved plot: {file_path}")
        return

    step_file, episode_file, plot_files = postprocess_from_model(
        case_name=args.case,
        model_path=args.model_path,
        test_episodes=args.test_episodes,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
    )

    print(f"Saved step log: {step_file}")
    print(f"Saved episode log: {episode_file}")
    for file_path in plot_files:
        print(f"Saved plot: {file_path}")


if __name__ == "__main__":
    main()

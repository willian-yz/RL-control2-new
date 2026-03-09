"""Case 3 runner based on the unified RL+Fluent framework."""

from rl_fluent_framework import EnvConfig, RLConfig, ExperimentManager


def build_case3_configs() -> tuple[EnvConfig, RLConfig]:
    env_config = EnvConfig(
        case_id=3,
        cas_path="70 angle unsteady_jet0_dt=0.03s.cas.h5",
        data_path="70 angle unsteady_jet0_dt=0.03s.dat.h5",
        workdir=r"D:\LYZ\case 3 test",
        max_decisions=80,
        initial_tploss=0.07,
    )
    rl_config = RLConfig(
        action_dim=1,
        amplitude_range=(0.0, 200.0),
        baseline_tploss=0.07,
        expr_mode="constant",
        n_steps=80,
    )
    return env_config, rl_config


def main() -> None:
    env_config, rl_config = build_case3_configs()
    manager = ExperimentManager(env_config, rl_config)

    #model_path = manager.train(train_steps=100 * 10, load_model_path=None)
    #print(f"Saved model: {model_path}")
    #plot_path_train = manager.plot_history(mode="train")
    #print(f"Train history plot: {plot_path_train}")

    rewards = manager.test(model_path=str(r'D:\LYZ\case 3 test\my_model_nosin5.zip'), tot_steps=1)
    plot_path_test = manager.plot_history(mode="test")
    print(f"Test history plot: {plot_path_test}")
    print(f"Test episode rewards: {rewards}")


if __name__ == "__main__":
    main()

"""Case 4 runner based on the unified RL+Fluent framework."""

from rl_fluent_framework import EnvConfig, RLConfig, ExperimentManager


def build_case4_configs() -> tuple[EnvConfig, RLConfig]:
    env_config = EnvConfig(
        case_id=4,
        cas_path="double_sided 70 angle steady_jet0_dt=0.0115s.cas.h5",
        data_path="./double_sided 70 angle steady_jet0_dt=0.0115s.dat.h5",
        workdir=r"D:\LYZ\case 4 test",
        max_decisions=100, #100
        initial_tploss=0.12,
    )
    rl_config = RLConfig(
        action_dim=2,
        amplitude_range=(0.0, 100.0),
        frequency_range=(200.0, 2000.0),
        baseline_tploss=0.12,
        expr_mode="sin",
        n_steps=50, #80 #2步训练一次
        n_epochs=5, #参数更新5次
    )
    return env_config, rl_config


def main() -> None:
    env_config, rl_config = build_case4_configs()
    manager = ExperimentManager(env_config, rl_config)

    # model_path = manager.train(train_steps=200, load_model_path=r'D:\LYZ\case 4 test\my_model_nosin4.zip')
    # print(f"Saved model: {model_path}")
    # plot_path_train = manager.plot_history(mode="train")
    # print(f"Train history plot: {plot_path_train}")

    rewards = manager.test(model_path=r'D:\LYZ\case 4 test\artifacts\my_model_case4_step1200.zip', tot_steps=80)
    plot_path_test = manager.plot_history(mode="test")
    print(f"Test episode rewards: {rewards}")
    print(f"Test history plot: {plot_path_test}")


if __name__ == "__main__":
    main()

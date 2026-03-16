"""Case 4 runner based on the unified RL+Fluent framework."""

from rl_fluent_framework_2d import EnvConfig, RLConfig, ExperimentManager


def build_case5_configs() -> tuple[EnvConfig, RLConfig]:
    env_config = EnvConfig(
        case_id=5,
        cas_path="i8.5 A=0 w=0.055.cas.h5",
        data_path="i8.5 A=0 w=0.055.dat.h5",
        workdir=r"D:\LYZ\A 2D case4",
        max_decisions=100, #100
        initial_tploss=0.055,
        processor_count = 4,
    )
    rl_config = RLConfig(
        action_dim=1,
        amplitude_range=(-100.0, 200.0),
        baseline_tploss=0.055,
        expr_mode="constant",
        n_steps=50, #80 #2步训练一次
        n_epochs=5, #参数更新5次
    )
    return env_config, rl_config


def main() -> None:
    env_config, rl_config = build_case5_configs()
    manager = ExperimentManager(env_config, rl_config)

    model_path = manager.train(train_steps=100*30)
    print(f"Saved model: {model_path}")
    plot_path_train = manager.plot_history(mode="train")
    print(f"Train history plot: {plot_path_train}")

    # rewards = manager.test(model_path=r'D:\LYZ\case 4 test\artifacts\my_model_case4_step1200.zip', tot_steps=80)
    # plot_path_test = manager.plot_history(mode="test")
    # print(f"Test episode rewards: {rewards}")
    # print(f"Test history plot: {plot_path_test}")


if __name__ == "__main__":
    main()

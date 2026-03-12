"""Case 4 runner based on the unified RL+Fluent framework."""

from rl_fluent_framework import EnvConfig, RLConfig, ExperimentManager


def build_case4_configs() -> tuple[EnvConfig, RLConfig]:
    env_config = EnvConfig(
        case_id=4,
        cas_path="double_sided 70 angle steady_jet0_dt=0.0115s.cas.h5",
        data_path="./double_sided 70 angle steady_jet0_dt=0.0115s.dat.h5",
        workdir=r"D:\LYZ\A case 4 no sin with hole jet ag70",
        max_decisions=100, #100
        initial_tploss=0.12,
    )
    rl_config = RLConfig(
        action_dim=1,
        amplitude_range=(-100.0, 200.0),
        baseline_tploss=0.12,
        expr_mode="sin",
        n_steps=50, #40
        n_epochs=5,  # 参数更新5次
    )
    return env_config, rl_config


def main() -> None:
    env_config, rl_config = build_case4_configs()
    manager = ExperimentManager(env_config, rl_config)

    model_path = manager.train(train_steps=200*4,load_model_path=r'D:\LYZ\A case 4 no sin with hole jet ag70\artifacts\my_model_case4_step400.zip')
    print(f"Saved model: {model_path}")
    plot_path_train = manager.plot_history(mode="train")
    print(f"Train history plot: {plot_path_train}")

    # rewards = manager.test(model_path=r'D:\LYZ\A case 4 no sin with hole jet ag70\artifacts\my_model_case4_step400.zip', tot_steps=80)
    # plot_path_test = manager.plot_history(mode="test")
    # print(f"Test episode rewards: {rewards}")
    # print(f"Test history plot: {plot_path_test}")


if __name__ == "__main__":
    main()

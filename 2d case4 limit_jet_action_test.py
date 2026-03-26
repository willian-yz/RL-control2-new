"""Case 4 runner based on the unified RL+Fluent framework."""

from rl_fluent_framework_2d_limit_action  import EnvConfig, RLConfig, ExperimentManager


def build_case5_configs() -> tuple[EnvConfig, RLConfig]:
    env_config = EnvConfig(
        case_id=6,
        cas_path="unsteady_rl.cas",
        data_path="unsteady_rl.dat",
        workdir=r"E:\A keti\Case\50theta_b_t1.6_bow20_M0.5_i+8.5_2D_change inlet large_ RL_test\iteration_13",
        max_decisions=250, #100
        slice_len = 40, # dt=5e-6s, 每个2e-4s决策一次, 250是因为冲角变化f=20hz
        initial_tploss=0.05,
        processor_count = 4,
        rake_names=("rake-1", "rake-2", "rake-3", "rake-4", "rake-5", "rake-6", "rake-7", "rake-8", "rake-9"),
        rake_point_count=60,
    )
    rl_config = RLConfig(
        action_dim=2,
        amplitude_range=(-100.0, 250.0),
        baseline_tploss=0.05,
        max_delta_amplitude = 20.0, #修改后
        use_delta_action=True,      #修改后
        expr_mode="constant",
        n_steps=125, #80 #2步训练一次
        batch_size=25,
        n_epochs=5, #参数更新5次
    )
    return env_config, rl_config


def main() -> None:
    env_config, rl_config = build_case5_configs()
    manager = ExperimentManager(env_config, rl_config)

    # model_path = manager.train(train_steps=250*20)
    # print(f"Saved model: {model_path}")
    # plot_path_train = manager.plot_history(mode="train")
    # print(f"Train history plot: {plot_path_train}")

    rewards = manager.test(model_path=r'E:\A keti\Case\50theta_b_t1.6_bow20_M0.5_i+8.5_2D_change inlet large_ RL_test\iteration_13\my_model_case6_step3250.zip', tot_steps=250)
    plot_path_test = manager.plot_history(mode="test")
    print(f"Test episode rewards: {rewards}")
    print(f"Test history plot: {plot_path_test}")


if __name__ == "__main__":
    main()

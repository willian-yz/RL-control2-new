"""Case 4 runner based on the unified RL+Fluent framework."""

from rl_fluent_framework_2d_limit_action_reward_change_2  import EnvConfig, RLConfig, ExperimentManager
import numpy as np


def build_case5_configs() -> tuple[EnvConfig, RLConfig]:
    env_config = EnvConfig(
        case_id=7,
        cas_path="jet_settings_i=11.cas",
        data_path="jet_settings_i=11.dat",
        workdir=r"E:\A keti\Case\mesh\50theta_b_t1.6\15 and 60 1.5b_2b_io",
        max_decisions=60, #100
        slice_len = 30, # dt=5e-6s, 每30dt决策一次, 冲角变化f=167hz，决策2周期
        initial_tploss = np.array([0.12023581, 0.11958303, 0.11902861, 0.11865546, 0.11849607,
       0.11857138, 0.11890123, 0.119453  , 0.12013197, 0.12084306,
       0.12149273, 0.12199621, 0.12232298, 0.12245072, 0.12237743,
       0.12209946, 0.12164915, 0.12109803, 0.12053154, 0.12001236,
       0.11961707, 0.11939027, 0.11934691, 0.11949423, 0.11983936,
       0.12031188, 0.12083819, 0.12133276, 0.12171283, 0.12194395,
       0.12201158, 0.12189482, 0.12158998, 0.12113735, 0.12060359,
       0.12006178, 0.11957217, 0.11920797, 0.11900271, 0.11898098,
       0.1191502 , 0.11951922, 0.12000738, 0.12054273, 0.1210556 ,
       0.12145407, 0.12171178, 0.12181168, 0.12173892, 0.12149392,
       0.12109944, 0.1206171 , 0.12012228, 0.11966726, 0.11932182,
       0.11912126, 0.11909269, 0.11924586, 0.11958641, 0.12005594]),
        processor_count = 6,
        rake_names=("rake-1", "rake-2", "rake-3", "rake-4", "rake-5", "rake-6", "rake-7", "rake-8", "rake-9"),
        rake_point_count=60, #60
    )
    rl_config = RLConfig(
        action_dim=2,
        amplitude_range=(-100.0, 250.0),
        max_delta_amplitude = 20.0, #修改后
        use_delta_action=True,      #修改后
        expr_mode="constant",
        n_steps=30, #80 #2步训练一次
        batch_size=15,
        n_epochs=5, #参数更新5次
        policy_kwargs={"net_arch": [128, 256, 128]}, # 网络结构
        learning_rate= 2e-4
    )
    return env_config, rl_config


def main() -> None:
    env_config, rl_config = build_case5_configs()
    manager = ExperimentManager(env_config, rl_config)

    model_path = manager.train(train_steps=250*20)
    print(f"Saved model: {model_path}")
    plot_path_train = manager.plot_history(mode="train")
    print(f"Train history plot: {plot_path_train}")

    # rewards = manager.test(model_path=r'E:\A keti\Case\2D RL\my_model_case7_step12250.zip', tot_steps=100)
    # plot_path_test = manager.plot_history(mode="test")
    # print(f"Test episode rewards: {rewards}")
    # print(f"Test history plot: {plot_path_test}")


if __name__ == "__main__":
    main()

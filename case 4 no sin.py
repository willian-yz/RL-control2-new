from fluent_rl_framework import PPOConfig, make_case4_config, train_and_test


if __name__ == "__main__":
    # === Case selection ===
    sim_cfg = make_case4_config()

    # === Optional overrides ===
    # sim_cfg.t0 = 0.210084
    # sim_cfg.baseline_tploss = 0.12
    # sim_cfg.action_dims = 2  # 1: control A only; 2: control A and F
    # sim_cfg.show_plots = True

    ppo_cfg = PPOConfig(n_steps=40)

    step_file, episode_file, plot_files = train_and_test(
        sim_cfg=sim_cfg,
        ppo_cfg=ppo_cfg,
        train_timesteps=100 * 10,
        test_episodes=1,
    )

    print(f"Saved step log to: {step_file}")
    print(f"Saved episode log to: {episode_file}")
    for plot_file in plot_files:
        print(f"Saved plot to: {plot_file}")

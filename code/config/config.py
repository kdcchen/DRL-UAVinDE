ENV_CONFIG = {
    "dt": 0.1,
    "max_speed": 1.0,
    "max_accel": 0.5,
    "wind": True,
    "obstacle": True,
    "map_size": 5.0,
    "num_obstacles": [2, 5],
    "obstacle_speed_range": [0.1, 0.4],
    "obstacle_radius": 0.3,
    "wind_std": 0.1,
}

PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "total_timesteps": 100_000,
    "save_freq": 5000,
    "device": "cuda",  # or "cpu"
}

PATHS = {
    "checkpoint_path": "./models/",
    "final_model": "./models/ppo_uav_final",
}

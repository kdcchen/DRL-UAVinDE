ENV_CONFIG = {
    "dt": 0.1,
    "max_speed": 1.0,
    "max_accel": 1.0,
    "wind": True,
    "obstacle": True,
    "obstacle_k": 5,
    "dynamicObstacle_rate": 0.0,
    "map_size": 5.0,
    "num_obstacles": [6, 8],
    "obstacle_speed_range": [0.1, 0.4],
    "obstacle_radius": 0.3,
    "safety_radius": 1.0,
    "wind_std": 0.1,
    "obstacle_layout": "random",
}

PPO_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 1024,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "total_timesteps": 100_000,
    "save_freq": 5000,
    "device": "cuda",
}

PATHS = {
    "checkpoint_path": "./models/",
    "final_model": "./models/ppo_uav_final",
}

STAGE_CONFIGS = {
    1: {
        "name": "stage1_empty",
        "steps": 150_000,
        "config": {
            "wind": False,
            "obstacle": False,
            "dynamicObstacle_rate": 0.0,
            "num_obstacles": [0, 0],
        },
    },
    2: {
        "name": "stage2_far_obs",
        "steps": 150_000,
        "config": {
            "wind": False,
            "obstacle": True,
            "obstacle_layout": "far_single",
            "dynamicObstacle_rate": 0.0,
            "num_obstacles": [1, 1],
        },
    },
    3: {
        "name": "stage3_near_obs",
        "steps": 250_000,
        "config": {
            "wind": False,
            "obstacle": True,
            "obstacle_layout": "near_single",
            "dynamicObstacle_rate": 0.0,
            "num_obstacles": [1, 1],
        },
    },
    4: {
        "name": "stage4_blocking_obs",
        "steps": 300_000,
        "config": {
            "wind": False,
            "obstacle": True,
            "obstacle_layout": "blocking_single",
            "dynamicObstacle_rate": 0.0,
            "num_obstacles": [1, 1],
        },
    },
    5: {
        "name": "stage5_dynamic",
        "steps": 500_000,
        "config": {
            "wind": False,
            "obstacle": True,
            "obstacle_layout": "random",
            "dynamicObstacle_rate": 0.3,
            "num_obstacles": [4, 6],
        },
    },
    6: {
        "name": "stage6_windy",
        "steps": 500_000,
        "config": {
            "wind": True,
            "obstacle": True,
            "obstacle_layout": "random",
            "dynamicObstacle_rate": 0.3,
            "num_obstacles": [6, 8],
        },
    },
}

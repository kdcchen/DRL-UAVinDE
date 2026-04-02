import sys,os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from uav_env.uav_env import UAVEnv
from config.config import ENV_CONFIG,PPO_CONFIG,PATHS


env = DummyVecEnv([lambda:UAVEnv(ENV_CONFIG)])

model = PPO("MlpPolicy",
    env,
    learning_rate=PPO_CONFIG["learning_rate"],
    n_steps=PPO_CONFIG["n_steps"],
    batch_size=PPO_CONFIG["batch_size"],
    gamma=PPO_CONFIG["gamma"],
    gae_lambda=PPO_CONFIG["gae_lambda"],
    clip_range=PPO_CONFIG["clip_range"],
    ent_coef=PPO_CONFIG["ent_coef"],
    device=PPO_CONFIG["device"],
    verbose=1
)

checkpoint_callback = CheckpointCallback(
    save_freq=PPO_CONFIG["save_freq"],
    save_path=PATHS["checkpoint_path"],
    name_prefix="PPO_UAV",
)

model.learn(total_timesteps=PPO_CONFIG["total_timesteps"],
            callback=checkpoint_callback
)

model.save(PATHS["final_model"])

env.close()
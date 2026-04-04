import os
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import ENV_CONFIG, PATHS
from uav_env.uav_env import UAVEnv


def eval_model(model_path, episodes=1):
    env = DummyVecEnv([lambda: UAVEnv(ENV_CONFIG)])

    model = PPO.load(model_path)

    for epi in range(episodes):
        obs = env.reset()
        done = False
        trajectory = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            done = done[0]

            if done:
                terminal_obs = info[0]["terminal_observation"]
                x, y = terminal_obs[0], terminal_obs[1]
            else:
                x, y = obs[0][0], obs[0][1]

            trajectory.append((x, y))

        trajectory = np.array(trajectory)
        np.save(f"models/trajectory_epi{epi}.npy", trajectory)
        print(f"Episode {epi} done")

    env.close()


if __name__ == "__main__":
    eval_model(PATHS["final_model"])

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.uav_env import UAVEnv
from config.config import ENV_CONFIG, PATHS

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

            x,y = obs[0][0],obs[0][1]
            trajectory.append((x,y))

        trajectory = np.array(trajectory)
        np.save(f"trajectory_epi{epi}.npy", trajectory)
        print(f"Episode {epi} done")

    env.close()

if __name__ == "__main__":
    eval_model(PATHS["model"])
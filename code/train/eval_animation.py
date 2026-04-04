import os
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy
import numpy as np
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import ENV_CONFIG, PATHS
from uav_env.uav_env import UAVEnv


def animate_epi(model_path, max_epi=300):
    env = DummyVecEnv([lambda: UAVEnv(ENV_CONFIG)])
    model = PPO.load(model_path)

    obs = env.reset()
    done = False

    fig, ax = plt.subplots(figsize=(6, 6))
    map_size = env.envs[0].map_size
    ax.set_xlim(-map_size, map_size)
    ax.set_ylim(-map_size, map_size)
    ax.grid(True)

    # UAV
    (uav_point,) = ax.plot([], [], "bo", markersize=6, label="UAV")

    # The trajectory line that UAV runs
    traj_x, traj_y = [], []
    (traj_line,) = ax.plot([], [], "b-", linewidth=1)

    true_goal = env.envs[0].goal
    goal_x, goal_y = true_goal[0], true_goal[1]
    goal_point = ax.scatter(goal_x, goal_y, c="green", s=80, label="Goal")

    obstacles_scatter = ax.scatter([], [], c="red", s=80, label="Obstacles")

    ax.legend()

    def update_animation(frame):
        nonlocal obs, done

        if done:
            animation.event_source.stop()
            return traj_line, uav_point, obstacles_scatter

        action, _ = model.predict(obs, deterministic=False)
        prev_state = env.envs[0].state.copy()
        next_obs, reward, dones, info = env.step(action)
        done = dones[0]

        if done:
            x, y = prev_state[0], prev_state[1]
            traj_x.append(x)
            traj_y.append(y)
            traj_line.set_data(traj_x, traj_y)
            uav_point.set_data([x], [y])
            animation.event_source.stop()
            return traj_line, uav_point, obstacles_scatter

        true_state = env.envs[0].state
        x, y = true_state[0], true_state[1]
        traj_x.append(x)
        traj_y.append(y)
        traj_line.set_data(traj_x, traj_y)
        uav_point.set_data([x], [y])

        obstacles = env.envs[0].obstacles
        obstacle_positions = np.array([obstacle["pos"] for obstacle in obstacles]).reshape(-1, 2)
        obstacles_scatter.set_offsets(obstacle_positions)

        return traj_line, uav_point, obstacles_scatter

    animation = FuncAnimation(
        fig, update_animation, frames=max_epi, interval=10, blit=False
    )
    plt.show()


if __name__ == "__main__":
    animate_epi(PATHS["final_model"])

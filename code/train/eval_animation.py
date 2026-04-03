import gymnasium as gym
import numpy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uav_env.uav_env import UAVEnv
from config.config import ENV_CONFIG, PATHS

def animate_epi(model_path, max_epi=300):
    env = DummyVecEnv([lambda: UAVEnv(ENV_CONFIG)])
    model = PPO.load(model_path)

    obs = env.reset()
    done = False

    fig,ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.grid(True)

    # UAV
    uav_point, = ax.plot([],[],'bo',markersize=6, label='UAV')

    # The trajectory line that UAV runs
    traj_x, traj_y = [], []
    traj_line, = ax.plot([],[],'b-',linewidth=1)

    goal = ENV_CONFIG['goal']
    ax.scatter(goal[0], goal[1], c='green', s=80, label='Goal')

    obstacles_scatter = ax.scatter([],[],c='red', s=80, label='Obstacles')

    ax.legend()

    def update_animation(frame):
        nonlocal obs,done

        if done:
            animation.event_source.stop()
            return traj_line, uav_point, obstacles_scatter

        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, dones, info = env.step(action)
        done = dones[0]

        if done:
            x, y = obs[0][0], obs[0][1]
            traj_x.append(x)
            traj_y.append(y)
            traj_line.set_data(traj_x, traj_y)
            uav_point.set_data([x], [y])
            animation.event_source.stop()
            return traj_line, uav_point, obstacles_scatter

        obs = next_obs
        x, y = obs[0][0], obs[0][1]
        traj_x.append(x)
        traj_y.append(y)
        traj_line.set_data(traj_x, traj_y)
        uav_point.set_data([x],[y])

        obstacles = env.envs[0].obstacles
        obstacle_positions = np.array([obstacle["pos"] for obstacle in obstacles])
        print("obstacles =", obstacles)
        print("obstacle_positions =", obstacle_positions)
        obstacles_scatter.set_offsets(obstacle_positions)

        return traj_line, uav_point, obstacles_scatter

    animation = FuncAnimation(fig, update_animation, frames=max_epi, interval=10, blit=False)
    plt.show()

if __name__ == '__main__':
    animate_epi(PATHS["final_model"])
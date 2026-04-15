import argparse
import os
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import ENV_CONFIG
from uav_env.uav_env import UAVEnv

STAGE_CONFIGS = {
    1: {
        "name": "stage1_easy",
        "config": {
            "wind": False,
            "obstacle": False,
            "dynamicObstacle_rate": 0.0,
            "num_obstacles": [0, 0],
        },
    },
    2: {
        "name": "stage2_medium",
        "config": {
            "wind": False,
            "obstacle": True,
            "dynamicObstacle_rate": 0.0,
            "num_obstacles": [2, 4],
        },
    },
    3: {
        "name": "stage3_hard",
        "config": {
            "wind": False,
            "obstacle": True,
            "dynamicObstacle_rate": 0.3,
            "num_obstacles": [4, 6],
        },
    },
    4: {
        "name": "stage4_final",
        "config": {
            "wind": True,
            "obstacle": True,
            "dynamicObstacle_rate": 0.3,
            "num_obstacles": [6, 8],
        },
    },
}


def animate_epi(model_path, env_config, max_epi=300):
    env = DummyVecEnv([lambda: UAVEnv(env_config)])
    model = PPO.load(model_path)

    obs = env.reset()
    done = False
    episode_reward = 0.0

    fig, ax = plt.subplots(figsize=(6, 6))
    score_text = ax.text(
        0.02,
        0.95,
        "Score: 0.0",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    fig.subplots_adjust(bottom=0.2)

    map_size = env.envs[0].map_size
    ax.set_xlim(-map_size, map_size)
    ax.set_ylim(-map_size, map_size)
    ax.grid(True)

    (uav_point,) = ax.plot([], [], "bo", markersize=6, label="UAV")
    traj_x, traj_y = [], []
    (traj_line,) = ax.plot([], [], "b-", linewidth=1)

    goal_x = env.envs[0].goal[0]
    goal_y = env.envs[0].goal[1]
    goal_point = ax.scatter(goal_x, goal_y, c="green", s=80, label="Goal")

    obstacles_scatter = ax.scatter([], [], c="red", s=80, label="Obstacles")
    ax.legend()

    def update_animation(frame):
        nonlocal obs, done, episode_reward

        if done:
            animation.event_source.stop()
            return traj_line, uav_point, obstacles_scatter, goal_point

        action, _ = model.predict(obs, deterministic=True)
        prev_state = env.envs[0].state.copy()
        next_obs, reward, dones, info = env.step(action)
        episode_reward += reward[0]
        score_text.set_text(f"Score: {episode_reward:.2f}")
        done = dones[0]

        obs = next_obs

        if done:
            x, y = prev_state[0], prev_state[1]
            traj_x.append(x)
            traj_y.append(y)
            traj_line.set_data(traj_x, traj_y)
            uav_point.set_data([x], [y])
            animation.event_source.stop()
            return traj_line, uav_point, obstacles_scatter, goal_point

        true_state = env.envs[0].state
        x, y = true_state[0], true_state[1]
        traj_x.append(x)
        traj_y.append(y)
        traj_line.set_data(traj_x, traj_y)
        uav_point.set_data([x], [y])

        obstacles = env.envs[0].obstacles
        if obstacles:
            obstacle_positions = np.array([obstacle["pos"] for obstacle in obstacles])
            obstacles_scatter.set_offsets(obstacle_positions)

        return traj_line, uav_point, obstacles_scatter, goal_point

    animation = FuncAnimation(
        fig, update_animation, frames=max_epi, interval=10, blit=False
    )

    ax_button = fig.add_axes([0.4, 0.05, 0.2, 0.075])
    btn_restart = Button(ax_button, "Restart")

    def restart_episode(event):
        nonlocal obs, done, traj_x, traj_y, episode_reward
        obs = env.reset()
        done = False
        traj_x, traj_y = [], []
        episode_reward = 0.0
        score_text.set_text("Score: 0.0")
        traj_line.set_data([], [])

        x, y = env.envs[0].state[0], env.envs[0].state[1]
        uav_point.set_data([x], [y])

        new_goal_x, new_goal_y = env.envs[0].goal
        goal_point.set_offsets(np.array([[new_goal_x, new_goal_y]]))

        obstacles = env.envs[0].obstacles
        if obstacles:
            obstacle_positions = np.array([obstacle["pos"] for obstacle in obstacles])
            obstacles_scatter.set_offsets(obstacle_positions)
        else:
            obstacles_scatter.set_offsets(np.empty((0, 2)))

        animation.event_source.start()
        fig.canvas.draw_idle()

    btn_restart.on_clicked(restart_episode)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAV Evaluation Animation")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help="Choose the curriculum stage to test (1: Easy, 2: Medium, 3: Hard, 4: Final)",
    )
    args = parser.parse_args()

    selected_stage = STAGE_CONFIGS[args.stage]
    test_config = ENV_CONFIG.copy()
    test_config.update(selected_stage["config"])

    model_file = f"./models/{selected_stage['name']}.zip"

    print(f"=========================================")
    print(f"Testing Stage: {args.stage} - {selected_stage['name']}")
    print(f"Model Path: {model_file}")
    print(f"Wind: {test_config['wind']}, Obstacles: {test_config['obstacle']}")
    print(f"Dynamic Rate: {test_config['dynamicObstacle_rate']}")
    print(f"=========================================")

    if not os.path.exists(model_file):
        print(f" 错误: 找不到模型文件 {model_file}。请确认训练是否成功保存。")
        sys.exit(1)

    animate_epi(model_file, test_config)

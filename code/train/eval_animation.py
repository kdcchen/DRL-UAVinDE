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

from config.config import ENV_CONFIG, STAGE_CONFIGS
from uav_env.uav_env import UAVEnv


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
        default=6,
        help="Choose the curriculum stage number to test (e.g., 1, 2, 3...)",
    )
    args = parser.parse_args()

    stage_num = args.stage

    # 1. 验证配置文件中是否存在该阶段
    if stage_num not in STAGE_CONFIGS:
        print(f"错误: 找不到阶段 {stage_num} 的环境配置。检查 config.py")
        sys.exit(1)

    selected_stage = STAGE_CONFIGS[stage_num]
    test_config = ENV_CONFIG.copy()
    test_config.update(selected_stage["config"])

    # 2. 动态扫描 ./models 文件夹，寻找以 stage{N}_ 开头的 zip 文件
    model_dir = "./models"
    prefix = f"stage{stage_num}_"

    if not os.path.exists(model_dir):
        print(f"错误: 模型文件夹 {model_dir} 不存在。")
        sys.exit(1)

    # 过滤寻找符合前缀且是 .zip 的文件
    matching_files = [
        f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith(".zip")
    ]

    if not matching_files:
        print(f"错误: 在 {model_dir} 中找不到以 '{prefix}' 开头的模型文件。")
        sys.exit(1)

    # 如果有多个匹配（比如人为备份），默认取第一个找到的
    model_file = os.path.join(model_dir, matching_files[0])

    print(f"=========================================")
    print(f"Testing Stage: {stage_num}")
    print(f"Loaded Config: {selected_stage['name']}")
    print(f"Found Model: {model_file}")
    print(
        f"Wind: {test_config.get('wind', False)}, Obstacles: {test_config.get('obstacle', False)}"
    )
    print(f"Dynamic Rate: {test_config.get('dynamicObstacle_rate', 0.0)}")
    print(f"Layout: {test_config.get('obstacle_layout', 'random')}")
    print(f"=========================================")

    animate_epi(model_file, test_config)

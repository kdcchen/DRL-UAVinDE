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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def animate_epi(model_path, env_config, max_epi=300):
    env = DummyVecEnv([lambda: UAVEnv(env_config)])
    model = PPO.load(model_path)

    obs = env.reset()
    done = False
    episode_reward = 0.0
    final_speed = 0.0

    # 加宽界面，为右侧信息栏留出空间
    fig = plt.figure(figsize=(10, 6))

    # 左侧：占据 60% 宽度的飞行地图
    ax = fig.add_axes([0.05, 0.15, 0.6, 0.8])
    map_size = env.envs[0].map_size
    ax.set_xlim(-map_size, map_size)
    ax.set_ylim(-map_size, map_size)
    ax.grid(True)

    # 右侧：占据 30% 宽度的信息展示面板
    ax_info = fig.add_axes([0.68, 0.15, 0.3, 0.8])
    ax_info.axis("off")
    info_text = ax_info.text(
        0.0,
        0.95,
        "=== 实时飞行数据 ===\n\n"
        "总分数 (Score):\n  0.00\n\n"
        "接近终点加分:\n  0.000\n\n"
        "接近障碍扣分:\n  0.000\n\n"
        "实时速度:\n  0.000 m/s",
        fontsize=12,
        verticalalignment="top",
    )

    (uav_point,) = ax.plot([], [], "bo", markersize=6, label="UAV")
    traj_x, traj_y = [], []
    (traj_line,) = ax.plot([], [], "b-", linewidth=1)

    goal_x = env.envs[0].goal[0]
    goal_y = env.envs[0].goal[1]
    goal_point = ax.scatter(goal_x, goal_y, c="green", s=80, label="Goal")

    obstacles_scatter = ax.scatter([], [], c="red", s=80, label="Obstacles")
    ax.legend()

    def update_animation(frame):
        nonlocal obs, done, episode_reward, final_speed

        if done:
            animation.event_source.stop()
            return traj_line, uav_point, obstacles_scatter, goal_point, info_text

        # 记录执行动作前的状态，用于计算进度差值
        action, _ = model.predict(obs, deterministic=True)
        prev_state = env.envs[0].state.copy()
        dist_before = np.linalg.norm(env.envs[0].goal - prev_state[:2])

        next_obs, reward, dones, info = env.step(action)
        episode_reward += reward[0]
        done = dones[0]
        obs = next_obs

        # 记录执行动作后的当前状态
        curr_state = env.envs[0].state
        dist_after = np.linalg.norm(env.envs[0].goal - curr_state[:2])

        if dist_after < 0.5:
            return

        # 1. 计算接近终点加分
        prog_r = 10.0 * (dist_before - dist_after)

        # 2. 计算接近障碍扣分 (还原环境中的 4次幂 惩罚逻辑)
        obs_p = 0.0
        if env.envs[0].obstacle:
            safe_r = env.envs[0].config["obstacle_radius"] + 0.2
            buffer_zone = 2.0
            for obs_obj in env.envs[0].obstacles:
                d_obs = np.linalg.norm(obs_obj["pos"] - curr_state[:2])
                if d_obs < safe_r:
                    obs_p -= 20.0
                elif d_obs < safe_r + buffer_zone:
                    dist_from_safe = d_obs - safe_r
                    obs_p -= 0.625 * (2.0 - dist_from_safe) ** 4

        real_speed = np.linalg.norm(curr_state[2:4])

        # 更新右侧面板文本
        info_text.set_text(
            f"=== 实时飞行数据 ===\n\n"
            f"总分数 (Score):\n  {episode_reward:.2f}\n\n"
            f"接近终点加分:\n  {prog_r:.3f}\n\n"
            f"接近障碍扣分:\n  {obs_p:.3f}\n\n"
            f"实时速度:\n  {real_speed:.3f} m/s"
        )

        if done:
            x, y = prev_state[0], prev_state[1]
            traj_x.append(x)
            traj_y.append(y)
            traj_line.set_data(traj_x, traj_y)
            uav_point.set_data([x], [y])
            animation.event_source.stop()
            return traj_line, uav_point, obstacles_scatter, goal_point, info_text

        x, y = curr_state[0], curr_state[1]
        traj_x.append(x)
        traj_y.append(y)
        traj_line.set_data(traj_x, traj_y)
        uav_point.set_data([x], [y])

        obstacles = env.envs[0].obstacles
        if obstacles:
            obstacle_positions = np.array([obstacle["pos"] for obstacle in obstacles])
            obstacles_scatter.set_offsets(obstacle_positions)

        return traj_line, uav_point, obstacles_scatter, goal_point, info_text

    animation = FuncAnimation(
        fig, update_animation, frames=max_epi, interval=10, blit=False
    )

    # 调整 Restart 按钮的位置，使其对齐到左侧地图的下方
    ax_button = fig.add_axes([0.25, 0.05, 0.2, 0.075])
    btn_restart = Button(ax_button, "Restart")

    def restart_episode(event):
        nonlocal obs, done, traj_x, traj_y, episode_reward, final_speed
        obs = env.reset()
        done = False
        traj_x, traj_y = [], []
        episode_reward = 0.0
        final_speed = 0.0

        info_text.set_text(
            "=== 实时飞行数据 ===\n\n"
            "总分数 (Score):\n  0.00\n\n"
            "接近终点加分:\n  0.000\n\n"
            "接近障碍扣分:\n  0.000\n\n"
            "实时速度:\n  0.000 m/s"
        )

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

    if stage_num not in STAGE_CONFIGS:
        print(f"错误: 找不到阶段 {stage_num} 的环境配置。检查 config.py")
        sys.exit(1)

    selected_stage = STAGE_CONFIGS[stage_num]
    test_config = ENV_CONFIG.copy()
    test_config.update(selected_stage["config"])

    model_dir = "./models"
    prefix = f"stage{stage_num}_"

    if not os.path.exists(model_dir):
        print(f"错误: 模型文件夹 {model_dir} 不存在。")
        sys.exit(1)

    matching_files = [
        f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith(".zip")
    ]

    if not matching_files:
        print(f"错误: 在 {model_dir} 中找不到以 '{prefix}' 开头的模型文件。")
        sys.exit(1)

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

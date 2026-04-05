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

from config.config import ENV_CONFIG, PATHS
from uav_env.uav_env import UAVEnv


def animate_epi(model_path, max_epi=300):
    env = DummyVecEnv([lambda: UAVEnv(ENV_CONFIG)])
    model = PPO.load(model_path)

    obs = env.reset()
    done = False
    episode_reward = 0.0

    fig, ax = plt.subplots(figsize=(6, 6))
    score_text = ax.text(
        0.02, 0.95, "Score: 0.0",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top"
    )

    fig.subplots_adjust(bottom=0.2)

    map_size = env.envs[0].map_size
    ax.set_xlim(-map_size, map_size)
    ax.set_ylim(-map_size, map_size)
    ax.grid(True)

    # UAV
    (uav_point,) = ax.plot([], [], "bo", markersize=6, label="UAV")

    # The trajectory line that UAV runs
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
        episode_reward += reward[0]  # DummyVecEnv 返回的是数组
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

    # ==========================================
    # [新增] 添加 Restart 按钮及回调逻辑
    # ==========================================
    # 在图表底部定义一个区域存放按钮：[left, bottom, width, height]
    ax_button = fig.add_axes([0.4, 0.05, 0.2, 0.075])
    btn_restart = Button(ax_button, "Restart")

    def restart_episode(event):
        nonlocal obs, done, traj_x, traj_y, episode_reward

        # 1. 重新生成环境，获取全新的初始观测值
        obs = env.reset()
        done = False
        traj_x, traj_y = [], []

        episode_reward = 0.0  #  重置分数
        score_text.set_text("Score: 0.0")

        # 2. 清除图表上旧的轨迹线
        traj_line.set_data([], [])

        # 3. 更新无人机初始位置
        x, y = env.envs[0].state[0], env.envs[0].state[1]
        uav_point.set_data([x], [y])

        # 4. 更新新的随机目标点位置
        new_goal_x, new_goal_y = env.envs[0].goal
        goal_point.set_offsets(np.array([[new_goal_x, new_goal_y]]))

        # 5. 更新新的随机障碍物位置
        obstacles = env.envs[0].obstacles
        if obstacles:
            obstacle_positions = np.array([obstacle["pos"] for obstacle in obstacles])
            obstacles_scatter.set_offsets(obstacle_positions)
        else:
            obstacles_scatter.set_offsets(np.empty((0, 2)))

        # 6. 恢复因之前结束而停止的动画计时器
        animation.event_source.start()

        # 7. 立即重绘画布
        fig.canvas.draw_idle()

    # 绑定按钮被点击时的触发事件
    btn_restart.on_clicked(restart_episode)
    # ==========================================

    plt.show()


if __name__ == "__main__":
    # animate_epi(PATHS["final_model"])
    animate_epi("./models/ppo_uav_final.zip")

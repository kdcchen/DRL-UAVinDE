import matplotlib.pyplot as plt
import numpy as np


def analyze_and_plot_trajectory(file_path="models/trajectory_epi0.npy"):
    try:
        traj = np.load(file_path)
    except FileNotFoundError:
        print(f"找不到文件: {file_path}，请确保路径正确。")
        return

    goal = np.array([4.0, 4.0])
    success_threshold = 0.3

    x_coords = traj[:, 0]
    y_coords = traj[:, 1]

    print("=" * 40)
    print("无人机飞行轨迹分析")
    print("=" * 40)

    total_steps = traj.shape[0]
    start_pos = traj[0]
    final_pos = traj[-1]

    print(f"总飞行步数: {total_steps} 步")
    print(f"起点坐标 (x, y): ({start_pos[0]:.4f}, {start_pos[1]:.4f})")
    print(f"终点坐标 (x, y): ({final_pos[0]:.4f}, {final_pos[1]:.4f})")

    differences = np.diff(traj, axis=0)
    distances = np.linalg.norm(differences, axis=1)
    total_distance = np.sum(distances)
    print(f"实际飞行总距离: {total_distance:.4f}")

    dist_to_goal = np.linalg.norm(goal - final_pos)
    is_success = dist_to_goal < success_threshold

    print("-" * 40)
    print(f"最终距离目标的距离: {dist_to_goal:.4f}")
    if is_success:
        print("任务结果: 成功 到达目标")
    else:
        print("任务结果: 未能 到达目标")
    print("=" * 40)

    plt.figure(figsize=(8, 8))

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    dx = np.diff(x_coords)
    dy = np.diff(y_coords)

    arrow_step = max(1, len(x_coords) // 20)

    plt.quiver(
        x_coords[:-1:arrow_step],
        y_coords[:-1:arrow_step],
        dx[::arrow_step],
        dy[::arrow_step],
        color="mediumblue",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.005,
        headwidth=4,
        headlength=6,
        headaxislength=4,
    )

    plt.plot(
        x_coords,
        y_coords,
        label="UAV Trajectory",
        color="royalblue",
        linewidth=2,
        linestyle="--",
    )

    plt.scatter(
        start_pos[0],
        start_pos[1],
        color="black",
        marker="o",
        s=100,
        label="Start Position",
        zorder=5,
    )

    plt.scatter(
        final_pos[0],
        final_pos[1],
        color="red",
        marker="X",
        s=100,
        label="End Position",
        zorder=5,
    )

    plt.scatter(
        goal[0], goal[1], color="forestgreen", marker="*", s=250, label="Goal", zorder=5
    )

    goal_circle = plt.Circle(
        (goal[0], goal[1]),
        success_threshold,
        color="forestgreen",
        fill=True,
        alpha=0.2,
        label="Success Zone",
    )
    plt.gca().add_patch(goal_circle)

    plt.title("UAV Flight Trajectory Analysis", fontsize=14, fontweight="bold")
    plt.xlabel("X Position", fontsize=12)
    plt.ylabel("Y Position", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.7)

    plt.show()


if __name__ == "__main__":
    analyze_and_plot_trajectory()

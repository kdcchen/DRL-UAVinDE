import gymnasium as gym
import numpy as np
from gymnasium import spaces


class UAVEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_steps = 300
        self.map_size = config["map_size"]
        self.dt = config["dt"]
        self.max_speed = config["max_speed"]
        self.max_accel = config["max_accel"]
        self.step_count = 0

        self.wind = config["wind"]

        self.obstacle = config["obstacle"]
        self.max_num_obstacles = config["num_obstacles"][1]
        self.obstacle_radius = config["obstacle_radius"]

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # [x,y,vx,vy,gx,gy]

        high = np.array(
            [
                self.map_size,
                self.map_size,
                self.max_speed,
                self.max_speed,
                self.map_size,
                self.map_size,
            ],
            dtype=np.float32,
        )
        if self.obstacle:
            obs_dim = 6 + 2 * self.max_num_obstacles
            self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32)
        else:
            obs_dim = 6
            self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32)

        self.goal = None

        self.state = None

        self.obstacles = []
        self.prev_vx = None
        self.prev_vy = None
        self.prev_dist = None
        self.render_mode = None
        self.prev_angle = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        # Random initial
        x, y = self.np_random.uniform(low=-self.map_size + 1, high=self.map_size - 1, size=(2,))
        vx, vy = 0.0, 0.0

        self.state = np.array(
            [
                x,
                y,
                vx,
                vy,
            ],
            dtype=np.float32,
        )

        # random goal
        gx = self.np_random.uniform(-self.map_size + 1, self.map_size - 1)
        gy = self.np_random.uniform(-self.map_size + 1, self.map_size - 1)
        self.goal = np.array([gx, gy], dtype=np.float32)

        # reset obstacles if it is true
        if self.obstacle:
            self._reset_obstacles()
        else:
            self.obstacles = []

        self.prev_dist = np.linalg.norm(self.goal - np.array([x, y]))

        goal_vec = self.goal - np.array([x, y])
        goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)

        speed_vec = np.array([vx, vy])
        speed_norm = np.linalg.norm(speed_vec)

        if speed_norm > 1e-6:
            speed_dir = speed_vec / speed_norm
            self.prev_angle = np.arccos(np.clip(np.dot(goal_dir, speed_dir), -1.0, 1.0))
        else:
            self.prev_angle = 0.0

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        x, y, vx, vy = self.state

        ax, ay = np.clip(action*self.max_accel, -self.max_accel, self.max_accel)

        wind = self._compute_wind() if self.wind else np.zeros(2)

        vx = np.clip(vx + (ax + wind[0]) * self.dt, -self.max_speed, self.max_speed)
        vy = np.clip(vy + (ay + wind[1]) * self.dt, -self.max_speed, self.max_speed)

        x_new = x + vx * self.dt
        y_new = y + vy * self.dt

        clipped_x = np.clip(x_new, -self.map_size, self.map_size)
        clipped_y = np.clip(y_new, -self.map_size, self.map_size)

        self.state = np.array([clipped_x, clipped_y, vx, vy], dtype=np.float32)

        if self.obstacle:
            self._update_obstacles()

        reward = self._compute_reward(action=action)

        dist_to_goal = np.linalg.norm(self.goal - np.array([clipped_x, clipped_y]))
        terminated = dist_to_goal < 0.3
        truncated = self.step_count >= self.max_steps

        if abs(clipped_x) > self.map_size - 0.1 or abs(clipped_y) > self.map_size - 0.1:
            reward -= 100.0
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _compute_wind(self):
        return self.config["wind_std"] * self.np_random.normal(0, 1, size=2)

    def _reset_obstacles(self):
        self.obstacles = []
        low, high = self.config["num_obstacles"]
        num_obs = self.np_random.integers(low, high + 1)

        speed_low, speed_high = self.config["obstacle_speed_range"]
        radius = self.config["obstacle_radius"]
        for _ in range(num_obs):
            # random position
            pos = self.np_random.uniform(
                -self.map_size + 1.0, self.map_size - 1.0, size=2
            )

            if self.np_random.random() < 1 - self.config["dynamicObstacle_rate"]:
                vel = np.zeros(2)
                obs_type = "static"
            else:
                # random direction + random speed
                angle = self.np_random.uniform(0, 2 * np.pi)
                speed = self.np_random.uniform(speed_low, speed_high)
                vel = np.array([np.cos(angle) * speed, np.sin(angle) * speed])
                obs_type = "dynamic"

            self.obstacles.append({"type": obs_type, "pos": pos, "vel": vel})

    def _update_obstacles(self):
        for obstacle in self.obstacles:
            if obstacle["type"] == "dynamic":
                obstacle["pos"] += obstacle["vel"] * self.dt

    def _get_obs(self):
        x,y,vx,vy = self.state

        x_n = x / self.map_size
        y_n = y / self.map_size
        vx_n = vx / self.max_speed
        vy_n = vy / self.max_speed

        gx_n = (self.goal[0] - x) / self.map_size
        gy_n = (self.goal[1] - y) / self.map_size

        obs = [x_n, y_n, vx_n, vy_n, gx_n, gy_n]

        # obstacles (relative, padded)
        if self.obstacles:
            for i in range(self.max_num_obstacles):
                if i < len(self.obstacles):
                    ox, oy = self.obstacles[i]["pos"]
                    obs.append((ox - x) / self.map_size)
                    obs.append((oy - y) / self.map_size)
                else:
                    obs.extend([0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, action=None):
        x, y, vx, vy = self.state

        # 1. 归一化距离（关键改动）
        dist_abs = np.linalg.norm(self.goal - np.array([x, y]))
        dist_norm = dist_abs / self.map_size

        # 用归一化距离做基础惩罚
        reward = -2.0 * dist_norm  # 原来是 -dist_abs，这里压到 [-2, 0] 左右

        # 2. 距离进步（同样用归一化）
        prev_dist_norm = self.prev_dist / self.map_size
        delta = prev_dist_norm - dist_norm
        reward += 6.0 * delta  # 提高一点权重，让“变近”更显眼
        self.prev_dist = dist_abs

        # 3. 方向奖励 + 转向奖励
        goal_vec = self.goal - np.array([x, y])
        goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)

        speed_vec = np.array([vx, vy])
        speed_norm = np.linalg.norm(speed_vec)

        speed_dir = None
        if speed_norm > 1e-6:
            speed_dir = speed_vec / speed_norm

            # 3.1 方向对准奖励（加大）
            cos_theta = np.clip(np.dot(goal_dir, speed_dir), -1.0, 1.0)
            reward += 4.0 * cos_theta

            # 3.2 角度减少奖励（加大）
            angle = np.arccos(cos_theta)
            angle_delta = self.prev_angle - angle
            reward += 3.0 * angle_delta
            self.prev_angle = angle
        else:
            angle = self.prev_angle

        if action is not None:
            ax, ay = action
            a_norm = np.linalg.norm(action)
            if a_norm > 1e-6:
                a_dir = action / a_norm
                if speed_dir is not None:
                    desired_dir = goal_dir - speed_dir
                else:
                    desired_dir = goal_dir.copy()
                desired_dir /= (np.linalg.norm(desired_dir) + 1e-6)
                reward += 3.0 * np.dot(a_dir, desired_dir)

        # 4. 靠墙惩罚
        wall_margin = self.map_size - 0.5
        if abs(x) > wall_margin:
            reward -= 4.0 * abs(vx)
        if abs(y) > wall_margin:
            reward -= 4.0 * abs(vy)

        # 5. 速度惩罚（减弱）
        reward -= 0.05 * speed_norm

        # 6. 到达奖励（可以保持）
        if dist_abs < 0.3:
            reward += 80.0

        return reward

    def render(self):
        pass

    def close(self):
        pass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class UAVEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.map_size = config["map_size"]
        self.dt = config["dt"]
        self.max_speed = config["max_speed"]
        self.max_accel = config["max_accel"]
        self.step_count = 0
        self.K = config["obstacle_k"]
        self.wind = config["wind"]
        self.obstacle = config["obstacle"]
        self.safety_radius = config["safety_radius"]

        self.action_space = spaces.Box(
            low=-self.max_accel, high=self.max_accel, shape=(2,), dtype=np.float32
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
        obs_dim = 7 + 2 * self.K
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)


        self.goal = None

        self.state = None
        self.episode_reward = 0.0
        self.obstacles = []
        self.prev_dist = None
        self.render_mode = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.episode_reward = 0.0
        # Random initial
        x, y = self.np_random.uniform(
            low=-self.map_size + 1.0,
            high=self.map_size - 1.0,
            size=(2,)
        )
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
        while True:
            gx = self.np_random.uniform(-self.map_size + 1, self.map_size - 1)
            gy = self.np_random.uniform(-self.map_size + 1, self.map_size - 1)
            if np.linalg.norm(np.array([gx, gy]) - np.array([x, y])) > self.map_size:
                break
        self.goal = np.array([gx, gy], dtype=np.float32)

        # reset obstacles if it is true
        if self.obstacle:
            self._reset_obstacles()

        self.prev_dist = np.linalg.norm(self.goal - np.array([x, y]))
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        truncated = self.step_count > 300
        x, y, vx, vy = self.state

        ax, ay = np.clip(action, -self.max_accel, self.max_accel)

        wind = self._compute_wind() if self.wind else np.zeros(2)

        vx = np.clip(vx + (ax + wind[0]) * self.dt, -self.max_speed, self.max_speed)
        vy = np.clip(vy + (ay + wind[1]) * self.dt, -self.max_speed, self.max_speed)

        x = x + vx * self.dt
        y = y + vy * self.dt

        x = np.clip(x, -self.map_size, self.map_size)
        y = np.clip(y, -self.map_size, self.map_size)

        self.state = np.array([x, y, vx, vy], dtype=np.float32)

        if self.obstacle:
            self._update_obstacles()

        reward = self._compute_reward()

        terminated = False
        success = False

        dist_to_goal = np.linalg.norm(self.goal - np.array([x, y]))
        if dist_to_goal < 0.3:
            terminated = True
            success = True
            reward -= 0.1 * self.step_count

        if abs(x) > self.map_size-0.1 or abs(y) > self.map_size - 0.1:
            terminated = True

        if self.obstacle:
            for obs_i in self.obstacles:
                d = np.linalg.norm(obs_i["pos"] - np.array([x, y]))
                if d < self.config["obstacle_radius"]:
                    terminated = True


        if terminated and not success:
            reward -= 50
        self.episode_reward += reward

        info = {"success": success, "dist_to_goal": dist_to_goal}

        return self._get_obs(), reward, terminated, truncated, info

    def _compute_wind(self):
        return self.config["wind_std"] * self.np_random.normal(0, 1, size=2)

    def _reset_obstacles(self):
        self.obstacles = []
        low, high = self.config["num_obstacles"]
        num_obs = self.np_random.integers(low, high + 1)

        speed_low, speed_high = self.config["obstacle_speed_range"]
        radius = self.config["obstacle_radius"]

        # Safety Radius

        safe_dist_start = self.safety_radius
        safe_dist_goal = self.safety_radius
        min_dist_between_obs = 2 * radius + 0.3

        start_pos = np.array([self.state[0], self.state[1]])
        goal_pos = self.goal

        for _ in range(num_obs):
            while True:
                # random position
                pos = self.np_random.uniform(
                    -self.map_size + 1.0, self.map_size - 1.0, size=2
                )
                # Too close to UAV → reset
                if np.linalg.norm(pos - start_pos) < safe_dist_start:
                   continue

                    # Too close to goal → reset
                if np.linalg.norm(pos - goal_pos) < safe_dist_goal:
                    continue
                too_close = False
                for obs in self.obstacles:
                    if np.linalg.norm(pos - obs["pos"]) < min_dist_between_obs:
                        too_close = True
                        break
                if too_close:
                    continue
                break

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
        x, y, vx, vy = self.state

        x_n = x / self.map_size
        y_n = y / self.map_size
        vx_n = vx / self.max_speed
        vy_n = vy / self.max_speed

        gx_n = (self.goal[0] - x) / self.map_size
        gy_n = (self.goal[1] - y) / self.map_size

        obs = [x_n, y_n, vx_n, vy_n, gx_n, gy_n]
        goal_vec = np.array([self.goal[0] - x, self.goal[1] - y])
        goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)

        vel = np.array([vx, vy])
        vel_norm = np.linalg.norm(vel)
        if vel_norm < 1e-6:
            vel_dir = np.array([0.0, 0.0])
        else:
            vel_dir = vel / vel_norm

        cos_theta = np.dot(goal_dir, vel_dir)
        obs.append(cos_theta)

        d_list = []
        if self.obstacle:
            d_list = [np.linalg.norm(obs_i["pos"] - np.array([x, y])) for obs_i in self.obstacles]
            sorted_idx = np.argsort(d_list)
        else:
            sorted_idx = []

        for i in range(self.K):
            if self.obstacle and i < len(sorted_idx):
                obs_i = self.obstacles[sorted_idx[i]]
                ox, oy = obs_i["pos"]
                obs.append((ox - x) / self.map_size)
                obs.append((oy - y) / self.map_size)
            else:
                obs.append(2.0)
                obs.append(2.0)

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        x, y, vx, vy = self.state
        pos = np.array([x, y])
        dist = np.linalg.norm(self.goal - pos)

        reward = -0.1

        reward += (self.prev_dist - dist)

        # reward of progress
        if hasattr(self, "prev_dist"):

            goal_vec = np.array([self.goal[0] - x, self.goal[1] - y])
            goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)

            vel = np.array([vx, vy])
            vel_norm = np.linalg.norm(vel)
            if vel_norm < 1e-6:
                vel_dir = np.array([0.0, 0.0])
            else:
                vel_dir = vel / vel_norm

            cos_theta = np.dot(goal_dir, vel_dir)
            reward += 0.1 * cos_theta

        self.prev_dist = dist

        if self.obstacle:
            for obs in self.obstacles:
                d = np.linalg.norm(obs["pos"] - pos)
                safe_r = self.config["obstacle_radius"] + 0.2

                if d < safe_r:
                    reward -= 20  # 撞上
                elif d < safe_r + 0.8:
                    reward -= 1.0 * (safe_r + 0.8 - d)

        if dist < 1.0:
            reward += 1.0 * (1.0 - dist)

        if dist < 0.3:
            reward += 50

        return reward

    def render(self):
        pass

    def close(self):
        pass

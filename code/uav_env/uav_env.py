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

        self.wind = config["wind"]
        self.obstacle = config["obstacle"]

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
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.goal = None

        self.state = None

        self.obstacles = []

        self.render_mode = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        # Random initial
        x, y = self.np_random.uniform(low=-4.5, high=-3.5, size=(2,))
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

        obs = np.concatenate([self.state, self.goal])

        return obs, {}

    def step(self, action):
        self.step_count += 1
        truncated = self.step_count > 500
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

        dist_to_goal = np.linalg.norm(self.goal - np.array([x, y]))
        terminated = dist_to_goal < 0.3
        info = {}
        if terminated:
            obs = np.concatenate([self.state, self.goal])
            return obs, reward, terminated, truncated, info

        obs = np.concatenate([self.state, self.goal])

        return obs, reward, terminated, truncated, info

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

    def _compute_reward(self):
        x, y, vx, vy = self.state
        dist = np.linalg.norm(self.goal - np.array([x, y]))

        reward = -dist

        radius = self.config["obstacle_radius"]
        # penalty of speed
        # speed = np.linalg.norm([vx,vy])
        # reward -= 0.005 * speed

        if abs(x) > self.map_size - 0.2 or abs(y) > self.map_size - 0.2:
            reward -= 5.0

        if dist < 0.3:
            reward += 100

        # punishment of collapse as well as too close

        if self.obstacle:
            for obstacle in self.obstacles:
                d = np.linalg.norm(obstacle["pos"] - np.array([x, y]))
                if d < 1.0:
                    reward -= (1.0 - d) * 0.5
                if d < radius:
                    reward -= 5.0

        return reward

    def render(self):
        pass

    def close(self):
        pass

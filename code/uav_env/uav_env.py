import gymnasium as gym
from gymnasium import spaces
import numpy as np

class UAVEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, config):
        super().__init__()

        self.dt = config["dt"]
        self.max_speed = config["max_speed"]
        self.max_accel = config["max_accel"]
        self.step_count = 0

        self.wind = config["wind"]
        self.obstacle = config["obstacle"]

        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel,shape=(2,),dtype=np.float32)

        # [x,y,vx,vy,gx,gy]

        high = np.array([10,10,2,2,10,10],dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.goal = np.array(config["goal"],dtype=np.float32)

        self.state = None

        self.obstacles = []

        self.prev_dist = None

        self.render_mode = None

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.step_count = 0
        # Random initial
        x,y = self.np_random.uniform(low=-1.0, high=1.0, size=(2,))
        vx,vy = 0.0,0.0

        self.state = np.array([x,y,vx,vy,], dtype=np.float32)

        # reset obstacles if it is true
        if self.obstacle:
            self._reset_obstacles()

        obs = np.concatenate([self.state, self.goal])

        self.prev_dist = np.linalg.norm(self.goal - np.array([x, y]))

        return obs, {}

    def step(self, action):
        self.step_count += 1
        truncated = self.step_count > 300
        x,y,vx,vy = self.state

        ax,ay = np.clip(action,-self.max_accel,self.max_accel)

        wind = self._compute_wind() if self.wind else np.zeros(2)

        vx = np.clip(vx + (ax + wind[0]) * self.dt , -self.max_speed, self.max_speed)
        vy = np.clip(vy + (ay + wind[1]) * self.dt , -self.max_speed, self.max_speed)

        x = x + vx * self.dt
        y = y + vy * self.dt

        x = np.clip(x, -5, 5)
        y = np.clip(y, -5, 5)

        self.state = np.array([x,y,vx,vy], dtype=np.float32)

        if self.obstacle:
            self._update_obstacles()

        reward = self._compute_reward()

        dist_to_goal = np.linalg.norm(self.goal - np.array([x,y]))
        terminated = dist_to_goal < 0.3
        info = {}
        if terminated:
            reward += 20
            obs = np.concatenate([self.state, self.goal])
            return obs, reward, terminated, truncated, info

        obs = np.concatenate([self.state, self.goal])


        return obs, reward, terminated, truncated, info

    def _compute_wind(self):
        return 0.1 * self.np_random.normal(loc=0.0, scale=1.0, size=2)

    def _reset_obstacles(self):
        self.obstacles = []

        num_obs = self.np_random.integers(2, 5)

        for _ in range(num_obs):
            # random position
            pos = self.np_random.uniform(low=-3.0, high=3.0, size=2)

            # 50% static，50% dynamic
            if self.np_random.random() < 0.5:
                vel = np.zeros(2)
                obs_type = "static"
            else:
                # random direction + random speed
                angle = self.np_random.uniform(0, 2 * np.pi)
                speed = self.np_random.uniform(0.1, 0.4)
                vel = np.array([np.cos(angle) * speed, np.sin(angle) * speed])
                obs_type = "dynamic"

            self.obstacles.append({
                "type": obs_type,
                "pos": pos,
                "vel": vel
            })

    def _update_obstacles(self):
        for obstacle in self.obstacles:
            if obstacle["type"] == "dynamic":
                obstacle["pos"] += obstacle["vel"] * self.dt

    def _compute_reward(self):
        x,y,vx,vy = self.state
        dist = np.linalg.norm(self.goal - np.array([x,y]))

        reward = self.prev_dist - dist

        if abs(x) > 4.8 or abs(y) > 4.8:
            reward -= 1.0

        if dist < 0.3:
            reward += 20

        if self.obstacle:
            for obstacle in self.obstacles:
                if np.linalg.norm(obstacle["pos"] - np.array([x,y])) < 0.3:
                    reward -= 5.0

        self.prev_dist = dist
        return reward

    def render(self):
        pass

    def close(self):
        pass
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TrainingStatsCallback(BaseCallback):
    """
    Collects:
    - success rate
    - mean episode reward
    - mean episode length
    - mean distance to goal
    """

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.distances = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                # Episode reward
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

                # Success flag
                self.successes.append(info.get("success", 0))

                # Distance to goal
                if "dist_to_goal" in info:
                    self.distances.append(info["dist_to_goal"])

        # Print every 2000 steps
        if self.n_calls % 2000 == 0 and len(self.episode_rewards) > 5:
            mean_r = np.mean(self.episode_rewards[-20:])
            mean_len = np.mean(self.episode_lengths[-20:])
            success_rate = np.mean(self.successes[-20:])
            mean_dist = np.mean(self.distances[-20:]) if self.distances else 0

            print(f"\n📊 Training Stats (last 20 episodes)")
            print(f"   Avg Reward:      {mean_r:.2f}")
            print(f"   Avg Length:      {mean_len:.1f}")
            print(f"   Success Rate:    {success_rate*100:.1f}%")
            print(f"   Avg Dist to Goal:{mean_dist:.2f}")

        return True
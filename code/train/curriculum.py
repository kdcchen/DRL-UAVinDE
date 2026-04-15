import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from callbacks import TrainingStatsCallback
from config.config import ENV_CONFIG, PPO_CONFIG, STAGE_CONFIGS
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from uav_env.uav_env import UAVEnv


def make_env(config):
    return lambda: UAVEnv(config)


def create_vec_env(config, num_envs=16):
    env = SubprocVecEnv([make_env(config) for _ in range(num_envs)])
    env = VecMonitor(env)
    return env


if __name__ == "__main__":
    model = None

    for stage_idx, stage in STAGE_CONFIGS.items():
        print(f"\n=== Training Stage {stage_idx}: {stage['name']} ===")

        stage_config = ENV_CONFIG.copy()
        stage_config.update(stage["config"])

        env = create_vec_env(stage_config, num_envs=16)

        if model is None:
            model = PPO(
                PPO_CONFIG["policy"],
                env,
                learning_rate=PPO_CONFIG["learning_rate"],
                n_steps=PPO_CONFIG["n_steps"],
                batch_size=PPO_CONFIG["batch_size"],
                gamma=PPO_CONFIG["gamma"],
                gae_lambda=PPO_CONFIG["gae_lambda"],
                clip_range=PPO_CONFIG["clip_range"],
                ent_coef=PPO_CONFIG["ent_coef"],
                device=PPO_CONFIG["device"],
                verbose=1,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            )
        else:
            model.set_env(env)

        log_dir = f"./logs/{stage['name']}"
        new_logger = configure(log_dir, ["stdout", "tensorboard"])
        model.set_logger(new_logger)

        stats_callback = TrainingStatsCallback()

        model.learn(total_timesteps=stage["steps"], callback=stats_callback)
        model.save(f"./models/{stage['name']}")

    env.close()
    print("Training complete.")

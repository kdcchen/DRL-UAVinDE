import argparse
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
    parser = argparse.ArgumentParser(description="UAV Training Without Curriculum")
    parser.add_argument(
        "--model_name",
        type=str,
        default="no_curriculum",
        help="保存模型的名称，例如 no_curriculum",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="加载已有模型继续训练",
    )
    args = parser.parse_args()

    base_config = ENV_CONFIG.copy()
    base_config.update(STAGE_CONFIGS[7]["config"])

    print("=========================================")
    print(" Training Baseline: No Curriculum")
    print(" Using Stage 7 Full Difficulty Environment")
    print(" Wind:", base_config["wind"])
    print(" Obstacles:", base_config["obstacle"])
    print(" Dynamic Rate:", base_config["dynamicObstacle_rate"])
    print(" Layout:", base_config["obstacle_layout"])
    print("=========================================")

    env = create_vec_env(base_config, num_envs=16)

    if args.load_model and os.path.exists(args.load_model):
        print(f"加载预训练模型: {args.load_model}")
        model = PPO.load(args.load_model, env=env, device=PPO_CONFIG["device"])
    else:
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

    log_dir = f"./logs/{args.model_name}"
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    stats_callback = TrainingStatsCallback()

    total_steps = sum(stage["steps"] for stage in STAGE_CONFIGS.values())
    print(f"总训练步数（与 curriculum 对齐）: {total_steps}")

    model.learn(total_timesteps=total_steps, callback=stats_callback)

    save_path = f"./models/{args.model_name}"
    model.save(save_path)
    print(f"✅ Baseline 训练完成，模型已保存至: {save_path}.zip")

    env.close()

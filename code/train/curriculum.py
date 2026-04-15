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


def create_vec_env(config, num_envs=4):
    env = SubprocVecEnv([make_env(config) for _ in range(num_envs)])
    env = VecMonitor(env)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAV Curriculum Training")
    parser.add_argument(
        "--start_stage",
        type=int,
        default=1,
        help="从哪个阶段开始训练 (例如: 2 代表从 Stage 2 开始)",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="初始模型的路径 (例如: ./models/stage1_empty.zip)",
    )
    args = parser.parse_args()

    model = None

    stages_to_run = {k: v for k, v in STAGE_CONFIGS.items() if k >= args.start_stage}

    if not stages_to_run:
        print(f"错误: 找不到从阶段 {args.start_stage} 开始的配置。请检查 config.py。")
        sys.exit(1)

    for stage_idx, stage in sorted(stages_to_run.items()):
        print(f"\n{'='*40}")
        print(f"Training Stage {stage_idx}: {stage['name']}")
        print(f"{'='*40}")

        stage_config = ENV_CONFIG.copy()
        stage_config.update(stage["config"])

        env = create_vec_env(stage_config, num_envs=8)

        if model is None:
            if args.load_model and os.path.exists(args.load_model):
                print(f"加载预训练模型: {args.load_model} ...")
                model = PPO.load(args.load_model, env=env, device=PPO_CONFIG["device"])
            else:
                if args.load_model:
                    print(f"警告: 找不到模型文件 {args.load_model}。将重新训练。")
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

        # 训练结束后按照配置表中的 stage['name'] 自动命名保存
        save_path = f"./models/{stage['name']}"
        model.save(save_path)
        print(f"✅ 阶段 {stage_idx} 训练完成。模型已保存至: {save_path}.zip")

    env.close()
    print("\n🎉 所有指定阶段的课程学习已全部完成！")

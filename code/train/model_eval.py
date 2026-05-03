import argparse
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import ENV_CONFIG, STAGE_CONFIGS
from uav_env.uav_env import UAVEnv


def evaluate(model_path, env_config, episodes=100):
    env = DummyVecEnv([lambda: UAVEnv(env_config)])
    model = PPO.load(model_path)

    success_count = 0
    collision_count = 0
    timeout_count = 0

    episode_rewards = []
    episode_lengths = []
    final_distances = []
    final_speeds = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            # DummyVecEnv returns 4 values
            obs, reward, done_arr, info = env.step(action)
            done = done_arr[0]

            total_reward += reward[0]
            steps += 1

        info = info[0]

        final_speed = np.linalg.norm(env.envs[0].state[2:4])
        final_speeds.append(final_speed)

        if info.get("success", False):
            success_count += 1
        else:
            x, y = env.envs[0].state[:2]
            collided = False
            for obs_i in env.envs[0].obstacles:
                if np.linalg.norm(obs_i["pos"] - np.array([x, y])) < env.envs[0].config["obstacle_radius"]:
                    collided = True
                    break

            if collided:
                collision_count += 1
            else:
                timeout_count += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        final_distances.append(info["dist_to_goal"])

    env.close()

    print("\n========== Evaluation Results ==========")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print("----------------------------------------")
    print(f"Success Rate:   {success_count / episodes * 100:.2f}%")
    print(f"Collision Rate: {collision_count / episodes * 100:.2f}%")
    print(f"Timeout Rate:   {timeout_count / episodes * 100:.2f}%")
    print("----------------------------------------")
    print(f"Avg Episode Reward: {np.mean(episode_rewards):.2f}")
    print(f"Avg Episode Length: {np.mean(episode_lengths):.1f}")
    print(f"Avg Final Distance: {np.mean(final_distances):.3f}")
    print(f"Avg Final Speed:    {np.mean(final_speeds):.3f} m/s")
    print("========================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a UAV PPO model")
    parser.add_argument("--model", type=str, required=True, help="Path to the model .zip file")
    parser.add_argument("--stage", type=int, default=7, help="Curriculum stage to load config (default: 7)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"错误: 模型文件 {args.model} 不存在。")
        sys.exit(1)

    if args.stage not in STAGE_CONFIGS:
        print(f"警告: Stage {args.stage} 不存在，自动使用 Stage 7 配置。")
        args.stage = 7

    env_config = ENV_CONFIG.copy()
    env_config.update(STAGE_CONFIGS[args.stage]["config"])

    print(f"Using Stage {args.stage} ({STAGE_CONFIGS[args.stage]['name']}) for evaluation.")
    evaluate(args.model, env_config, episodes=args.episodes)

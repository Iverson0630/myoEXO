import argparse
import csv
import os
import time

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO
from register_env import register_mme
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default='nn/human/walk/sb3_600000_steps.zip',
        help="Path to a trained SB3 model .zip file",
    )
    parser.add_argument("--env-id", default="fullBodyWalk-v0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    register_mme()
    model = PPO.load(args.model_path, device=args.device)
    env = gym.make(args.env_id)

    obs, info = env.reset()
    epi_len, reward_sum = 0, 0
    for step in range(args.max_steps):

        # SB3 的 predict 必须接 obs，返回 action
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        epi_len += 1
        if terminated:
            print("======")
            print("episode len", epi_len, "episode reward", reward_sum)
            print(info['rwd_dict'])

            # logger.save_if_good(reward_sum)
            epi_len, reward_sum = 0, 0
            
            obs, info = env.reset()
        env.mj_render()   # 如果你想用自己的渲染
        time.sleep(0.02)
    env.close()


if __name__ == "__main__":
    main()

import argparse
import csv
import os
import time
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO, SAC
from register_env import make_env
from config import Config

config = Config()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default='nn/human/walk/sb3_final.zip',
        help="Path to a trained SB3 model .zip file",
    )

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--deterministic",  default=False,action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    env = make_env(config.model.env)
    model = SAC.load(args.model_path, device=args.device)
   

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
         

            # logger.save_if_good(reward_sum)
            epi_len, reward_sum = 0, 0
            
            obs, info = env.reset()
        env.mj_render()   # 如果你想用自己的渲染
        time.sleep(0.01)
    env.close()


if __name__ == "__main__":
    main()

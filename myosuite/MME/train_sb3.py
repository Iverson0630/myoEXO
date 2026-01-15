import torch
import numpy as np
import argparse
from collections import deque
import torch
import functools
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback,CallbackList,EveryNTimesteps
from wandb.integration.sb3 import WandbCallback
import wandb
from register_env import make_env
from config import Config


parser = argparse.ArgumentParser()

parser.add_argument('--resume', type=bool, default=False,
                    help='whether to resume training')
parser.add_argument('--resume_path', type=str, default="./output/sb3_fly/can_track_good_landing_x_vara_vel/sb3_fly_7000000_steps.zip",
                    help='path to resume training')
args = parser.parse_args()

config = Config()

def main():
    wandb.init(
        project = config.save_dir.wandb_project,
        name = config.save_dir.wandb_dir,
        sync_tensorboard=True,
    )



    NUM_ENVS = config.model.num_env
    env_fn = functools.partial(make_env, config.model.env)
    vec_env = SubprocVecEnv([env_fn for _ in range(NUM_ENVS)])


    # ----------------------------
    # 5. Configure PPO model
    # ----------------------------
    if args.resume:
        print("resume training from:", args.resume_path)
      
        model = PPO.load(args.resume_path, device="cuda")
        model.set_env(vec_env)


    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            device="cuda",              # 使用 GPU
            n_steps=config.train.n_steps,               # 每次 rollout 步数（越大训练越稳定）
            batch_size=config.train.batch_size,
            n_epochs=config.train.n_epochs,
            learning_rate=config.train.learning_rate,
            gamma=config.train.gamma,
            gae_lambda=config.train.gae_lambda,
            ent_coef=config.train.ent_coef,
            tensorboard_log = config.save_dir.nn_dir+"/logs",
        )


    checkpoint_cb = EveryNTimesteps(
        n_steps=config.train.save_step,
        callback=CheckpointCallback(
            save_freq=config.train.save_freq,   
            save_path=config.save_dir.nn_dir,
            name_prefix="sb3"
        )
    )

    callback_list = CallbackList([
        checkpoint_cb,
        WandbCallback(
            gradient_save_freq=1000,
            model_save_path=config.save_dir.nn_dir+"/models/",
            verbose=2
        )
    ])


    model.learn(
        total_timesteps=config.train.max_iteration,           # 训练 200 万步
        callback=callback_list,
    )

    model.save(config.save_dir.nn_dir+"/sb3_final")
    wandb.finish()

if __name__ == "__main__":
    main()

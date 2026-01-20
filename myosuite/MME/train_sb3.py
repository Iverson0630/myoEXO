import torch
import numpy as np
import argparse
from collections import deque
import torch
import functools
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback,CallbackList,EveryNTimesteps
from wandb.integration.sb3 import WandbCallback
import wandb
from register_env import make_env
from config import Config


parser = argparse.ArgumentParser()

parser.add_argument('--resume', type=bool, default=False,
                    help='whether to resume training')
parser.add_argument('--resume_path', type=str, default="nn/human/myoLegWalk/sb3_3400000_steps.zip",
                    help='path to resume training')
args = parser.parse_args()

config = Config()

def linear_schedule(initial_value):
    def schedule(progress_remaining):
        return progress_remaining * initial_value
    return schedule

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
    # 5. Configure PPO/SAC model
    # ----------------------------
    if args.resume:
        print("resume training from:", args.resume_path)
        algo = config.train.algo.upper()
        if algo == "PPO":
            model = PPO.load(args.resume_path, device="cuda")
        elif algo == "SAC":
            model = SAC.load(args.resume_path, device="cuda")
        else:
            raise ValueError(f"Unknown algo: {config.train.algo}")
        model.set_env(vec_env)


    else:
        algo = config.train.algo.upper()
        if algo == "PPO":
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
                clip_range=config.train.clip_ratio,
                tensorboard_log = config.save_dir.nn_dir+"/logs",
            )
        elif algo == "SAC":
            if config.train.lr_schedule == "linear":
                learning_rate = linear_schedule(config.train.learning_rate)
            else:
                learning_rate = config.train.learning_rate

            model = SAC(
                policy="MlpPolicy",
                env=vec_env,
                verbose=0,
                device="cuda",              # 使用 GPU
                buffer_size=config.train.buffer_size,
                batch_size=config.train.batch_size,
                learning_rate=learning_rate,
                gamma=config.train.gamma,
                train_freq=config.train.train_freq,
                gradient_steps=config.train.gradient_steps,
                learning_starts=config.train.learning_starts,
                tau=config.train.tau,
                target_update_interval=config.train.target_update_interval,
                ent_coef=config.train.ent_coef,
                target_entropy=config.train.target_entropy,
                policy_kwargs={
                    "net_arch": {
                        "pi": list(config.train.policy_hiddens),
                        "qf": list(config.train.q_hiddens),
                    },
                    "activation_fn": torch.nn.ReLU,
                },
                tensorboard_log = config.save_dir.nn_dir+"/logs",
            )
        else:
            raise ValueError(f"Unknown algo: {config.train.algo}")


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

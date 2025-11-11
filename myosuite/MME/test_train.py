from myosuite.utils import gym
import time
import mujoco
from mujoco import viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
class Config:
    num_env: 8


def make_env(rank):
    def _init():
        env = gym.make("myoLegWalk-v0")   # 你也可以换成 MyoHandPose-v0
        return env
    return _init


envs = SubprocVecEnv([make_env(i) for i in range(8)])

# 创建并行策略模型
model = PPO("MlpPolicy", envs, verbose=1, n_steps=1024, batch_size=2048)
model.learn(total_timesteps=5_000_000)

model.save("ppo_myoleg_multienv")
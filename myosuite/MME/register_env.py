from config import Config
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

config = Config()
def register_mme(model_path = config.model.model_path):
    gym.envs.registration.register(
        id="fullBodyWalk-v0",
        entry_point="env.fullbodywalk_v0:FullBodyWalkEnvV0",
        kwargs={
            "model_path":  model_path, 
        },
    )
     
def make_env(env_id):
    register_mme()
    env = gym.make(env_id)
    env = Monitor(env)
    return env

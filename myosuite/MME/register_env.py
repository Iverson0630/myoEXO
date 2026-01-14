from myosuite.utils import gym
from config import Config
config = Config()
def register_mme(model_path = config.model.model_path):
    gym.envs.registration.register(
        id="fullBodyWalk-v0",
        entry_point="env.fullbodywalk_v0:FullBodyWalkEnvV0",
        max_episode_steps=1000,
        kwargs={
            "model_path":  model_path, 
        },
    )
     

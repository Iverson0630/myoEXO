from gym.envs.registration import register

def register_mme(model_path):
    register(
        id="fullBodyWalk-v0",
        entry_point="env.fullbodywalk_v0:FullBodyWalkEnvV0",
        max_episode_steps=1000,
        kwargs={
            "model_path":  model_path, 
        },
    )
     

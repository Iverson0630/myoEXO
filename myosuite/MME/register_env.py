from myosuite.utils import gym

def register_mme(model_path = "../simhive/myo_sim/body/myofullbodyarms_muscle.xml"):
    gym.envs.registration.register(
        id="fullBodyWalk-v0",
        entry_point="env.fullbodywalk_v0:FullBodyWalkEnvV0",
        max_episode_steps=1000,
        kwargs={
            "model_path":  model_path, 
        },
    )
     

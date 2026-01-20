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
     
    gym.envs.registration.register(
        id="fullBodyBalance-v0",
        entry_point="env.fullbodywalk_v0:FullBodyBalanceEnvV0",
        kwargs={
            "model_path":  model_path, 
        },
    )
     
     # Gait Torso Walking ==============================
    gym.envs.registration.register(
    id="myoLegWalk-v1",
    entry_point="env.myolegwalk_v1:WalkEnvV0",
    max_episode_steps=1000,
    kwargs={
        "model_path":"../simhive/myo_sim/leg/myolegs.xml",
        "normalize_act": True,
        "min_height": 0.8,  # minimum center of mass height before reset
        "max_rot": 0.8,  # maximum rotation before reset
        "hip_period": 100,  # desired periodic hip angle movement
        "reset_type": "init",  # none, init, random
        "target_x_vel": 0.0,  # desired x velocity in m/s
        "target_y_vel": 1.2,  # desired y velocity in m/s
        "target_rot": None,  # if None then the initial root pos will be taken, otherwise provide quat
        },
    )
def make_env(env_id):
    register_mme()
    env = gym.make(env_id)
    env.sim.model.opt.timestep = 1/config.model.sim_hz
    env.frame_skip = config.model.sim_hz/config.model.ctl_hz
    env = Monitor(env)
    return env

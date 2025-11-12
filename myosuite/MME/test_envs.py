#from myosuite.utils import gym
import gym
import time
import mujoco
from mujoco import viewer
from register_env import register_mme
from gym.vector import SyncVectorEnv
import gym, myosuite
from config import Config


def make_env():
    def _init():
        return gym.make("fullBodyWalk-v0")
    return _init



cfg = Config()
register_mme(cfg.model.model_path)
envs = SyncVectorEnv([make_env() for _ in range(cfg.model.num_env)])  
action = envs.action_space.sample()
envs.reset()
env = envs.envs[0]   
obs, reward, done, truncated, info  =   envs.step(action) # take a random action
print(info.keys()) #dict_keys(['final_observation', '_final_observation', 'final_info', '_final_info'])
print('===============environment info=====================')

sim_dt = env.sim.model.opt.timestep
frame_skip = env.frame_skip
ctrl_dt = env.dt


print(f"Simulation frequency: {1/sim_dt:.1f} Hz")
print(f"Control frequency: {1/ctrl_dt:.1f} Hz")

print('human action shape',action.shape[1])
print('human state shape',obs.shape[1])
#print('info dict',info['final_info'][0].keys())
print('state dict',info['final_info'][0]['obs_dict'].keys())
print('qpos_without_xy shape',info['final_info'][0]['obs_dict']['qpos_without_xy'].shape[0])
print('human torque shape',info['final_info'][0]['obs_dict']['human_torque'].shape[0])





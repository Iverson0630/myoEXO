from myosuite.utils import gym
import time
import mujoco
from mujoco import viewer
from register_env import register_mme
from config import Config


def make_env():
    def _init():
        return gym.make("fullBodyWalk-v0")
    return _init



cfg = Config()
register_mme(cfg.model.model_path)
envs = gym.vector.SyncVectorEnv([make_env() for _ in range(cfg.model.num_env)])
action = envs.action_space.sample()
envs.reset()
env = envs.envs[0]   
obs, reward, done, truncated, info  =   envs.step(action) # take a random action
print('===============environment info=====================')

sim_dt = env.sim.model.opt.timestep
frame_skip = env.frame_skip
ctrl_dt = env.dt


print(f"Simulation frequency: {1/sim_dt:.1f} Hz")
print(f"Control frequency: {1/ctrl_dt:.1f} Hz")
print("joint number:", env.sim.model.njnt)
print("dof number:", env.sim.model.nq)
print('human action shape:',action.shape)
print('human state shape:',obs.shape)
print('reward shape:',reward.shape)
print('info dict:',info.keys())
print('obs dict dims:', {k: getattr(v, "shape", ()) for k, v in info['obs_dict'][0].items()})
print('state dict dims:', {k: getattr(v, "shape", ()) for k, v in info['state'][0].items()})
print('reward dict:',info['rwd_dict'][0].keys())


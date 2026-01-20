import time
import mujoco
from mujoco import viewer
from register_env import make_env
from config import Config
cfg = Config()


def main():
    env = make_env(cfg.model.env)
    env.reset()
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)  # take a random action
    print("===============environment info=====================")

    sim_dt = env.sim.model.opt.timestep
    frame_skip = env.frame_skip

    ctrl_dt = env.dt

    print(f"Simulation frequency: {1/sim_dt:.1f} Hz")
    print(f"Control frequency: {1/ctrl_dt:.1f} Hz")
    print("joint number:", env.sim.model.njnt)
    print("dof number:", env.sim.model.nq)
    print("human action shape:", action.shape)
    print("human state shape:", obs.shape)
    print("reward shape:", reward.shape)
    print("info dict:", info.keys())
    print("obs dict dims:", {k: getattr(v, "shape", ()) for k, v in info["obs_dict"].items()})
    print("state dict dims:", {k: getattr(v, "shape", ()) for k, v in info["state"].items()})
    print("reward dict:", info["rwd_dict"].keys())


if __name__ == "__main__":
    main()

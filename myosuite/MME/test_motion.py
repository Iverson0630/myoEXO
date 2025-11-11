from myosuite.utils import gym
import time
import mujoco
from mujoco import viewer
from bvh import Bvh
import gym, myosuite, numpy as np


from gym.envs.registration import register

register(
    id="fullBodyWalk-v0",
    entry_point="myosuite.envs.myo.myobase.fullbodywalk_v0:FullBodyWalkEnvV0",
    max_episode_steps=200,
    kwargs={
        "model_path":  '../simhive/myo_sim/leg/myolegs.xml', 
    },
)

env = gym.make('fullBodyWalk-v0')
env.reset()

def bvh_play():
    with open("motion/walk.bvh") as f:
        mocap = Bvh(f.read())


    mapping = {
    # 右腿
    "Character1_RightUpLeg": "hip_flexion_r",
    "Character1_RightLeg": "knee_angle_r",
    "Character1_RightFoot": "ankle_angle_r",
    "Character1_RightToeBase": "mtp_angle_r",
    # 左腿
    "Character1_LeftUpLeg": "hip_flexion_l",
    "Character1_LeftLeg": "knee_angle_l",
    "Character1_LeftFoot": "ankle_angle_l",
    "Character1_LeftToeBase": "mtp_angle_l",

    }

    for i in range(1000):
        for f_idx in range(0, mocap.nframes):
            qpos = env.sim.data.qpos.copy()
            
            # 遍历每个 BVH 关节并赋值给 MuJoCo
            for bvh_name, mj_name in mapping.items():
                try:
                    jid = env.sim.model.name2id(mj_name, "joint")
                    addr = env.sim.model.jnt_qposadr[jid]
                    channels = mocap.joint_channels(bvh_name)
                    rot_channels = [ch for ch in channels if "rotation" in ch.lower()]
                    angle_deg = [float(mocap.frame_joint_channel(f_idx, bvh_name, ch)) for ch in rot_channels]

                    print(bvh_name,angle_deg)
                    qpos[addr] = np.deg2rad(angle_deg[0])  # 简化示例
                except KeyError:
                    pass
            
            env.sim.data.qpos[:] = qpos
            env.sim.forward()
            env.mj_render()

def log_muscle_joint():
    
    muscle_names = [env.sim.model.id2name(i, "actuator") for i in range(env.sim.model.nu)]


    forces = env.sim.data.actuator_force
    activations = env.sim.data.act

    for name, a, f in zip(muscle_names, activations, forces):
        print(f"{name:20s} | activation={a:.3f} | force={f:.1f}")

    for i in range(env.sim.model.njnt):
        name = env.sim.model.id2name(i, "joint")
        addr = env.sim.model.jnt_qposadr[i]
        print(f"{i:02d}  {name:20s}  qpos index: {addr}")


bvh_play()
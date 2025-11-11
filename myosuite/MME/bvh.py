from myosuite.utils import gym
import time
import mujoco
from mujoco import viewer
from bvh import Bvh
import gym, myosuite, numpy as np


from gym.envs.registration import register



def laod_bvh(env, path):
    with open(path) as f:
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
    qpos_list = []
    root_channels = mocap.joint_channels("Character1_Hips")
    root_trans = [ch for ch in root_channels if "position" in ch.lower()]
    initial_pos = [float(mocap.frame_joint_channel(0, "Character1_Hips", ch)) for ch in root_trans]

    # BVH 单位通常是 cm，MyoSuite 是 m
    initial_pos = np.array(initial_pos) / 100.0


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
                if bvh_name=='Character1_RightUpLeg' or  bvh_name=='Character1_LeftUpLeg':
                    angle_deg[1] =   - angle_deg[1]
                qpos[addr] = np.deg2rad(angle_deg[1])  # x rotation channel

            except KeyError:
                pass

        root_channels = mocap.joint_channels("Character1_Hips")
        root_trans = [ch for ch in root_channels if "position" in ch.lower()]
        pos_vals = [float(mocap.frame_joint_channel(f_idx, "Character1_Hips", ch)) for ch in root_trans]
        pos_vals = np.array(pos_vals) / 100.0 - initial_pos

        # BVH 单位通常是 cm，MyoSuite 是 m
        root_x, root_y, root_z = pos_vals
     
        #  mujoco x-> left  y->forward   z->updown
        #  bvh x-> left    y-> updown      z-> forward

        qpos[0:3] = qpos[0:3] + [ root_x, -root_z, root_y ]
        qpos_list.append(qpos.copy())
    return qpos_list

def bvh_play(env, path):
    for i in range(1000):
        qpos_list = laod_bvh(env, path)
      
        for j in range(len(qpos_list)):
            qpos = qpos_list[j]
            env.sim.data.qpos[4:] = qpos[4:]
            env.sim.data.qpos[0:3] = qpos[0:3]
            env.sim.forward()
            env.mj_render()
            time.sleep(0.1)

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

if __name__ == "__main__":

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
    bvh_play(env, "motion/walk.bvh")
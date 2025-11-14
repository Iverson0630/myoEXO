from myosuite.utils import gym
import time
import mujoco
from mujoco import viewer
from bvh import Bvh
import gym, myosuite, numpy as np

from scipy.spatial.transform import Rotation as R
from gym.envs.registration import register



def load_bvh(sim, path):
    """
    获取向量化 MyoSuite 环境的观测（obs）数组。

    支持:
    - SyncVectorEnv（含多个子环境）
    - 单个 MyoSuite 环境（BaseV0 派生类）

    返回:
    - obs_array: np.ndarray，形状为 (num_envs, obs_dim)
    - obs_dicts: 每个子环境的原始观测字典列表（方便分析）
    """
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
    joint_pos_list = []

    root_channels = mocap.joint_channels("Character1_Hips")
    root_trans = [ch for ch in root_channels if "position" in ch.lower()]
    initial_pos = [float(mocap.frame_joint_channel(0, "Character1_Hips", ch)) for ch in root_trans]

    # BVH 单位通常是 cm，MyoSuite 是 m
    initial_pos = np.array(initial_pos) / 100.0


    for f_idx in range(0, mocap.nframes):
        qpos = sim.data.qpos.copy()
        qpos[2] = sim.model.key_qpos[0][2] 
        # 遍历每个 BVH 关节并赋值给 MuJoCo
        for bvh_name, mj_name in mapping.items():
            try:
                jid = sim.model.name2id(mj_name, "joint")
                addr = sim.model.jnt_qposadr[jid]
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

        # spine_chaanles = mocap.joint_channels("Character1_Hips")
        # rot_channels = [ch for ch in spine_chaanles if "rotation" in ch.lower()]
        # rot_deg = [float(mocap.frame_joint_channel(f_idx, "Character1_Hips", ch)) for ch in rot_channels]
        # print(rot_deg)

        # rot_channels = [ch for ch in root_channels if "rotation" in ch.lower()]
        # rot_deg = [float(mocap.frame_joint_channel(f_idx, "Character1_Hips", ch)) for ch in rot_channels]
        # rot_rad = np.deg2rad(rot_deg)

        # # 根据指定旋转顺序生成四元数
        # rot = R.from_euler("ZXY", rot_rad)
        # quat_xyzw = rot.as_quat()  # scipy返回 [x, y, z, w]
        # fix_rot = R.from_euler("x", -90, degrees=True)
        # rot_fixed = fix_rot * rot
        # quat_xyzw = rot_fixed.as_quat()  #
     
        # # fix_rot = R.from_euler('x', -90, degrees=True)
        # # rot_fixed = fix_rot * R.from_quat(quat_xyzw)
        # # quat_xyzw_fixed = rot_fixed.as_quat()
        # # quat_wxyz_fixed = np.array([quat_xyzw_fixed[3], quat_xyzw_fixed[0],
        # #                             quat_xyzw_fixed[1], quat_xyzw_fixed[2]])
        #qpos[3:7] = -quat_xyzw

        # BVH 单位通常是 cm，MyoSuite 是 m
        root_x, root_y, root_z = pos_vals
     
        #  mujoco x-> left  y->forward   z->updown
        #  bvh x-> left    y-> updown      z-> forward
    
        qpos[0:3] = qpos[0:3] + [ root_x, -root_z, root_y]
        
        qpos_list.append(qpos.copy())
    return qpos_list, mocap.frame_time, mapping

def bvh_play(env, path):

    for i in range(1000):
        qpos_list,_ ,_= load_bvh(env.sim, path)
      
        for j in range(len(qpos_list)):
       
            qpos = qpos_list[j]
            print('frame', j ,qpos[7:])
            env.sim.data.qpos[4:] = qpos[4:]
            env.sim.data.qpos[0:3] = qpos[0:3]
            env.sim.forward()
            env.mj_render()
            time.sleep(0.1)

# from test_motion_opensim import read_mot
# def load_osim(sim):
#     url = "https://raw.githubusercontent.com/opensim-org/opensim-models/refs/heads/master/Pipelines/Gait2392_Simbody/OutputReference/subject01_walk1_ik.mot"
#     df = read_mot(url)
#     joint_names= []
#     qpos_list = []

#     for i in range(env.sim.model.njnt):
#         joint_names.append( sim.model.id2name(i, "joint"))
#     subc = [c for c in df.columns if c in joint_names]
#     for t in range(len(df)):
#         qpos = sim.data.qpos.copy()
       
#         for jn in subc:
#             angle =  np.deg2rad(df[jn].loc[t])
#             if "knee_angle" in jn:  # knee joints have negative sign in myosuite
#                 angle*= -1
     
#         qpos_list.append(angles)
#     return qpos_list
 
# def osim_play(env):
    
#     qpos_list = load_osim(env.sim)
#     for i in range(1000):
#         qpos_list = load_osim(env.sim)

#         for j in range(len(qpos_list)):
       
#             qpos = qpos_list[j]
#             print('frame', j ,qpos[0:3])
#             env.sim.data.qpos[4:] = qpos[4:]
#             env.sim.forward()
#             env.mj_render()
#             time.sleep(0.1)

if __name__ == "__main__":

    register(
        id="fullBodyWalk-v0",
        entry_point="env.fullbodywalk_v0:FullBodyWalkEnvV0",
        max_episode_steps=200,
        kwargs={
            "model_path":  '../simhive/myo_sim/leg/myolegs.xml', 
        },
    )

    env = gym.make('fullBodyWalk-v0')
    env.reset()
    bvh_play(env, "motion/walk.bvh")
    #osim_play(env)
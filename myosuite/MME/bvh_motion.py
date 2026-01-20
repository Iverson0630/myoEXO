
import time
import mujoco
from mujoco import viewer
from bvh import Bvh
import numpy as np
from scipy.spatial.transform import Rotation as R
from register_env import register_mme
from myosuite.utils import gym
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

    # Each entry maps BVH joint -> list of (mujoco_joint, bvh_axis, scale).
    mapping = {
    # 右腿
    "Character1_RightUpLeg": [
        ("hip_flexion_r", "X", 1.0),
        #("hip_adduction_r", "Z", 1.0),
        ("hip_rotation_r", "Y", 1.0),
    ],
    "Character1_RightLeg": [("knee_angle_r", "X", 1.0)],
    "Character1_RightFoot": [("ankle_angle_r", "X", 1.0)],
    "Character1_RightToeBase": [("mtp_angle_r", "X", 1.0)],
    # 左腿
    "Character1_LeftUpLeg": [
        ("hip_flexion_l", "X", 1.0),
        #("hip_adduction_l", "Z", 1.0),
        ("hip_rotation_l", "Y", 1.0),
    ],
    "Character1_LeftLeg": [("knee_angle_l", "X", 1.0)],
    "Character1_LeftFoot": [("ankle_angle_l", "X", 1.0)],
    "Character1_LeftToeBase": [("mtp_angle_l", "X", 1.0)],
    # 躯干（分配到三个腰椎自由度）
    # "Character1_Spine": [
    #     ("flex_extension", "X", 0.5),
    #     ("lat_bending", "Z", 0.5),
    #     ("axial_rotation", "Y", 0.5),
    # ],
    # "Character1_Spine1": [
    #     ("flex_extension", "X", 0.5),
    #     ("lat_bending", "Z", 0.5),
    #     ("axial_rotation", "Y", 0.5),
    # ],
    # 头颈
    "Character1_Neck": [("neck_rotation", "Y", 1.0)],
    "Character1_Head": [("neck_flexion", "X", 1.0)],
    # 右臂（简化关节优先）
    "Character1_RightShoulder": [
        ("arm_flex_l", "X", 5.0),
        # ("arm_add_r", "Y", 1.0),
        # ("arm_rot_r", "Z", 1.0),
        ("unrothum_l1", "Z", 1.0),
        ("unrothum_l2", "Y", 1.0),
        ("unrothum_l3", "X", 5.0),
    ],
    "Character1_RightArm": [],
    "Character1_RightForeArm": [
        ("elbow_flex_r", "X", 1.0),
        ("elbow_flexion_r", "X", 1.0),
        ("pro_sup_r", "Y", 1.0),
    ],
    "Character1_RightHand": [
        ("wrist_flex_r", "X", 1.0),
        ("wrist_dev_r", "Z", 1.0),
        ("flexion_r", "X", 1.0),
        ("deviation_r", "Z", 1.0),
    ],
    # 左臂（简化关节优先）
    "Character1_LeftShoulder": [
        ("arm_flex_r", "X", 5.0),
        # ("arm_add_l", "Y", 1.0),
        # ("arm_rot_l", "Z", 1.0),
        ("unrothum_r1", "Z", 1.0),
        ("unrothum_r2", "Y", 1.0),
        ("unrothum_r3", "X", 5.0),
    ],
    "Character1_LeftArm": [],
    "Character1_LeftForeArm": [
        ("elbow_flex_l", "X", 1.0),
        ("elbow_flexion_l", "X", 1.0),
        ("pro_sup_l", "Y", 1.0),
    ],
    "Character1_LeftHand": [
        ("wrist_flex_l", "X", 1.0),
        ("wrist_dev_l", "Z", 1.0),
        ("flexion_l", "X", 1.0),
        ("deviation_l", "Z", 1.0),
    ],
    }
    qpos_list = []
    joint_pos_list = []

    root_channels = mocap.joint_channels("Character1_Hips")
    root_trans = [ch for ch in root_channels if "position" in ch.lower()]
    initial_pos = [float(mocap.frame_joint_channel(0, "Character1_Hips", ch)) for ch in root_trans]

    # BVH 单位通常是 cm，MyoSuite 是 m
    initial_pos = np.array(initial_pos) / 100.0

 
    #  mujoco x-> left  y->forward   z->updown
    #  bvh x-> left    y-> updown      z-> forward
    for f_idx in range(0, mocap.nframes):
        qpos = sim.data.qpos.copy()
       
        # 遍历每个 BVH 关节并赋值给 MuJoCo（多关节/多轴映射）
        joint_values = {}
        for bvh_name, mj_mappings in mapping.items():
            try:
                channels = mocap.joint_channels(bvh_name)
            except Exception:
                continue
            rot_values = {}
            for ch in channels:
                if "rotation" not in ch.lower():
                    continue
                axis = ch[0].upper()
                rot_values[axis] = float(mocap.frame_joint_channel(f_idx, bvh_name, ch))

            for mj_name, axis, scale in mj_mappings:
                if axis not in rot_values:
                    continue
                angle_deg = rot_values[axis] * scale
                if bvh_name in ("Character1_RightUpLeg", "Character1_LeftUpLeg") and axis == "X":
                    angle_deg = -angle_deg
                joint_values[mj_name] = joint_values.get(mj_name, 0.0) + np.deg2rad(angle_deg)

        for mj_name, angle_rad in joint_values.items():
            try:
                jid = sim.model.name2id(mj_name, "joint")
                addr = sim.model.jnt_qposadr[jid]
                qpos[addr] = angle_rad
            except Exception:
                # Skip joints that are not present in this model.
                pass

    

        root_channels = mocap.joint_channels("Character1_Hips")
        root_trans = [ch for ch in root_channels if "position" in ch.lower()]
        pos_vals = [float(mocap.frame_joint_channel(f_idx, "Character1_Hips", ch)) for ch in root_trans]
        pos_vals = np.array(pos_vals) / 100.0 - initial_pos

        # spine_chaanles = mocap.joint_channels("Character1_Hips")
        # rot_channels = [ch for ch in spine_chaanles if "rotation" in ch.lower()]
        # rot_deg = [float(mocap.frame_joint_channel(f_idx, "Character1_Hips", ch)) for ch in rot_channels]
  
        # rot_channels = [ch for ch in root_channels if "rotation" in ch.lower()]
        # rot_deg = [float(mocap.frame_joint_channel(f_idx, "Character1_Hips", ch)) for ch in rot_channels]
        # rot_rad = np.deg2rad(rot_deg)

        # # 根据指定旋转顺序生成四元数
        # rot = R.from_euler("ZXY", rot_rad)
        # quat_xyzw = rot.as_quat()  # scipy返回 [x, y, z, w]
        # fix_rot = R.from_euler("x", -90, degrees=True)
        # rot_fixed = fix_rot * rot
        # quat_xyzw = rot_fixed.as_quat()  #
     
        # fix_rot = R.from_euler('x', -90, degrees=True)
        # rot_fixed = fix_rot * R.from_quat(quat_xyzw)
        # quat_xyzw_fixed = rot_fixed.as_quat()
        # quat_wxyz_fixed = np.array([quat_xyzw_fixed[3], quat_xyzw_fixed[0],
        #                             quat_xyzw_fixed[1], quat_xyzw_fixed[2]])
        # qpos[3:7] = -quat_xyzw

        # BVH 单位通常是 cm，MyoSuite 是 m
        root_x, root_y, root_z = pos_vals
    
    
        qpos[0:3] = qpos[0:3] + [ root_x, -root_z, root_y]
        
        qpos_list.append(qpos.copy())
    return qpos_list, mocap.frame_time, mapping

def bvh_play(env, path):

    for i in range(1000):
        qpos_list,_ ,_= load_bvh(env.sim, path)
      
        for j in range(len(qpos_list)):
       
            qpos = qpos_list[j]

            env.sim.data.qpos[4:] = qpos[4:]
            env.sim.data.qpos[0:3] = qpos[0:3]
            env.sim.forward()
            env.mj_render()
            time.sleep(0.1)


if __name__ == "__main__":
    register_mme()
    env = gym.make('fullBodyWalk-v0')
    env.reset()
    bvh_play(env, "motion/walk.bvh")

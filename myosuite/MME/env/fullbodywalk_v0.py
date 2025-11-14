"""=================================================
# Copyright (c) MyoSuite Authors
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Pierre Schumacher (schumacherpier@gmail.com), Cameron Berg (cam.h.berg@gmail.com)
================================================="""

import collections

import numpy as np

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils import gym
from myosuite.utils.quat_math import quat2mat
from bvh_motion import load_bvh
from config import Config
class FullBodyWalkEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy",
        "qvel",
        "com_vel",
        "torso_angle",
        "feet_heights",
        "height",
        "feet_rel_positions",
        "phase_var",
        "muscle_length",
        "muscle_velocity",
        "muscle_force",
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "joint_rwd": 0.5,
        "done": 0,
        "pos_rwd": 0.2,
        "ori_rwd": 0.3,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(
            model_path=model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            env_credits=self.MYO_CREDIT,
        )

        # load bvh file
        bvh_path = Config().model.bvh_path
        self.qpos_bvh,self.frame_time, self.mapping = load_bvh(self.sim, bvh_path)
        self.bvh_start_frame = 0


        self._setup(**kwargs)

    def _setup(
        self,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        min_height= 0.9,
        max_rot=0.8,
        hip_period=100,
        reset_type="init",
        target_x_vel=0.0,
        target_y_vel=1.2,
        target_rot=None,
        **kwargs,
    ):
        self.min_height = min_height
        self.max_rot = max_rot
        self.hip_period = hip_period
        self.reset_type = reset_type
        self.target_x_vel = target_x_vel
        self.target_y_vel = target_y_vel
        self.target_rot = target_rot
        self.steps = 0
        super()._setup(
            obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, **kwargs
        )

        self.init_qpos[:] =  self.qpos_bvh[0]
        self.init_qvel[:] = 0.0


        # move heightfield down if not used
        # self.sim.model.geom_rgba[self.sim.model.geom_name2id("terrain")][-1] = 0.0
        # self.sim.model.geom_pos[self.sim.model.geom_name2id("terrain")] = np.array(
        #     [0, 0, -10]
        # )

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["t"] = np.array([sim.data.time])
        obs_dict["time"] = np.array([sim.data.time])
        obs_dict["qpos_without_xy"] = sim.data.qpos[2:].copy()
        obs_dict["qvel"] = sim.data.qvel[:].copy() * self.dt
        obs_dict["com_vel"] = np.array([self._get_com_velocity().copy()])
        obs_dict["torso_angle"] = np.array([self._get_torso_angle().copy()])
        obs_dict["feet_heights"] = self._get_feet_heights().copy()
        obs_dict["height"] = np.array([self._get_height()]).copy()
        obs_dict["feet_rel_positions"] = self._get_feet_relative_position().copy()
        obs_dict["phase_var"] = np.array([(self.steps / self.hip_period) % 1]).copy()
        obs_dict["muscle_length"] = self.muscle_lengths()
        obs_dict["muscle_velocity"] = self.muscle_velocities()
        obs_dict["muscle_force"] = self.muscle_forces()
        obs_dict["human_torque"] = self.human_torque()
        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()

        return obs_dict

     

    def get_reward_dict(self, obs_dict):
        self.preprocess_bvh_vel()
        t = self.sim.data.time / self.frame_time
        frame0 = int(np.floor(t)) % len(self.qpos_bvh)
        frame_idx = (self.bvh_start_frame + frame0) % len(self.qpos_bvh)
        self.qpos_ref = self.qpos_bvh[frame_idx]
        self.qvel_ref = self.qvel_bvh[frame_idx]

        joint_pos_rwd = self._get_all_joint_rwd()
        joint_vel_rwd = self._get_all_joint_vel_rwd()
        joint_rwd = (joint_pos_rwd+joint_vel_rwd)/2
        root_rwd = self._get_root_rwd()
        vel_rwd = self._get_pos_rwd()
        pos_rwd = root_rwd[0:2]
        ori_rwd = root_rwd[3:7]
        # print("===============================")
        # print('joint pos reward: ', np.floor(np.array(joint_pos_rwd) * 100) / 100,  np.mean(joint_pos_rwd))
        # print('joint vel reward: ', np.floor(np.array(joint_vel_rwd) * 100) / 100,  np.mean(joint_vel_rwd))
        # print('root pos reward: ', np.floor(np.array(pos_rwd) * 100) / 100,  np.mean(pos_rwd))
        # print('root ori reward: ', np.floor(np.array(ori_rwd) * 100) / 100,  np.mean(root_rwd))
        #ee_rwd = self._get
        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                ("joint_rwd", np.mean(joint_rwd)),
                ("pos_rwd", np.mean(pos_rwd)),
                ("ori_rwd", np.mean(ori_rwd)),
                # Must keys
                ("sparse", joint_rwd),
                ("solved", joint_rwd >= 0.9),
                ("done", self._get_done()),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict

    def _get_target_joint_rwd(self):
        joint = []
        ref_joint = []
        for bvh_name, mujoco_name in self.mapping.items():
            jid = self.sim.model.joint_name2id(mujoco_name)
            adr = self.sim.model.jnt_qposadr[jid]
            joint.append(self.sim.data.qpos[adr])
            ref_joint.append(self.qpos_ref[adr])
        # print('=======================================')
        # print('real joint : ', np.floor(np.array(joint) * 100) / 100)
        # print('refer joint : ', np.floor(np.array(ref_joint) * 100) / 100)
        return self.exp_of_squared(np.mean(np.array(joint) - np.array(ref_joint))*180/np.pi, 10.0)
    
    def _get_all_joint_rwd(self):
        # print('=======================================')
        # print('real joint : ', np.floor(np.array(self.sim.data.qpos[7:]) * 100) / 100)
        # print('refer joint : ', np.floor(np.array(self.qpos_ref[7:]) * 100) / 100)

        diff = (self.sim.data.qpos[7:] - self.qpos_ref[7:]) * 180 / np.pi
        return self.exp_of_squared(np.mean(diff),0.1)
    
    def _get_all_joint_vel_rwd(self):
        vdiff = (self.sim.data.qvel[6:] - self.qvel_ref[7:]) * 180 / np.pi
        return  self.exp_of_squared(np.mean(vdiff),1.0)
    def _get_root_rwd(self):    
        root_pos =  self.sim.data.qpos[0:7]

        # print('real pos : ', np.floor(np.array(root_pos) * 100) / 100)   
        # print('refer pos: ', np.floor(np.array(self.qpos_ref[0:7]) * 100) / 100)
        return self.exp_of_squared((root_pos - self.qpos_ref[0:7])*100, 0.5)
    
    def _get_pos_rwd(self):
        # 获取 pelvis（或 root）的线速度
        root_vel = self.sim.data.qvel[0:3]  # root 的线速度 [vx, vy, vz]
        vx = root_vel[1]

        # 奖励接近期望速度（高斯形式）
        return np.exp(-2.0 * (vx - 0.8)**2)


    def exp_of_squared(self, val, w):
        return np.exp(-w*val*val)
    
    def get_randomized_initial_state(self):
        # randomly start with flexed left or right knee
        if self.np_random.uniform() < 0.5:
            qpos = self.sim.model.key_qpos[2].copy()
            qvel = self.sim.model.key_qvel[2].copy()
        else:
            qpos = self.sim.model.key_qpos[3].copy()
            qvel = self.sim.model.key_qvel[3].copy()

        # randomize qpos coordinates
        # but dont change height or rot state
        rot_state = qpos[3:7]
        height = qpos[2]
        qpos[:] = qpos[:] + self.np_random.normal(0, 0.02, size=qpos.shape)
        qpos[3:7] = rot_state
        qpos[2] = height
        return qpos, qvel

    def step(self, *args, **kwargs):
        results = super().step(*args, **kwargs)
        self.steps += 1
        return results

    def reset(self, **kwargs):
        self.steps = 0
        self.bvh_start_frame = 0# np.random.randint(0, len(self.qpos_bvh))
   
        if self.reset_type == "random":
            qpos, qvel = self.get_randomized_initial_state()
        elif self.reset_type == "init":
        
            qpos, qvel = self.qpos_bvh[0], self.sim.model.key_qvel[0] #reset the pos of bvh

        else:
            qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]
        self.robot.sync_sims(self.sim, self.sim_obsd)
      
        obs = super().reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        return obs

    def muscle_lengths(self):
        return self.sim.data.actuator_length

    def muscle_forces(self):
        return np.clip(self.sim.data.actuator_force / 1000, -100, 100)

    def muscle_velocities(self):
        return np.clip(self.sim.data.actuator_velocity, -100, 100)

    def human_torque(self):
        return self.sim.data.qfrc_actuator
    
    def _get_done(self):
        height = self._get_height()

        if height < self.min_height:
            return 1
        # if self._get_rot_condition():
        #     return 1
        return 0


    def _get_feet_heights(self):
        """
        Get the height of both feet.
        """
        foot_id_l = self.sim.model.body_name2id("talus_l")
        foot_id_r = self.sim.model.body_name2id("talus_r")
        return np.array(
            [
                self.sim.data.body_xpos[foot_id_l][2],
                self.sim.data.body_xpos[foot_id_r][2],
            ]
        )
    def preprocess_bvh_vel(self):
        self.qpos_bvh =  np.array(self.qpos_bvh)  # shape [T, nq]
        T, nq = self.qpos_bvh.shape
        
        qvel = np.zeros_like(self.qpos_bvh)

        fps = 1.0 / self.frame_time

        # 一阶差分计算速度
        qvel[1:] = (self.qpos_bvh[1:] - self.qpos_bvh[:-1]) * fps
        qvel[0] = qvel[1]

        self.qvel_bvh = qvel

    def _get_feet_relative_position(self):
        """
        Get the feet positions relative to the pelvis.
        """
        foot_id_l = self.sim.model.body_name2id("talus_l")
        foot_id_r = self.sim.model.body_name2id("talus_r")
        pelvis = self.sim.model.body_name2id("pelvis")
        return np.array(
            [
                self.sim.data.body_xpos[foot_id_l] - self.sim.data.body_xpos[pelvis],
                self.sim.data.body_xpos[foot_id_r] - self.sim.data.body_xpos[pelvis],
            ]
        )

    def _get_ref_rotation_rew(self):
        """
        Incentivize staying close to the initial reference orientation up to a certain threshold.
        """
        target_rot = [
            self.target_rot if self.target_rot is not None else self.init_qpos[3:7]
        ][0]
        return np.exp(-np.linalg.norm(5.0 * (self.sim.data.qpos[3:7] - target_rot)))

    def _get_torso_angle(self):
        body_id = self.sim.model.body_name2id("torso")
        return self.sim.data.body_xquat[body_id]

    def _get_com_velocity(self):
        """
        Compute the center of mass velocity of the model.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = -self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]

    def _get_height(self):
        """
        Get center-of-mass height.
        """
        return self._get_com()[2]

    def _get_rot_condition(self):
        """
        MuJoCo specifies the orientation as a quaternion representing the rotation
        from the [1,0,0] vector to the orientation vector. To check if
        a body is facing in the right direction, we can check if the
        quaternion when applied to the vector [1,0,0] as a rotation
        yields a vector with a strong x component.
        """
        # quaternion of root
        quat = self.sim.data.qpos[3:7].copy()

        return [1 if np.abs((quat2mat(quat) @ [1, 0, 0])[0]) < self.max_rot else 0][0]

    def _get_com(self):
        """
        Compute the center of mass of the robot.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        com = self.sim.data.xipos
        return np.sum(mass * com, 0) / np.sum(mass)

    def _get_angle(self, names):
        """
        Get the angles of a list of named joints.
        """
        return np.array(
            [
                self.sim.data.qpos[
                    self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(name)]
                ]
                for name in names
            ]
        )




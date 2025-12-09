import math
import random
import time
import os
import sys
from datetime import datetime

import collections
from collections import namedtuple
from collections import deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from Model import *

import gym
import  myosuite
import time

from mujoco import viewer
from register_env import register_mme
from gym.vector import SyncVectorEnv
from config import Config
import wandb  

use_cuda = torch.cuda.is_available()



class Render():
	def __init__(self):
		#np.random.seed(seed = int(time.time()))
	
		self.cfg = Config()
		register_mme(self.cfg.model.model_path)
		self.env = gym.make(self.cfg.model.env)  
	
		self.initial_obs = self.env.reset()
		action = self.env.action_space.sample()

		self.num_state = self.initial_obs.shape[0]
		self.num_action = action.shape[0]


		timestep = self.env.sim.model.opt.timestep
		frame_skip = self.env.frame_skip
		sim_freq = 1.0 / timestep
		ctrl_freq = 1.0 / (timestep * frame_skip)


		print('===============environment info=====================')
		print('env name',self.cfg.model.env)
		print('human state shape',self.num_state)
		print('human action shape',self.num_action)
		print('simulation frequence',sim_freq)
		print('control frequence',ctrl_freq)
		self.model = SimulationHumanNN(self.num_state,self.num_action)

	def LoadModel(self,path):
		self.model.load(path+'/max.pt')

		if use_cuda:
			self.model.cuda()
	
	def forward(self,max_step=10000):
		self.env.reset()
		for i in range(max_step):
			states = self.get_state()
			a_dist,v = self.model(Tensor(states))
		
			actions = a_dist.loc.cpu().detach().numpy().squeeze()
	
			obs, rewards, done, info  = self.env.step(actions)
		
			time.sleep(0.1)
			if done:
				self.env.reset()
			self.env.mj_render()

	def get_state(self):
		"""
		获取向量化 MyoSuite 环境的观测（obs）数组。

		支持:
		- SyncVectorEnv（含多个子环境）
		- 单个 MyoSuite 环境（BaseV0 派生类）

		返回:
		- obs_array: np.ndarray，形状为 (num_envs, obs_dim)
		- obs_dicts: 每个子环境的原始观测字典列表（方便分析）
		"""
		# 判断是否为向量化环境
		if hasattr(self.env, "envs"):
			num_envs = len(self.envs.envs)
			obs_list = []
			obs_dicts = []
			for i, e in enumerate(self.envs.envs):
				obs_dict = e.get_obs_dict(e.sim)
				obs_vec = np.concatenate([obs_dict[k].ravel() for k in e.obs_keys])
				obs_list.append(obs_vec)
				obs_dicts.append(obs_dict)
			obs_array = np.stack(obs_list)
		else:
			# 单环境情况
			obs_dict = self.env.get_obs_dict(self.env.sim)
			obs_array = np.concatenate([obs_dict[k].ravel() for k in self.env.obs_keys])[None, :]
			obs_dicts = [obs_dict]
		return obs_array


	

	def Evaluate(self):
		self.num_evaluation = self.num_evaluation + 1
		h = int((time.time() - self.tic)//3600.0)
		m = int((time.time() - self.tic)//60.0)
		s = int((time.time() - self.tic))
		m = m - h*60
		s = int((time.time() - self.tic))
		s = s - h*3600 - m*60
		if self.num_episode == 0:
			self.num_episode = 1
		if self.num_tuple == 0:
			self.num_tuple = 1
		if self.max_return < self.sum_return/self.num_episode:
			self.max_return = self.sum_return/self.num_episode
			self.max_return_epoch = self.num_evaluation
		print('# {} === {}h:{}m:{}s ==='.format(self.num_evaluation,h,m,s))
		print('||Loss Actor               : {:.4f}'.format(self.loss_actor))
		print('||Loss Critic              : {:.4f}'.format(self.loss_critic))
		print('||Noise                    : {:.3f}'.format(self.model.log_std.exp().mean()))		
		print('||Num Transition So far    : {}'.format(self.num_tuple_so_far))
		print('||Num Transition           : {}'.format(self.num_tuple))
		print('||Num Episode              : {}'.format(self.num_episode))
		print('||Avg Return per episode   : {:.3f}'.format(self.sum_return/self.num_episode))
		print('||Avg Reward per transition: {:.3f}'.format(self.sum_return/self.num_tuple))
		print('||Avg Step per episode     : {:.1f}'.format(self.num_tuple/self.num_episode))
		print('||Max Avg Retun So far     : {:.3f} at #{}'.format(self.max_return,self.max_return_epoch))

	

import argparse
import os
if __name__=="__main__":
	config = Config()
	
	render = Render()
	
	nn_dir =config.save_dir.nn_dir
	render.LoadModel(nn_dir)
	render.forward()
  

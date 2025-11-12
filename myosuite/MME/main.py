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
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
Episode = namedtuple('Episode',('s','a','r', 'value', 'logprob'))
class EpisodeBuffer(object):
	def __init__(self):
		self.data = []

	def Push(self, *args):
		self.data.append(Episode(*args))
	def Pop(self):
		self.data.pop()
	def GetData(self):
		return self.data
MuscleTransition = namedtuple('MuscleTransition',('JtA','tau_des','L','b'))
class MuscleBuffer(object):
	def __init__(self, buff_size = 10000):
		super(MuscleBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def Push(self,*args):
		self.buffer.append(MuscleTransition(*args))

	def Clear(self):
		self.buffer.clear()
Transition = namedtuple('Transition',('s','a', 'logprob', 'TD', 'GAE'))
class ReplayBuffer(object):
	def __init__(self, buff_size = 10000):
		super(ReplayBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def Push(self,*args):
		self.buffer.append(Transition(*args))

	def Clear(self):
		self.buffer.clear()

def make_env():
    def _init():
        return gym.make("fullBodyWalk-v0")
    return _init

class PPO(object):
	def __init__(self):
		np.random.seed(seed = int(time.time()))
	
		self.cfg = Config()
		register_mme(self.cfg.model.model_path)
		self.envs = SyncVectorEnv([make_env() for _ in range(self.cfg.model.num_env)])  
	



		self.num_evaluation = 0
		self.num_tuple_so_far = 0
		self.num_episode = 0
		self.num_tuple = 0

		timestep = self.envs.envs[0].sim.model.opt.timestep
		frame_skip = self.envs.envs[0].frame_skip
		sim_freq = 1.0 / timestep
		ctrl_freq = 1.0 / (timestep * frame_skip)

		self.num_simulation_Hz = sim_freq
		self.num_control_Hz = ctrl_freq #self.envs.GetControlHz()
		self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

		self.gamma = 0.99
		self.lb = 0.99

		self.num_slaves = self.cfg.model.num_env
		self.num_epochs = self.cfg.train.num_epochs
		self.buffer_size = self.cfg.train.buffer_size
		self.batch_size = self.cfg.train.batch_size
		self.default_learning_rate = self.cfg.train.default_learning_rate
		self.default_clip_ratio = self.cfg.train.default_clip_ratio
		self.max_iteration = self.cfg.train.max_iteration


	
		self.replay_buffer = ReplayBuffer(30000)

		self.initial_obs = self.envs.reset()
		action = self.envs.action_space.sample()

		self.num_state = self.initial_obs[0].shape[1]
		self.num_action = action.shape[1]
		self.model = SimulationHumanNN(self.num_state,self.num_action)

		if use_cuda:
			self.model.cuda()
	

	
		self.learning_rate = self.default_learning_rate
		self.clip_ratio = self.default_clip_ratio
		self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)
	

		self.w_entropy = -0.001

		self.loss_actor = 0.0
		self.loss_critic = 0.0

		self.rewards = []
		self.sum_return = 0.0
		self.max_return = -1.0
		self.max_return_epoch = 1
		self.tic = time.time()

		self.episodes = [None]*self.num_slaves
		for j in range(self.num_slaves):
			self.episodes[j] = EpisodeBuffer()
		print('===============environment info=====================')
		print('human state shape',self.num_state)
		print('human action shape',self.num_action)

		print('simulation frequence',self.num_simulation_Hz)
		print('control frequence',self.num_control_Hz)
	def SaveModel(self):
		self.model.save(nn_dir+'/current.pt')
	
		
		if self.max_return_epoch == self.num_evaluation:
			self.model.save(nn_dir+'/max.pt')
	
		if self.num_evaluation%100 == 0:
			self.model.save(nn_dir+'/'+str(self.num_evaluation//100)+'.pt')
			

	def LoadModel(self,path):
		self.model.load('../nn/'+path+'.pt')
	

	def ComputeTDandGAE(self):
		self.replay_buffer.Clear()
	
		self.sum_return = 0.0
		for epi in self.total_episodes:
			data = epi.GetData()
			size = len(data)
			if size == 0:
				continue
			states, actions, rewards, values, logprobs = zip(*data)

			values = np.concatenate((values, np.zeros(1)), axis=0)
			advantages = np.zeros(size)
			ad_t = 0

			epi_return = 0.0
			for i in reversed(range(len(data))):
				epi_return += rewards[i]
				delta = rewards[i] + values[i+1] * self.gamma - values[i]
				ad_t = delta + self.gamma * self.lb * ad_t
				advantages[i] = ad_t
			self.sum_return += epi_return
			TD = values[:size] + advantages
			
			for i in range(size):
				self.replay_buffer.Push(states[i], actions[i], logprobs[i], TD[i], advantages[i])
		self.num_episode = len(self.total_episodes)
		self.num_tuple = len(self.replay_buffer.buffer)
		print('SIM : {}'.format(self.num_tuple))
		self.num_tuple_so_far += self.num_tuple


	import numpy as np

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
		if hasattr(self.envs, "envs"):
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
			obs_dict = self.envs.get_obs_dict(self.envs.sim)
			obs_array = np.concatenate([obs_dict[k].ravel() for k in self.envs.obs_keys])[None, :]
			obs_dicts = [obs_dict]
		return obs_array


	def GenerateTransitions(self):
		
		self.total_episodes = []
		states = [None]*self.num_slaves
		actions = [None]*self.num_slaves
		rewards = [None]*self.num_slaves
		states_next = [None]*self.num_slaves
		states = self.get_state()
	
		local_step = 0
		terminated = [False]*self.num_slaves
		counter = 0
		while True:
			counter += 1
			if counter%100 == 0:
				print('SIM : {}'.format(local_step),end='\r')
			a_dist,v = self.model(Tensor(states))
			actions = a_dist.sample().cpu().detach().numpy()
		
			logprobs = a_dist.log_prob(Tensor(actions)).cpu().detach().numpy().reshape(-1)
			values = v.cpu().detach().numpy().reshape(-1)
		

			obs, rewards, done, truncated, info  = self.envs.step(actions)

			# self.envs.envs[0].mj_render()

			for j in range(self.num_slaves):
			
				nan_occur = False
				terminated_state = True

				if np.any(np.isnan(states[j])) or np.any(np.isnan(actions[j])) or np.any(np.isnan(states[j])) or np.any(np.isnan(values[j])) or np.any(np.isnan(logprobs[j])):
					print('nan occur')
					nan_occur = True
				
				elif done[j] == False:
	
					terminated_state = False
					self.episodes[j].Push(states[j], actions[j], rewards[j], values[j], logprobs[j])
					local_step += 1

				if terminated_state or (nan_occur==True):
					if (nan_occur is True):
						self.episodes[j].Pop()
					self.total_episodes.append(self.episodes[j])
					self.episodes[j] = EpisodeBuffer()

					self.envs.envs[j].reset()

			if local_step >= self.buffer_size:
				break
				
			states = self.get_state()
		
	def OptimizeSimulationNN(self):
		all_transitions = np.array(self.replay_buffer.buffer, dtype=object)
		for j in range(self.num_epochs):
			np.random.shuffle(all_transitions)
			for i in range(len(all_transitions)//self.batch_size):
				transitions = all_transitions[i*self.batch_size:(i+1)*self.batch_size]
				batch = Transition(*zip(*transitions))

				stack_s = np.vstack(batch.s).astype(np.float32)
				stack_a = np.vstack(batch.a).astype(np.float32)
				stack_lp = np.vstack(batch.logprob).astype(np.float32)
				stack_td = np.vstack(batch.TD).astype(np.float32)
				stack_gae = np.vstack(batch.GAE).astype(np.float32)
				
				a_dist,v = self.model(Tensor(stack_s))
				'''Critic Loss'''
				loss_critic = ((v-Tensor(stack_td)).pow(2)).mean()
				
				'''Actor Loss'''
				ratio = torch.exp(a_dist.log_prob(Tensor(stack_a))-Tensor(stack_lp))
				stack_gae = (stack_gae-stack_gae.mean())/(stack_gae.std()+ 1E-5)
				stack_gae = Tensor(stack_gae)
				surrogate1 = ratio * stack_gae
				surrogate2 = torch.clamp(ratio,min =1.0-self.clip_ratio,max=1.0+self.clip_ratio) * stack_gae
				loss_actor = - torch.min(surrogate1,surrogate2).mean()
				'''Entropy Loss'''
				loss_entropy = - self.w_entropy * a_dist.entropy().mean()

				self.loss_actor = loss_actor.cpu().detach().numpy().tolist()
				self.loss_critic = loss_critic.cpu().detach().numpy().tolist()
				
				loss = loss_actor + loss_entropy + loss_critic

				self.optimizer.zero_grad()
				loss.backward(retain_graph=True)
				for param in self.model.parameters():
					if param.grad is not None:
						param.grad.data.clamp_(-0.5,0.5)
				self.optimizer.step()
			print('Optimizing sim nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		print('')

	def generate_shuffle_indices(self, batch_size, minibatch_size):
		n = batch_size
		m = minibatch_size
		p = np.random.permutation(n)

		r = m - n%m
		if r>0:
			p = np.hstack([p,np.random.randint(0,n,r)])

		p = p.reshape(-1,m)
		return p
	def OptimizeModel(self):
		self.ComputeTDandGAE()
		self.OptimizeSimulationNN()
	
	def Train(self):
		self.GenerateTransitions()
		self.OptimizeModel()


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

	
		self.SaveModel()
		
		print('=============================================')
		wandb.log({ 
            "Loss actor human": self.loss_actor, 
            "Loss critic human": self.loss_critic,  
			"Num Transition": self.num_tuple,  
			"Num Episode": self.num_episode, 
			"Avg Human Return per episode": self.sum_return/self.num_episode,
			"Avg Human Reward per transition": self.sum_return/self.num_tuple,  
			"Avg Human Effort per episode": self.sum_return/self.num_episode,
			"Avg Step per episode": self.num_tuple/self.num_episode,
			"Max Avg Retun So far":  self.max_return, 
			"Max Avg Return Epoch": self.max_return_epoch,
		})  



import argparse
import os
if __name__=="__main__":
	config = Config()
	wandb.init(
		project=config.save_dir.wandb_project,   
		name= config.save_dir.wandb_dir,  
		config=Config()  
    )  
	ppo = PPO()

	nn_dir =config.save_dir.nn_dir
	if not os.path.exists(nn_dir):
	    os.makedirs(nn_dir)
  
	if config.save_dir.checkpoints != 'None':
		ppo.LoadModel(config.save_dir.checkpoints)
	else:
		ppo.SaveModel()

	for i in range(ppo.max_iteration-5):
		ppo.Train()
		ppo.Evaluate()
		#Plot(rewards,'reward',0,False)
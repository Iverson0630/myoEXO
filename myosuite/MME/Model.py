import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math
import time
from collections import OrderedDict
import numpy as np
from IPython import embed
import action_filter  

MultiVariateNormal = torch.distributions.Normal
temp = MultiVariateNormal.log_prob
MultiVariateNormal.log_prob = lambda self, val: temp(self,val).sum(-1, keepdim=True)

temp2 = MultiVariateNormal.entropy
MultiVariateNormal.entropy = lambda self: temp2(self).sum(-1)  
MultiVariateNormal.mode = lambda self: self.mean  

use_cuda = torch.cuda.is_available()  
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor    

def weights_init(m):  
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		torch.nn.init.xavier_uniform_(m.weight)
		m.bias.data.zero_()   

class MuscleNN(nn.Module):
	def __init__(self,num_total_muscle_related_dofs,num_dofs,num_muscles):
		super(MuscleNN,self).__init__()
		self.num_total_muscle_related_dofs = num_total_muscle_related_dofs
		self.num_dofs = num_dofs
		self.num_muscles = num_muscles

		num_h1 = 1024
		num_h2 = 512
		num_h3 = 512
		self.fc = nn.Sequential(
			nn.Linear(num_total_muscle_related_dofs+num_dofs,num_h1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h1,num_h2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h2,num_h3),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h3,num_muscles),
			nn.Tanh(),
			nn.ReLU()		
		)
		self.std_muscle_tau = torch.zeros(self.num_total_muscle_related_dofs)
		self.std_tau = torch.zeros(self.num_dofs)

		for i in range(self.num_total_muscle_related_dofs):
			self.std_muscle_tau[i] = 200.0

		for i in range(self.num_dofs):
			self.std_tau[i] = 200.0
		if use_cuda:
			self.std_tau = self.std_tau.cuda()
			self.std_muscle_tau = self.std_muscle_tau.cuda()
			self.cuda()
		self.fc.apply(weights_init)
	
	def forward(self,muscle_tau,tau): 
		muscle_tau = muscle_tau/self.std_muscle_tau

		tau = tau/self.std_tau
		out = self.fc.forward(torch.cat([muscle_tau,tau],dim=1))
		return out		

	def load(self,path):
		print('load muscle nn {}'.format(path))
		self.load_state_dict(torch.load(path))

	def save(self,path):
		print('save muscle nn {}'.format(path))
		torch.save(self.state_dict(),path)
		
	def get_activation(self,muscle_tau,tau):
		act = self.forward(Tensor(muscle_tau.reshape(1,-1).astype(np.float32)),Tensor(tau.reshape(1,-1).astype(np.float32)))
		return act.cpu().detach().numpy().squeeze()
		
class SimulationHumanNN(nn.Module):
	def __init__(self,num_states,num_actions):
		super(SimulationHumanNN,self).__init__()
		
		num_h1 = 512
		num_h2 = 512

		self.p_fc1 = nn.Linear(num_states,num_h1)
		self.p_fc2 = nn.Linear(num_h1,num_h2)
		self.p_fc3 = nn.Linear(num_h2,num_actions)
		self.log_std = nn.Parameter(torch.zeros(num_actions)) # exoloration noise

		self.v_fc1 = nn.Linear(num_states,num_h1)
		self.v_fc2 = nn.Linear(num_h1,num_h2)
		self.v_fc3 = nn.Linear(num_h2,1)

		torch.nn.init.xavier_uniform_(self.p_fc1.weight)
		torch.nn.init.xavier_uniform_(self.p_fc2.weight)
		torch.nn.init.xavier_uniform_(self.p_fc3.weight)

		self.p_fc1.bias.data.zero_()
		self.p_fc2.bias.data.zero_()
		self.p_fc3.bias.data.zero_()

		torch.nn.init.xavier_uniform_(self.v_fc1.weight)
		torch.nn.init.xavier_uniform_(self.v_fc2.weight)
		torch.nn.init.xavier_uniform_(self.v_fc3.weight)

		self.v_fc1.bias.data.zero_()
		self.v_fc2.bias.data.zero_()
		self.v_fc3.bias.data.zero_()

	def forward(self,x):
		p_out = F.relu(self.p_fc1(x))
		p_out = F.relu(self.p_fc2(p_out))
		p_out = self.p_fc3(p_out)
		#mu = torch.sigmoid(p_out)  # 保证在 [0,1]
		#self.log_std_ = torch.clamp(self.log_std, min=-5, max=-1)
		p_out = MultiVariateNormal(p_out,self.log_std.exp())

		v_out = F.relu(self.v_fc1(x))
		v_out = F.relu(self.v_fc2(v_out))
		v_out = self.v_fc3(v_out)
		return p_out,v_out

	def load(self,path):
		print('load simulation nn {}'.format(path))
		self.load_state_dict(torch.load(path))

	def save(self,path):
		print('save simulation nn {}'.format(path))
		torch.save(self.state_dict(),path)
		
	def get_action(self,s):
		ts = torch.tensor(s.astype(np.float32))
		p,_ = self.forward(ts)
		return p.loc.cpu().detach().numpy().squeeze()

	def get_random_action(self,s):
		ts = torch.tensor(s.astype(np.float32))
		p,_ = self.forward(ts)
		return p.sample().cpu().detach().numpy().squeeze()

class SimulationExoNN(nn.Module):  
	def __init__(self, num_states, num_actions):  
		super(SimulationExoNN, self).__init__() 

		num_h1 = 128   
		num_h2 = 64    
		self.num_actions = num_actions  
		self.p_fc1 = nn.Linear(num_states, num_h1)
		self.p_fc2 = nn.Linear(num_h1, num_h2)
		self.p_fc3 = nn.Linear(num_h2, num_actions)    # for LSTM network, originally was num_h2
		self.log_std = nn.Parameter(torch.zeros(num_actions))     

		self.v_fc1 = nn.Linear(num_states, num_h1)
		self.v_fc2 = nn.Linear(num_h1, num_h2)
		self.v_fc3 = nn.Linear(num_h2, 1) # for LSTM network, originally was num_h2

		torch.nn.init.xavier_uniform_(self.p_fc1.weight)
		torch.nn.init.xavier_uniform_(self.p_fc2.weight)
		torch.nn.init.xavier_uniform_(self.p_fc3.weight)

		self.p_fc1.bias.data.zero_()
		self.p_fc2.bias.data.zero_()
		self.p_fc3.bias.data.zero_() 

		torch.nn.init.xavier_uniform_(self.v_fc1.weight)
		torch.nn.init.xavier_uniform_(self.v_fc2.weight)
		torch.nn.init.xavier_uniform_(self.v_fc3.weight)

		self.v_fc1.bias.data.zero_()
		self.v_fc2.bias.data.zero_()
		self.v_fc3.bias.data.zero_()

		self._action_filter = self._BuildActionFilter()  

	def _BuildActionFilter(self):
		sampling_rate = 30
		num_joints = self.num_actions
		a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate, num_joints=num_joints, 
													filter_low_cut = 0, filter_high_cut = 8)
		return a_filter

	def _FilterAction(self, action):
		if sum(self._action_filter.xhist[0])[0] == 0:
			self._action_filter.init_history(action)

		return self._action_filter.filter(action)

	def forward(self, x):
		p_out = F.relu(self.p_fc1(x))
		p_out = F.relu(self.p_fc2(p_out))
		p_out = self.p_fc3(p_out)

		p_out = MultiVariateNormal(p_out, self.log_std.exp())   
		
		if np.any(np.isnan(self.v_fc1.weight.cpu().detach().numpy())):
			print("here")
		v_out = F.relu(self.v_fc1(x))
		v_out = F.relu(self.v_fc2(v_out))
		v_out = self.v_fc3(v_out)

		return p_out, v_out  

	def load(self, path):
		print('load simulation nn {}'.format(path))
		self.load_state_dict(torch.load(path))

	def save(self, path):
		print('save simulation nn {}'.format(path))
		torch.save(self.state_dict(), path)

	def get_action(self, s):  
		ts = torch.tensor(s.astype(np.float32))
		p, _ = self.forward(ts)
		p_ = self._FilterAction(p.loc.cpu().detach().numpy())
		return p_.astype(np.float32)   

	def get_random_action(self, s):
		ts = torch.tensor(s.astype(np.float32))   
		p, _ = self.forward(ts)
		return p.sample().cpu().detach().numpy()      

	# def get_action(self,s):
	# 	ts = torch.tensor(s.astype(np.float32))
	# 	p,_ = self.forward(ts)
	# 	return p.loc.cpu().detach().numpy().squeeze()

	# def get_random_action(self,s):
	# 	ts = torch.tensor(s.astype(np.float32))
	# 	p,_ = self.forward(ts)
	# 	return p.sample().cpu().detach().numpy().squeeze()   

class SimulationExoLSTMNN(nn.Module):  
	def __init__(self, num_states, num_actions):   
		super(SimulationExoLSTMNN, self).__init__()   

		num_h1 = 256
		num_lstm_layers = 2   
		
		self.num_actions = num_actions
		
		self.p_lstm1 = nn.LSTM(num_states, num_h1, num_lstm_layers, batch_first=True) # for LSTM network
		self.p_fc3 = nn.Linear(num_h1, num_actions) # for LSTM network, originally was num_h2
		self.log_std = nn.Parameter(torch.zeros(num_actions))

		self.v_lstm1 = nn.LSTM(num_states, num_h1, num_lstm_layers, batch_first=True) # for LSTM network
		self.v_fc3 = nn.Linear(num_h1, 1) # for LSTM network, originally was num_h2

		for name, param in self.p_lstm1.named_parameters():  # for LSTM network
			if 'weight' in name:
				torch.nn.init.xavier_uniform_(param)  
		torch.nn.init.xavier_uniform_(self.p_fc3.weight)

		self.p_fc3.bias.data.zero_()

		for name, param in self.v_lstm1.named_parameters():  # for LSTM network
			if 'weight' in name:
				torch.nn.init.xavier_uniform_(param)
		torch.nn.init.xavier_uniform_(self.v_fc3.weight)

		self.v_fc3.bias.data.zero_()  

	def forward(self, x):   
		# actor network
		p_out, _ = self.p_lstm1(x)
		if p_out.dim() == 2:
			p_out = p_out.unsqueeze(1)
		p_out = p_out[:, -1, :]   
		p_out = F.relu(p_out)   
		p_out = self.p_fc3(p_out)     
		
		p_out = MultiVariateNormal(p_out, self.log_std.exp())   
		
		# critic network
		v_out, _ = self.v_lstm1(x)
		if v_out.dim() == 2:
			v_out = v_out.unsqueeze(1)
		v_out = v_out[:, -1, :]   
		v_out = F.relu(v_out)
		v_out = self.v_fc3(v_out)   

		return p_out, v_out

	def load(self, path):
		print('load simulation nn {}'.format(path))
		self.load_state_dict(torch.load(path))

	def save(self, path):
		print('save simulation nn {}'.format(path))  
		torch.save(self.state_dict(), path) 

	def get_action(self, s): 
		
		s = np.expand_dims(s, axis=0)   
		ts = torch.tensor(s.astype(np.float32)) 
		# ts.to(device)  

		p,_ = self.forward(ts)  
		#print('actor output', p.loc.cpu().detach().numpy().squeeze()  )
		return p.loc.cpu().detach().numpy().squeeze()  

	def get_random_action(self, s):
		ts = torch.tensor(s)  
		p, _ = self.forward(ts)   
		return p.sample().cpu().detach().numpy().squeeze() 

class SimulationExoTorqueCTLNN(nn.Module):  
	def __init__(self, num_states, num_actions):
		super(SimulationExoTorqueCTLNN, self).__init__()

		num_h1 = 256
		num_lstm_layers = 2

		self.num_actions = num_actions
		torque_limit = 15  

		# ==== Actor ====
		self.p_lstm1 = nn.LSTM(num_states, num_h1, num_lstm_layers, batch_first=True)
		self.p_fc3 = nn.Linear(num_h1, num_actions)
		self.log_std = nn.Parameter(torch.zeros(num_actions))

		# ==== Critic ====
		self.v_lstm1 = nn.LSTM(num_states, num_h1, num_lstm_layers, batch_first=True)
		self.v_fc3 = nn.Linear(num_h1, 1)

		# ==== 初始化 ====
		for name, param in self.p_lstm1.named_parameters():
			if 'weight' in name:
				nn.init.xavier_uniform_(param, gain=1.2)
		nn.init.xavier_uniform_(self.p_fc3.weight, gain=3.0)   # 放大输出层权重
		self.p_fc3.bias.data.uniform_(-0.5, 0.5)

		for name, param in self.v_lstm1.named_parameters():
			if 'weight' in name:
				nn.init.xavier_uniform_(param)
		nn.init.xavier_uniform_(self.v_fc3.weight)
		self.v_fc3.bias.data.zero_()

		self.torque_limit = torque_limit


	def forward(self, x):
		# Actor
		p_out, _ = self.p_lstm1(x)
		if p_out.dim() == 2:
			p_out = p_out.unsqueeze(1)
		p_out = p_out[:, -1, :]
		p_out = F.relu(p_out)
		p_out = self.p_fc3(p_out)
		# 缩放到物理范围
		mu = torch.tanh(p_out) * self.torque_limit
		p_out = MultiVariateNormal(mu, self.log_std.exp())
	
		# Critic
		v_out, _ = self.v_lstm1(x)
		if v_out.dim() == 2:
			v_out = v_out.unsqueeze(1)
		v_out = v_out[:, -1, :]
		v_out = F.relu(v_out)
		v_out = self.v_fc3(v_out)

		return p_out, v_out



	def load(self, path):
		print('load simulation torque control nn {}'.format(path))
		self.load_state_dict(torch.load(path))

	def save(self, path):
		print('save simulation torque control nn {}'.format(path))  
		torch.save(self.state_dict(), path) 

	def get_action(self, s): 
		
		s = np.expand_dims(s, axis=0)   
		ts = torch.tensor(s.astype(np.float32)) 
		# ts.to(device)  

		p,_ = self.forward(ts)  
		#print('actor output', p.loc.cpu().detach().numpy().squeeze()  )
		return p.loc.cpu().detach().numpy().squeeze()  

	def get_random_action(self, s):
		ts = torch.tensor(s)  
		p, _ = self.forward(ts)   
		return p.sample().cpu().detach().numpy().squeeze() 

import argparse
import os
if __name__=="__main__":  
	parser = argparse.ArgumentParser()  

	parser.add_argument('-lp', '--load_path', default=None, help='load model path')    
	parser.add_argument('-sp', '--save_path', default=None, help='save model path')    
	
	parser.add_argument('-n','--name',help='model name')    
	parser.add_argument('-d','--meta',help='meta file')     
	parser.add_argument('-a','--algorithm',help='mass nature tmech')     
	parser.add_argument('-t','--type',help='wm: with muscle, wo: without muscle')   
	parser.add_argument('-f','--flag',default='',help='recognize the main features')       

	parser.add_argument('-wp', '--wandb_project', default='junxi_training', help='wandb project name')
	parser.add_argument('-we', '--wandb_entity', default='markzhumi1805', help='wandb entity name')
	parser.add_argument('-wn', '--wandb_name', default='Test', help='wandb run name')
	parser.add_argument('-ws', '--wandb_notes', default='', help='wandb notes')   
 
	parser.add_argument('--max_iteration',type=int, default=5000, help='meta file')    
 
	args =parser.parse_args()    
 
	nn_human_network = SimulationHumanNN(136, 50)  
 
	# load human model network
	nn_human_network.load('../trained_policy/nn_tmech_wo_100_600/max_human.pt')   
 
 
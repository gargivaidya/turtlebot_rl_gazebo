#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import argparse
import gym
import os
import sys
import pickle
import time
import datetime
from collections import deque
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import itertools
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, Image


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc_coorl import DiscretePolicy
from models.mlp_discriminator import Discriminator
from core.ppo import ppo_step
from core.trpo import trpo_step
from core.common import estimate_advantages
from core.agent_coorl import Agent
from tb3env_sparse import DiscreteTurtleGym


parser = argparse.ArgumentParser(description='PyTorch COORL example')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
					help='damping (default: 1e-2)')
parser.add_argument('--render', action='store_true', default=False,
					help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
					help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
					help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
					help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
					help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
					help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
					help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
					help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
					help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
					help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
					help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=1000, metavar='N',
					help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
					help='interval between training status logs (default: 10)')
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
					help='max kl value (default: 1e-2)')



parser.add_argument('--sparse', action='store_true', default=False,
					help='Setting it to sparse env')
parser.add_argument('--ppo', action='store_true', default=False,
					help='PPO')
parser.add_argument('--final', action='store_true', default=False,
					help='Final')
parser.add_argument('--adaptive', action='store_true', default=False,
					help='Adaptive KL')
parser.add_argument('--decay-constant', type=float, default= -1., metavar='G',
					help='Decay Constant')
parser.add_argument('--adaptive-2', type=int, default=-1, metavar='N',
					help='warmup samples befrore decay')
parser.add_argument('--observe', type=int, default=0, metavar='N',
					help='observe first n states')
parser.add_argument('--env-num', type=int, default=1, metavar='N',
					help='Env number')
parser.add_argument('--exp-traj', type=int, default=1, metavar='N',
					help='exp traj number')
parser.add_argument('--window', type=int, default=10, metavar='N',
					help='observation window')
parser.add_argument('--delay-val', type=int, default=-1, metavar='N',
					help='reward freq')
parser.add_argument('--inc', type=float, default=1.01, metavar='G',
					help='KL decrease')
parser.add_argument('--dec', type=float, default=0.9, metavar='G',
					help='KL increase')
parser.add_argument('--expert-limit', type=int, default=0, metavar='N',
					help='expert examples')
parser.add_argument('--high-kl', type=float, default=3e-2, metavar='G',
					help='max kl value (default: 1e-2)')
parser.add_argument('--low-kl', type=float, default=5e-4, metavar='G',
					help='max kl value (default: 1e-2)')
parser.add_argument('--nn-param', nargs='+', type=int,default=[128,128])
parser.add_argument('--save-model', action='store_true', default=False,
					help='saves model')
parser.add_argument('--save-reward', type=float, default=-150, metavar='G',
					help='Save reward if log_eval is above this value')
parser.add_argument('--episode-length', type=int, default=1000, metavar='N',
					help='Episode Length')
args = parser.parse_args()






nn_size = tuple(args.nn_param)
print('NN Size',nn_size)


if args.exp_traj == 1:
	args.model_path = 'learned_models/DiscreteDubinGymDense_PDM.p'




def train(env):
	args.env_name = 'GzboDiscTurtle'
	if args.final is False:
		writer = SummaryWriter('Results/runs/Env_{}/COORL/COORL_{}'
			.format(args.env_name,datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
	else:
		writer = SummaryWriter('Results/final_run/Env_{}/COORL/COORL_{}'
			.format(args.env_name,datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))



	writer.add_text('Env name',str(args.env_name))
	writer.add_text('Data type','Un-Filtered')
	writer.add_text('NN Size', str(nn_size))
	writer.add_text('Min bath size', str(args.min_batch_size))
	writer.add_text('Eval bath size', str(args.eval_batch_size))

	if args.ppo:
		writer.add_text('Algo','PPO')
	else:
		writer.add_text('Algo','TRPO')
	if args.adaptive:
		writer.add_text('KL decay','Adaptive')
		writer.add_text('KL decay window',str(args.window))
		writer.add_text('KL decay inc',str(args.inc))
		writer.add_text('KL decay dec',str(args.dec))
		print('Adaptive')

	elif args.adaptive_2 > 0:
		writer.add_text('KL decay','Adaptive_2')
		writer.add_text('KL decay window',str(args.window))
		writer.add_text('KL decay warmup',str(args.adaptive_2))
		writer.add_text('KL decay dec',str(args.dec))
		writer.add_text('KL High',str(args.high_kl))
		writer.add_text('KL Low',str(args.low_kl))
		print('Adaptive 2')

	elif args.decay_constant > 0.:
		writer.add_text('KL decay','decay_constant')
		writer.add_text('KL decay val',str(args.decay_constant))
		print('Constant')


	##########################################################################

	dtype = torch.float64
	torch.set_default_dtype(dtype)
	device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
	if torch.cuda.is_available():
		torch.cuda.set_device(args.gpu_index)
		print(device)




	if args.observe == 0:
		args.observe = env.observation_space.shape[0]
	writer.add_text('Observable state',str(args.observe))



	state_dim = env.observation_space.shape[0]
	is_disc_action = len(env.action_space.shape) == 0
	action_dim = 1 if is_disc_action else env.action_space.shape[0]


	"""seeding"""

	writer.add_text('Seed',str(args.seed))
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	env.seed(args.seed)

	"""define actor and critic"""
	if is_disc_action:
		policy_net = DiscretePolicy(state_dim, env.action_space.n)
	else:
		policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std,hidden_size=nn_size)

	value_net = Value(state_dim,hidden_size=nn_size)
	value_net_exp = Value(state_dim,hidden_size=nn_size)

	# load expert model
	expert_policy_net, _ = pickle.load(open(args.model_path, "rb"))
	to_device(device, policy_net, value_net, value_net_exp,expert_policy_net)

	if args.ppo:
		optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
		optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
		# optimization epoch number and batch size for PPO
		optim_epochs = 10
		optim_batch_size = 64





	def expert_reward(state, action):
		partial_states = tensor(state[:,:args.observe],dtype=dtype).to(device)
		full_states = tensor(state,dtype=dtype).to(device)
		action = tensor(action,dtype=dtype).to(device)
		with torch.no_grad():
			expert_log_probs = expert_policy_net.get_log_prob_1(partial_states, action)
			policy_log_probs = policy_net.get_log_prob(full_states, action)
			return expert_log_probs - policy_log_probs


	"""create agent"""
	agent = Agent(env, policy_net, device,
				  num_threads=args.num_threads)


	def update_params(batch, i_iter,kl):	
		states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
		actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
		rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
		masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
		rewards_exp = expert_reward(np.stack(batch.state),np.stack(batch.action))
		with torch.no_grad():
			values = value_net(states)
			values_exp = value_net_exp(states)		
			fixed_log_probs = policy_net.get_log_prob(states, actions)


		"""get advantage estimation from the trajectories"""
		advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

		advantages_exp, returns_exp = estimate_advantages(rewards_exp, masks, values_exp, args.gamma, args.tau, device)

		if args.ppo:
			"""perform mini-batch PPO update"""
			optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
			for _ in range(optim_epochs):
				perm = np.arange(states.shape[0])
				np.random.shuffle(perm)
				perm = LongTensor(perm).to(device)

				states, actions, returns, advantages, fixed_log_probs = \
					states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

				for i in range(optim_iter_num):
					ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
					states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
						states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

					ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
							 advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg) 		

		else:
			trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl, args.damping, args.l2_reg)

		if kl > 5.1e-7:
			trpo_step(policy_net, value_net_exp, states, actions, returns_exp, advantages_exp, kl, args.damping, args.l2_reg,fixed_log_probs = fixed_log_probs)


	def main_loop():
		kl = args.max_kl
		prev_rwd = deque(maxlen = args.window)
		prev_rwd.append(0)

		if args.adaptive_2 > 0:
			kl = args.high_kl
		if args.decay_constant > 0:
			kl = args.decay_constant

		for i_iter in range(args.max_iter_num):
			"""generate multiple trajectories that reach the minimum batch_size"""
			t1 = time.time()
			batch, log = agent.collect_samples(args.min_batch_size)

			if args.adaptive:			
				if i_iter > args.window:
					avg_prev_rwd = np.mean(prev_rwd)
					if (avg_prev_rwd >= log['avg_reward']):
						kl = min(args.high_kl,kl*args.inc) 
					else:
						# kl = min(args.low_kl,kl*args.dec)
						kl = max(args.low_kl,kl*args.dec)

			elif (args.adaptive_2 > 0):
				if (i_iter > args.adaptive_2):
					avg_prev_rwd = np.mean(prev_rwd)						
					if ((avg_prev_rwd < log['avg_reward']) or (avg_prev_rwd > 0.9)):					
						kl = max(args.low_kl,kl*args.dec)


			writer.add_scalar('KL',kl,i_iter)
			prev_rwd.append(log['avg_reward'])
			t0 = time.time()
			update_params(batch, i_iter,kl)
			

			t2 = time.time()

			if i_iter % args.log_interval == 0:

				print('{}\tT_sample {:.4f}\tT_update {:.4f}\ttrain_R_avg {:.2f} \t eval_R_avg {:.2f} \t KL{:}'
					.format(i_iter, log['sample_time'], t1-t0,log['avg_reward'],log['true_reward'],kl))

			writer.add_scalar('rewards/train_R_avg',log['avg_reward'],i_iter+1)
			

			if (log['true_reward'] > args.save_reward and args.save_model):
				log_name = str(log['true_reward']) 
				to_device(torch.device('cpu'), policy_net, value_net)
				pickle.dump((policy_net, value_net),
							open('learned_models/{}_{}_coorl.p'.format(args.env_name,log_name), 'wb'))
				to_device(device, policy_net, value_net)
				print("Done!!!")

			"""clean up gpu memory"""
			torch.cuda.empty_cache()


	main_loop()



if __name__ == '__main__':
	try:
		rospy.init_node('train', anonymous=True)		
		env = DiscreteTurtleGym()
		train(env)
		rospy.spin()
	except rospy.ROSInterruptException:
		pass

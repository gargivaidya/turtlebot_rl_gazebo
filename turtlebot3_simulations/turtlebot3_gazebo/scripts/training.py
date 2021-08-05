#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import itertools
import argparse
import datetime
import torch
import sys
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, Image
import numpy as np
from tb3env import ContinuousTurtleGym, Discrete4TurtleGym, Discrete15TurtleGym
from std_srvs.srv import Empty

sys.path.append('./algorithm/SAC/')
from sac import SAC
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Turtlebot Env & Soft Actor-Critic Args')
parser.add_argument('--env-name', default="ContinuousTurtleGym",
					help='Turtle Gym environment (default: ContinuousTurtleGym)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=200000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', type = int, default = 0, metavar = 'N',
                    help='run on CUDA (default: False)')
parser.add_argument('--max_episode_length', type=int, default=400, metavar='N',
					help='max episode length (default: 3000)')
args = parser.parse_args()

def train(env):

	agent = SAC(env.observation_space.shape[0], env.action_space, args)
	memory = ReplayMemory(args.replay_size, args.seed)
	writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'TurtleGym',
														 args.policy, "autotune" if args.automatic_entropy_tuning else ""))

	total_numsteps = 0
	updates = 0
	num_goal_reached = 0

	for i_episode in itertools.count(1):
		episode_reward = 0
		episode_steps = 0
		done = False
		state = env.reset()
		
		while not done:
			start_time = time.time()
			if args.start_steps > total_numsteps:
				action = env.action_space.sample()  
			else:
				action = agent.select_action(state)  

			next_state, reward, done, _ = env.step(action) 
			if (reward > 9) and (episode_steps > 1): 
				num_goal_reached += 1 

			episode_steps += 1
			total_numsteps += 1
			episode_reward += reward
			if episode_steps > args.max_episode_length:
				done = True

			# Ignore the "done" signal if it comes from hitting the time horizon.
			# (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
			mask = 1 if episode_steps == args.max_episode_length else float(not done)
			# mask = float(not done)
			memory.push(state, action, reward, next_state, mask) # Append transition to memory

			state = next_state

		# if i_episode % UPDATE_EVERY == 0: 
		if len(memory) > args.batch_size:
			# Number of updates per step in environment
			for i in range(args.updates_per_step*args.max_episode_length):
				# Update parameters of all the networks
				critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

				writer.add_scalar('loss/critic_1', critic_1_loss, updates)
				writer.add_scalar('loss/critic_2', critic_2_loss, updates)
				writer.add_scalar('loss/policy', policy_loss, updates)
				writer.add_scalar('loss/entropy_loss', ent_loss, updates)
				writer.add_scalar('entropy_temprature/alpha', alpha, updates)
				updates += 1

		if total_numsteps > args.num_steps:
			break

		if (episode_steps > 1):
			writer.add_scalar('reward/train', episode_reward, i_episode)
			writer.add_scalar('reward/episode_length',episode_steps, i_episode)
			writer.add_scalar('reward/num_goal_reached',num_goal_reached, i_episode)

		print("Episode: {} \t total numsteps: {} \t episode steps: {} \t reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
		print("Number of Goals Reached: ",num_goal_reached)

	print('----------------------Training Ending----------------------')

	agent.save_model("burger", suffix = "1")
	return True


if __name__ == '__main__':
	try:
		rospy.init_node('train', anonymous=True)
		if args.env_name == "ContinuousTurtleGym":
			env =  ContinuousTurtleGym()
		elif args.env_name == "Discrete4TurtleGym" :
			env = Discrete4TurtleGym()
		else :
			env = Discrete15TurtleGym()
		train(env)
		rospy.spin()
	except rospy.ROSInterruptException:
		pass
		

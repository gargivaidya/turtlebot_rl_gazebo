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
from tb3env import ContinuousTurtleGym, DiscreteTurtleGym, ContinuousTurtleObsGym, DiscreteTurtleObsGym
from std_srvs.srv import Empty
# from stable_baselines.sac.policies import MlpPolicy
# from stable_baselines import SAC, DQN, PPO2
# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.deepq.policies import FeedForwardPolicy, register_policy
# from stable_baselines.sac.policies import FeedForwardPolicy, register_policy
from stable_baselines3 import PPO

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="DiscreteTurtleGym",
					help='Turtlebot Gazebo Gym environment (default: DiscreteTurtleGym)')
parser.add_argument('--n-actions', type=int, default=15, metavar='N',
					help='number of discrete actions 5 or 15 (default: 4)')
args = parser.parse_args()

# class CustomDQNPolicy(FeedForwardPolicy):
# 		def __init__(self, *args, **kwargs):
# 			super(CustomDQNPolicy, self).__init__(*args, **kwargs, layers=[128, 128, 128], layer_norm=True, feature_extraction="mlp")

# register_policy('CustomDQNPolicy', CustomDQNPolicy)

# class CustomSACPolicy(FeedForwardPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomSACPolicy, self).__init__(*args, **kwargs,
#                                            layers=[128, 128, 128],
#                                            layer_norm=False,
#                                            feature_extraction="mlp")

def train(env) :
	model = PPO("MlpPolicy", env, verbose = 0, tensorboard_log="./ppo_turtle/")
	model.learn(total_timesteps = 300000)
	model.save("./sbmodels/ppo_unit_distance_3box_singletarget")

	# model = SAC(MlpPolicy, env, learning_rate=1e-3, buffer_size=50000, 
	# 	learning_starts=100, train_freq=1, batch_size=64, tau=0.005, 
	# 	ent_coef='auto', verbose=1, tensorboard_log="./sac_turtle/")
	# model.learn(total_timesteps=50000, log_interval=10)
	# model.save("./sbmodels/sac_turtle_1")    

	# model = DQN('CustomDQNPolicy', env, learning_rate=1e-3, 
	# 		buffer_size = 200000, exploration_fraction = 0.1, 
	# 		exploration_final_eps = 0.05, exploration_initial_eps = 1.0,  
	# 		prioritized_replay=True, prioritized_replay_alpha = 0.3, 
	# 		prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
 #            prioritized_replay_eps=1e-6, verbose=1, 
	# 		tensorboard_log="./dqn_turtle_obs/")
	# model.learn(total_timesteps=200000, log_interval=10)
	# model.save("./sbmodels/dqn_turtle_obs_5")

if __name__ == '__main__':
	try:
		rospy.init_node('sbtrain', anonymous=True)
		if args.env_name == "ContinuousTurtleGym":
			env =  ContinuousTurtleGym()
		elif args.env_name == "DiscreteTurtleGym" :
			env = DiscreteTurtleGym(args.n_actions)
		elif args.env_name == "ContinuousTurtleObsGym" :
			env =  ContinuousTurtleObsGym()
		elif args.env_name == "DiscreteTurtleObsGym" :
			env = DiscreteTurtleObsGym(args.n_actions)
		train(env)
		rospy.spin()
	except rospy.ROSInterruptException:
		pass


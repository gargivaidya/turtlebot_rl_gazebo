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
from tb3env import ContinuousTurtleGym, DiscreteTurtleGym
from std_srvs.srv import Empty
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.deepq.policies import FeedForwardPolicy, register_policy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Discrete4TurtleGym",
					help='Turtlebot Gazebo Gym environment (default: Discrete4TurtleGym)')
parser.add_argument('--n-actions', type=int, default=4, metavar='N',
					help='number of discrete actions 4 or 15 (default: 4)')
args = parser.parse_args()

class CustomDQNPolicy(FeedForwardPolicy):
		def __init__(self, *args, **kwargs):
			super(CustomDQNPolicy, self).__init__(*args, **kwargs, layers=[128, 128, 128], layer_norm=True, feature_extraction="mlp")

register_policy('CustomDQNPolicy', CustomDQNPolicy)

def train(env) :

	# model = SAC(MlpPolicy, env, verbose=1, tensorboard_log="./dqn_turtle/")
	# model.learn(total_timesteps=50000, log_interval=10)
	# model.save("sac_turtle")    

	model = DQN('CustomDQNPolicy', env, learning_rate=1e-3, 
			buffer_size = 50000, exploration_fraction = 0.1, 
			exploration_final_eps = 0.05, exploration_initial_eps = 1.0,  
			prioritized_replay=True, prioritized_replay_alpha = 0.3, 
			prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
            prioritized_replay_eps=1e-6, verbose=1, 
			tensorboard_log="./dqn_turtle/")
	model.learn(total_timesteps=50000, log_interval=10)
	model.save("dqn_turtle_1")

if __name__ == '__main__':
	try:
		rospy.init_node('sbtrain', anonymous=True)
		if args.env_name == "ContinuousTurtleGym":
			env =  ContinuousTurtleGym()
		elif args.env_name == "DiscreteTurtleGym" :
			env = DiscreteTurtleGym(args.n_actions)
		train(env)
		rospy.spin()
	except rospy.ROSInterruptException:
		pass


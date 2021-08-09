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
from tb3envbot import ContinuousTurtleGym, DiscreteTurtleGym, ContinuousTurtleObsGym, DiscreteTurtleObsGym
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
parser.add_argument('--env-name', default="DiscreteTurtleGym",
          help='Turtlebot Gazebo Gym environment (default: DiscreteTurtleGym)')
parser.add_argument('--n-actions', type=int, default=5, metavar='N',
          help='number of discrete actions 4 or 15 (default: 5)')
args = parser.parse_args()

def evaluate(env) :
  model = DQN.load("./sbmodels/dqn_turtle_obs_1")
  episode_rewards = []
  for _ in range(10) :
    done = False
    obs = env.reset()
    ep_reward = 0
    while not done:
      action, _ = model.predict(obs)
      obs, reward, done, _ = env.step(action)
      # rospy.sleep(0.05)
      ep_reward += reward
      if done :
        episode_rewards.append(ep_reward)
        continue
  mean_10ep_reward = round(np.mean(episode_rewards[-10:]), 1)
  print("Mean reward:", mean_10ep_reward, "Num episodes:", len(episode_rewards))

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
    evaluate(env)
    rospy.spin()
  except rospy.ROSInterruptException:
    pass
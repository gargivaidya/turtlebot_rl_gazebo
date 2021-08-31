#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import argparse
import gym
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tb3env import ContinuousTurtleGym, DiscreteTurtleGym, ContinuousTurtleObsGym, DiscreteTurtleObsGym
from itertools import count
from utils import *


parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-expert-state-num', type=int, default=2000, metavar='N',
                    help='maximal number of main iterations (default: 2000)')
parser.add_argument('--disc', action='store_true', default=False,
                    help='disc env')
parser.add_argument('--episode-length', type=int, default=1000, metavar='N',
                    help='Episode Length')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
args.model_path = 'learned_models/DiscreteDubinGymDense_Best.p'


def eval_pol(env):
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    is_disc_action = len(env.action_space.shape) == 0
    state_dim = env.observation_space.shape[0]

    policy_net, _ = pickle.load(open(args.model_path, "rb"))

    def main_loop():

        num_steps = 0
        avg_rwd = []

        for i_episode in count():

            state = env.reset()
            reward_episode = 0
            done = False
            while (not done):            
                state_var = tensor(state).unsqueeze(0).to(dtype)
                # choose mean action
                if is_disc_action:
                    # action = policy_net.select_action(state_var)[0].cpu().numpy()
                    action_prob = policy_net(state_var)[0].detach().numpy()
                    action = np.argmax(action_prob)
                else:
                    action = policy_net(state_var)[0][0].detach().numpy()
                    action = [max(0,min(0.22,action[0])),max(-2.84,min(2.84,action[1]))]
                
                action = int(action) if is_disc_action else action.astype(np.float64)
                next_state, reward, done, _ = env.step(action)
                
                reward_episode += reward
                num_steps += 1

                if done:
                    break

                state = next_state

            print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))
            avg_rwd.append(reward_episode)

            if num_steps >= args.max_expert_state_num:                
                print('Avg_rwd {:.2f}\t std_rwd: {:.2f}'.format(np.mean(avg_rwd), np.std(avg_rwd)))
                print('Max_rwd {:.2f}\t Min_rwd: {:.2f}'.format(np.max(avg_rwd), np.min(avg_rwd)))
                break

    main_loop()


if __name__ == '__main__':
    try:
        rospy.init_node('sbtrain', anonymous=True)
        env = DiscreteTurtleGym(15)
        eval_pol(env)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


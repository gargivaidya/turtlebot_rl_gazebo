#!/usr/bin/env python3
import itertools
import argparse
import rospy
from tb3env import ContinuousTurtleGym, Discrete4TurtleGym, Discrete15TurtleGym
from stable_baselines.common.env_checker import check_env

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="ContinuousTurtleGym",
					help='Dubin Gym environment (default: ContinuousDubinGym)')
args = parser.parse_args()

def test():
	
	if args.env_name == "ContinuousTurtleGym":
		env =  ContinuousTurtleGym()
	elif args.env_name == "Discrete4TurtleGym" :
		env = Discrete4TurtleGym()
	else :
		env = Discrete15TurtleGym()

	print("Issues with Custom Environment : ", check_env(env))

	print("Testing sample action...")

	# max_steps = int(1e6)
	state = env.reset()

	for i in range(239):
		if args.env_name == "ContinuousTurtleGym":
			action = [0.22, 0.]
		else:
			action = 0			
		n_state,reward,done,info = env.step(action)
		# print(i, "\t", env.pose[0], env.pose[1])
	env.reset()

if __name__ == '__main__':
	try:
		rospy.init_node('test', anonymous=True)
		env = ContinuousTurtleGym()
		test()
		rospy.spin()
	except rospy.ROSInterruptException:
		env.reset_simulation_proxy()		
		pass
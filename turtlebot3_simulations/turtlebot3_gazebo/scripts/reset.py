#!/usr/bin/env python3
import itertools
import argparse
import rospy
from std_srvs.srv import Empty

def stop() :	
	pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
	unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
	reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
	
	rospy.wait_for_service('/gazebo/reset_simulation')
	try:
		pause()
		reset_simulation_proxy()
		unpause()
		print('Simulation reset')
	except rospy.ServiceException as exc:
		print("Reset Service did not process request: " + str(exc))

if __name__ == '__main__':
	try:
		rospy.init_node('stop', anonymous=True)
		stop()
		rospy.spin()
	except rospy.ROSInterruptException:
		pass
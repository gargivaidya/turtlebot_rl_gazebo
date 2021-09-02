#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
from gym import spaces
import rospy
import time
import math
import random
import collections
from std_msgs.msg import Bool, Float32, Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, Image, LaserScan
import numpy as np
from std_srvs.srv import Empty

# Constants
MAX_STEER = 2.84
MAX_SPEED = 0.22
MIN_SPEED = 0.
THRESHOLD = 0.2
GRID = 3.
THETA0 = np.pi/4
MAX_EP_LEN = 500
OBS_THRESH = 0.15

class ContinuousTurtleGym(gym.Env):
	"""
	Continuous Action Space Gym Environment
	State - relative x, y, theta
	Action - linear vel range {-0.22 , 0.22}, 
			 angular vel range {-2.84, 2.84}
	Reward - 
	"""
	def __init__(self):
		super(ContinuousTurtleGym,self).__init__()		
		metadata = {'render.modes': ['console']}
		print("Initialising Turtlebot 3 Continuous Gym Environment...")
		self.action_space = spaces.Box(np.array([-0.22, -2.84]), np.array([0.22, 2.84]), dtype = np.float16) # max rotational velocity of burger is 2.84 rad/s
		low = np.array([-1.,-1.,-4.])
		high = np.array([1.,1.,4.])
		self.observation_space = spaces.Box(low, high, dtype=np.float16)
		self.target = [0., 0., 0.]
		self.ep_steps = 0

		self.pose = np.zeros(3) # pose_callback
		self.sector_scan = np.zeros(36) # scan_callback
		self.depth = np.zeros(1843200) # cam_callback. 
		# Depth image is compressed image of size 1280x720. data length = 1843200 (720rows x 2560step)
		
		self.action = [0., 0.] # Publisher is /cmd_vel Twist message for linear and angular velocity

		# Initialize ROS nodes
		self.sub = [0]
		self.sub[0] = rospy.Subscriber("/odom", Odometry, self.pose_callback)
		# self.sub[1] = rospy.Subscriber("scan", LaserScan, self.scan_callback)
		# self.sub[2] = rospy.Subscriber("/camera/depth/image_raw", Image, self.cam_callback)
		self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

		# Gazebo Services
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

	def pose_callback(self, pose_data) :
		# ROS Callback function for the /odom topic
		orient = pose_data.pose.pose.orientation
		q = (orient.x, orient.y, orient.z, orient.w)
		euler = self.euler_from_quaternion(q[0], q[1], q[2], q[3])
		self.pose = np.array([pose_data.pose.pose.position.x, pose_data.pose.pose.position.y, euler[2]])
		# print("Pose - ", self.pose)

	def scan_callback(self, scan_data) :
		# ROS Callback function for the /scan topic
		scan = np.array(scan_data.ranges)
		self.sector_scan = np.mean(scan.reshape(-1, 10), axis=1) # Sectorizes the lidar data to 36 sectors of 10 degrees each

	def cam_callback(self, cam_data) : 
		# ROS Callback function for the /camera/depth/image_raw topic
		self.depth = cam_data.data

	def euler_from_quaternion(self, x, y, z, w):
		'''
		Convert a quaternion into euler angles (roll, pitch, yaw)
		roll is rotation around x in radians (counterclockwise)
		pitch is rotation around y in radians (counterclockwise)
		yaw is rotation around z in radians (counterclockwise)
		'''
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll_x = math.atan2(t0, t1)

		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch_y = math.asin(t2)

		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_z = math.atan2(t3, t4)

		return roll_x, pitch_y, yaw_z # in radians

	def reset(self):
		rospy.wait_for_service('/gazebo/reset_simulation')
		try:
			self.pause()
			self.reset_simulation_proxy()
			self.unpause()
			# rospy.sleep(1)
			print('Simulation reset')
		except rospy.ServiceException as exc:
			print("Reset Service did not process request: " + str(exc))

		x = random.uniform(-1., 1.)
		y = random.choice([-1., 1.])
		self.target[0], self.target[1] = random.choice([[x, y], [y, x]])

		print("Reset target to : [{:.2f}, {:.2f}]".format(self.target[0], self.target[1]))
		head_to_target = self.get_heading(self.pose, self.target)
		yaw = random.uniform(head_to_target - THETA0, head_to_target + THETA0)

		obs = [(self.target[0] - self.pose[0])/GRID, (self.target[1] - self.pose[1])/GRID, head_to_target - self.pose[2]]
		obs = [round(x, 2) for x in obs]
		self.ep_steps = 0

		return np.array(obs)

	def get_distance(self,x1,x2):
		return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

	def get_heading(self, x1,x2):
		return math.atan2((x2[1] - x1[1]), (x2[0] - x1[0]))

	def get_reward(self):
		yaw_car = self.pose[2]
		head_to_target = self.get_heading(self.pose, self.target)

		alpha = head_to_target - yaw_car
		ld = self.get_distance(self.pose, self.target)
		crossTrackError = math.sin(alpha) * ld

		headingError = abs(alpha)
		alongTrackError = abs(self.pose[0] - self.target[0]) + abs(self.pose[1] - self.target[1])
		return -1*(abs(crossTrackError)**2 + alongTrackError + 3*headingError/1.57)/6

	def check_goal(self):
		done = False

		if (abs(self.pose[0]) < GRID) or (abs(self.pose[1]) < GRID):
			if (abs(self.pose[0] - self.target[0]) < THRESHOLD and abs(self.pose[1] - self.target[1]) < THRESHOLD) :
				done = True
				reward = 10
				print("Goal Reached")
				self.stop_bot()
			else :
				reward = self.get_reward()
		else:
			done = True
			reward = -1
			print("Outside range")
			self.stop_bot()

		if self.ep_steps > MAX_EP_LEN :
			print("Reached max episode length")
			done = True

		return done, reward

	def step(self,action):	
		reward = 0
		done = False
		info = {}
		self.ep_steps += 1

		self.action = [round(x, 2) for x in action]
		msg = Twist()
		msg.linear.x = self.action[0]
		msg.angular.z = self.action[1]
		# print("Lin Vel : ", msg.linear.x , "\tAng Vel : ", msg.angular.z, end = '\r')
		self.pub.publish(msg)
		rospy.sleep(0.02)

		head_to_target = self.get_heading(self.pose, self.target)

		done, reward = self.check_goal()

		obs = [(self.target[0] - self.pose[0])/GRID, (self.target[1] - self.pose[1])/GRID, head_to_target - self.pose[2]]
		obs = [round(x, 2) for x in obs]
		return np.array(obs), reward, done, info  

	def stop_bot(self):
		# print("Stopping Bot...")
		msg = Twist()
		msg.linear.x = 0.
		msg.linear.y = 0.
		msg.linear.z = 0.
		msg.angular.x = 0.
		msg.angular.y = 0.
		msg.angular.z = 0.
		self.pub.publish(msg)
		rospy.sleep(1)

	def close(self):
		pass

class DiscreteTurtleGym(gym.Env):
	"""
	Discrete Action Space Gym Environment
	State - relative x, y, theta
	Action - 4 actions - 
				0: [0.2, 0.], 1: [0., 1.25], 2: [-0.2, 0.], 3: [0., -1.25]
			 15 actions - 
			 	0: [0., -2.5], 1: [0., -1.25], 2: [0., 0.], 3: [0., 1.25], 4: [0., 2.5],
				5: [0.1, -2.5], 6: [0.1, -1.25], 7: [0.1, 0.], 8: [0.1, 1.25], 9: [0.1, 2.5],
				10: [0.2, -2.5], 11: [0.2, -1.25], 12: [0.2, 0.], 13: [0.2, 1.25], 14: [0.2, 2.5]
	Reward - 
	"""
	def __init__(self, n_actions = 4):
		super(DiscreteTurtleGym,self).__init__()		
		metadata = {'render.modes': ['console']}
		print("Initialising Turtlebot 3 Discrete Gym Environment...")
		self.action_space = spaces.Discrete(15) 
		self.actSpace = collections.defaultdict(list)
		if n_actions == 4 :			
			self.actSpace = {
				0: [0.2, 0.], 1: [0., 1.25], 2: [-0.2, 0.], 3: [0., -1.25]
			}
		else :
			self.actSpace = {
				0: [0., -2.5], 1: [0., -1.25], 2: [0., 0.], 3: [0., 1.25], 4: [0., 2.5],
				5: [0.1, -2.5], 6: [0.1, -1.25], 7: [0.1, 0.], 8: [0.1, 1.25], 9: [0.1, 2.5],
				10: [0.2, -2.5], 11: [0.2, -1.25], 12: [0.2, 0.], 13: [0.2, 1.25], 14: [0.2, 2.5]
			}

		low = np.array([-1.,-1.,-4.])
		high = np.array([1.,1.,4.])
		self.observation_space = spaces.Box(low, high, dtype=np.float32)
		self.target = [0., 0., 0.]
		self.ep_steps = 0

		self.pose = np.zeros(3) # pose_callback
		self.sector_scan = np.zeros(36) # scan_callback
		self.depth = np.zeros(1843200) # cam_callback. 
		# Depth image is compressed image of size 1280x720. data length = 1843200 (720rows x 2560step)
		
		self.action = [0., 0.] # Publisher is /cmd_vel Twist message for linear and angular velocity

		# Initialize ROS nodes
		self.sub = [0]
		self.sub[0] = rospy.Subscriber("/odom", Odometry, self.pose_callback)
		# self.sub[1] = rospy.Subscriber("scan", LaserScan, self.scan_callback)
		# self.sub[2] = rospy.Subscriber("/camera/depth/image_raw", Image, self.cam_callback)
		self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

		# Gazebo Services
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

	def pose_callback(self, pose_data) :
		# ROS Callback function for the /odom topic
		orient = pose_data.pose.pose.orientation
		q = (orient.x, orient.y, orient.z, orient.w)
		euler = self.euler_from_quaternion(q[0], q[1], q[2], q[3])
		self.pose = np.array([pose_data.pose.pose.position.x, pose_data.pose.pose.position.y, euler[2]])
		# print("Pose - ", self.pose)

	def scan_callback(self, scan_data) :
		# ROS Callback function for the /scan topic
		scan = np.array(scan_data.ranges)
		self.sector_scan = np.mean(scan.reshape(-1, 10), axis=1) # Sectorizes the lidar data to 36 sectors of 10 degrees each

	def cam_callback(self, cam_data) : 
		# ROS Callback function for the /camera/depth/image_raw topic
		self.depth = cam_data.data

	def euler_from_quaternion(self, x, y, z, w):
		'''
		Convert a quaternion into euler angles (roll, pitch, yaw)
		roll is rotation around x in radians (counterclockwise)
		pitch is rotation around y in radians (counterclockwise)
		yaw is rotation around z in radians (counterclockwise)
		'''
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll_x = math.atan2(t0, t1)

		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch_y = math.asin(t2)

		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_z = math.atan2(t3, t4)

		return roll_x, pitch_y, yaw_z # in radians

	def reset(self):
		rospy.wait_for_service('/gazebo/reset_simulation')
		try:
			self.pause()
			self.reset_simulation_proxy()
			self.unpause()
			# rospy.sleep(1)
			print('Simulation reset')
		except rospy.ServiceException as exc:
			print("Reset Service did not process request: " + str(exc))

		x = random.uniform(-1., 1.)
		y = random.choice([-1., 1.])
		self.target[0], self.target[1] = [-1., 0.4]#random.choice([[x, y], [y, x]])

		print("Reset target to : [{:.2f}, {:.2f}]".format(self.target[0], self.target[1]))
		head_to_target = self.get_heading(self.pose, self.target)
		yaw = random.uniform(head_to_target - THETA0, head_to_target + THETA0)

		obs = [(self.target[0] - self.pose[0])/GRID, (self.target[1] - self.pose[1])/GRID, head_to_target - self.pose[2]]
		obs = [round(x, 2) for x in obs]
		self.ep_steps = 0
		return np.array(obs)

	def get_distance(self,x1,x2):
		return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

	def get_heading(self, x1,x2):
		return math.atan2((x2[1] - x1[1]), (x2[0] - x1[0]))

	def get_reward(self):
		yaw_car = self.pose[2]
		head_to_target = self.get_heading(self.pose, self.target)

		alpha = head_to_target - yaw_car
		ld = self.get_distance(self.pose, self.target)
		crossTrackError = math.sin(alpha) * ld

		headingError = abs(alpha)
		alongTrackError = abs(self.pose[0] - self.target[0]) + abs(self.pose[1] - self.target[1])
		return -1*(abs(crossTrackError)**2 + alongTrackError + 3*headingError/1.57)/6

	def check_goal(self):
		done = False

		if (abs(self.pose[0]) < GRID) or (abs(self.pose[1]) < GRID):
			if (abs(self.pose[0] - self.target[0]) < THRESHOLD and abs(self.pose[1] - self.target[1]) < THRESHOLD) :
				done = True
				reward = 10
				print("Goal Reached")
				self.stop_bot()
			else :
				reward = self.get_reward()
		else:
			done = True
			reward = -1
			print("Outside range")
			self.stop_bot()

		if self.ep_steps > MAX_EP_LEN :
			print("Reached max episode length")
			done = True

		return done, reward

	def step(self, discrete_action):
		reward = 0
		done = False
		info = {}
		self.ep_steps += 1

		action = self.actSpace[discrete_action]
		
		self.action = [round(x, 2) for x in action]
		msg = Twist()
		msg.linear.x = self.action[0]
		msg.angular.z = self.action[1]
		# print("Lin Vel : ", msg.linear.x , "\tAng Vel : ", msg.angular.z)
		self.pub.publish(msg)
		time.sleep(0.1)

		head_to_target = self.get_heading(self.pose, self.target)

		done, reward = self.check_goal()

		obs = [(self.target[0] - self.pose[0])/GRID, (self.target[1] - self.pose[1])/GRID, head_to_target - self.pose[2]]
		obs = [round(x, 2) for x in obs]
		return np.array(obs), reward, done, info  

	def stop_bot(self):
		# print("Stopping Bot...")
		msg = Twist()
		msg.linear.x = 0.
		msg.linear.y = 0.
		msg.linear.z = 0.
		msg.angular.x = 0.
		msg.angular.y = 0.
		msg.angular.z = 0.
		self.pub.publish(msg)
		rospy.sleep(1)

	def close(self):
		pass

class ContinuousTurtleObsGym(gym.Env):
	"""
	Continuous Action Space Gym Environment w/ Obstacles
	State - relative x, y, theta, [sectorized lidar scan]
	Action - linear vel range {0. , 0.22}, 
			 angular vel range {-2.84, 2.84}
	Reward - 
	"""
	def __init__(self):
		super(ContinuousTurtleObsGym,self).__init__()		
		metadata = {'render.modes': ['console']}
		print("Initialising Turtlebot 3 Continuous Gym Obstacle Environment...")
		self.action_space = spaces.Box(np.array([0., -2.84]), np.array([0.22, 2.84]), dtype = np.float16) # max rotational velocity of burger is 2.84 rad/s
		low = np.concatenate((np.array([-1.,-1.,-4.]), np.zeros(36)))
		high = np.concatenate((np.array([1.,1.,4.]), np.ones(36)*4.))
		self.observation_space = spaces.Box(low, high, dtype=np.float16)
		self.target = [0., 0., 0.]
		self.ep_steps = 0

		self.pose = np.zeros(3) # pose_callback
		self.sector_scan = np.zeros(36) # scan_callback
		self.depth = np.zeros(1843200) # cam_callback
		# Depth image is compressed image of size 1280x720. data length = 1843200 (720rows x 2560step)
		
		self.action = [0., 0.] # Publisher is /cmd_vel Twist message for linear and angular velocity

		# Initialize ROS nodes
		self.sub = [0, 0]
		self.sub[0] = rospy.Subscriber("/odom", Odometry, self.pose_callback)
		self.sub[1] = rospy.Subscriber("scan", LaserScan, self.scan_callback)
		# self.sub[2] = rospy.Subscriber("/camera/depth/image_raw", Image, self.cam_callback)
		self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

		# Gazebo Services
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

		self.stop_bot()
		self.reset_pose()

	def pose_callback(self, pose_data) :
		# ROS Callback function for the /odom topic
		orient = pose_data.pose.pose.orientation
		q = (orient.x, orient.y, orient.z, orient.w)
		euler = self.euler_from_quaternion(q[0], q[1], q[2], q[3])
		self.pose = np.array([pose_data.pose.pose.position.x, pose_data.pose.pose.position.y, euler[2]])
		# print("Pose - ", self.pose)

	def scan_callback(self, scan_data) :
		# ROS Callback function for the /scan topic
		scan = np.array(scan_data.ranges)
		scan = np.nan_to_num(scan, copy=False, nan=0.0, posinf=5., neginf=0.)
		self.sector_scan = np.min(scan.reshape(-1, 10), axis=1) # Sectorizes the lidar data to 36 sectors of 10 degrees each

	def cam_callback(self, cam_data) : 
		# ROS Callback function for the /camera/depth/image_raw topic
		self.depth = cam_data.data

	def euler_from_quaternion(self, x, y, z, w):
		'''
		Convert a quaternion into euler angles (roll, pitch, yaw)
		roll is rotation around x in radians (counterclockwise)
		pitch is rotation around y in radians (counterclockwise)
		yaw is rotation around z in radians (counterclockwise)
		'''
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll_x = math.atan2(t0, t1)

		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch_y = math.asin(t2)

		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_z = math.atan2(t3, t4)

		return roll_x, pitch_y, yaw_z # in radians

	def reset(self):
		self.stop_bot()
		# rospy.wait_for_service('/gazebo/reset_simulation')
		# try:
		# 	self.pause()
		# 	self.reset_simulation_proxy()
		# 	self.unpause()
		# 	# rospy.sleep(1)
		# 	print('Simulation reset')
		# except rospy.ServiceException as exc:
		# 	print("Reset Service did not process request: " + str(exc))

		x = 10*random.uniform(-1, 1)
		y = 10*random.uniform(-1, 1)
		self.target[0], self.target[1] = [x, y]

		print("Reset target to : [{:.2f}, {:.2f}]".format(self.target[0], self.target[1]))
		head_to_target = self.get_heading(self.pose, self.target)
		yaw = random.uniform(head_to_target - THETA0, head_to_target + THETA0)

		obs = [(self.target[0] - self.pose[0])/GRID, (self.target[1] - self.pose[1])/GRID, head_to_target - self.pose[2]]
		obs = [round(x, 2) for x in obs]
		self.ep_steps = 0

		return np.concatenate((np.array(obs), self.sector_scan))

	def get_distance(self,x1,x2):
		return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

	def get_heading(self, x1,x2):
		return math.atan2((x2[1] - x1[1]), (x2[0] - x1[0]))

	def get_reward(self):
		yaw_car = self.pose[2]
		head_to_target = self.get_heading(self.pose, self.target)

		alpha = head_to_target - yaw_car
		ld = self.get_distance(self.pose, self.target)
		crossTrackError = math.sin(alpha) * ld

		headingError = abs(alpha)
		alongTrackError = abs(self.pose[0] - self.target[0]) + abs(self.pose[1] - self.target[1])
		# return -1*(abs(crossTrackError)**2 + alongTrackError + 3*headingError/1.57)/6
		return -1*(headingError + alongTrackError)

	def reset_pose(self):
		rospy.wait_for_service('/gazebo/reset_simulation')
		try:
			self.pause()
			self.reset_simulation_proxy()
			self.unpause()
			# rospy.sleep(1)
			print('Simulation reset')
		except rospy.ServiceException as exc:
			print("Reset Service did not process request: " + str(exc))

	def check_goal(self):
		done = False

		if (abs(self.pose[0]) < GRID) or (abs(self.pose[1]) < GRID):
			if (abs(self.pose[0] - self.target[0]) < THRESHOLD and abs(self.pose[1] - self.target[1]) < THRESHOLD) :
				done = True
				reward = 100
				print("Goal Reached")
				self.stop_bot()
				self.reset_pose()
			else :
				reward = self.get_reward()
				if np.min(self.sector_scan) < OBS_THRESH :
					print("Collision Detected")
					reward = -100
					self.stop_bot()
					self.reset_pose()
					done = True
		else:
			done = True
			reward = -100
			print("Outside range")
			self.stop_bot()
			self.reset_pose()

		if self.ep_steps > MAX_EP_LEN :
			print("Reached max episode length")
			done = True

		return done, reward

	def step(self,action):	
		reward = 0
		done = False
		info = {}
		self.ep_steps += 1

		self.action = [round(x, 2) for x in action]
		msg = Twist()
		msg.linear.x = self.action[0]
		msg.angular.z = self.action[1]
		# print("Lin Vel : ", msg.linear.x , "\tAng Vel : ", msg.angular.z, end = '\r')
		self.pub.publish(msg)
		rospy.sleep(0.05)

		head_to_target = self.get_heading(self.pose, self.target)

		done, reward = self.check_goal()

		obs = [(self.target[0] - self.pose[0])/GRID, (self.target[1] - self.pose[1])/GRID, head_to_target - self.pose[2]]
		obs = [round(x, 2) for x in obs]
		return np.concatenate((np.array(obs), self.sector_scan)), reward, done, info  

	def stop_bot(self):
		print("Stopping Bot...")
		msg = Twist()
		msg.linear.x = 0.
		msg.linear.y = 0.
		msg.linear.z = 0.
		msg.angular.x = 0.
		msg.angular.y = 0.
		msg.angular.z = 0.
		self.pub.publish(msg)
		rospy.sleep(1)

	def close(self):
		pass

class DiscreteTurtleObsGym(gym.Env):
	"""
	Discrete Action Space Gym Environment w/ Obstacles
	State - relative x, y, theta
	Action - 4 actions - 
				0: [0.2, 0.], 1: [0., 1.25], 2: [-0.2, 0.], 3: [0., -1.25]
			 15 actions - 
			 	0: [0., -2.5], 1: [0., -1.25], 2: [0., 0.], 3: [0., 1.25], 4: [0., 2.5],
				5: [0.1, -2.5], 6: [0.1, -1.25], 7: [0.1, 0.], 8: [0.1, 1.25], 9: [0.1, 2.5],
				10: [0.2, -2.5], 11: [0.2, -1.25], 12: [0.2, 0.], 13: [0.2, 1.25], 14: [0.2, 2.5]
	Reward - 
	"""
	def __init__(self, n_actions = 4):
		super(DiscreteTurtleObsGym,self).__init__()		
		metadata = {'render.modes': ['console']}
		print("Initialising Turtlebot 3 Discrete Obs Gym Environment...")
		 
		self.actSpace = collections.defaultdict(list)
		if n_actions == 4 :	
			self.action_space = spaces.Discrete(4)		
			self.actSpace = {
				0: [0.2, 0.], 1: [0., 1.0], 2: [-0.2, 0.], 3: [0., -1.0]
			}
		else :
			self.action_space = spaces.Discrete(15)
			self.actSpace = {
				0: [0., -2.5], 1: [0., -1.25], 2: [0., 0.], 3: [0., 1.25], 4: [0., 2.5],
				5: [0.1, -2.5], 6: [0.1, -1.25], 7: [0.1, 0.], 8: [0.1, 1.25], 9: [0.1, 2.5],
				10: [0.2, -2.5], 11: [0.2, -1.25], 12: [0.2, 0.], 13: [0.2, 1.25], 14: [0.2, 2.5]
			}

		low = np.concatenate((np.array([-1.,-1.,-4.]), np.zeros(36)))
		high = np.concatenate((np.array([1.,1.,4.]), np.ones(36)*4.))
		self.observation_space = spaces.Box(low, high, dtype=np.float16)
		self.target = [0., 0., 0.]
		self.ep_steps = 0

		self.pose = np.zeros(3) # pose_callback
		self.sector_scan = np.zeros(36) # scan_callback
		self.depth = np.zeros(1843200) # cam_callback. 
		# Depth image is compressed image of size 1280x720. data length = 1843200 (720rows x 2560step)
		
		self.action = [0., 0.] # Publisher is /cmd_vel Twist message for linear and angular velocity

		# Initialize ROS nodes
		self.sub = [0, 0]
		self.sub[0] = rospy.Subscriber("/odom", Odometry, self.pose_callback)
		self.sub[1] = rospy.Subscriber("scan", LaserScan, self.scan_callback)
		# self.sub[2] = rospy.Subscriber("/camera/depth/image_raw", Image, self.cam_callback)
		self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

		# Gazebo Services
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

		self.stop_bot()
		self.reset_pose()

	def pose_callback(self, pose_data) :
		# ROS Callback function for the /odom topic
		orient = pose_data.pose.pose.orientation
		q = (orient.x, orient.y, orient.z, orient.w)
		euler = self.euler_from_quaternion(q[0], q[1], q[2], q[3])
		self.pose = np.array([pose_data.pose.pose.position.x, pose_data.pose.pose.position.y, euler[2]])
		# print("Pose - ", self.pose)

	def scan_callback(self, scan_data) :
		# ROS Callback function for the /scan topic
		scan = np.array(scan_data.ranges)
		scan = np.nan_to_num(scan, copy=False, nan=0.0, posinf=5., neginf=0.)
		self.sector_scan = np.min(scan.reshape(-1, 10), axis=1) # Sectorizes the lidar data to 36 sectors of 10 degrees each

	def cam_callback(self, cam_data) : 
		# ROS Callback function for the /camera/depth/image_raw topic
		self.depth = cam_data.data

	def euler_from_quaternion(self, x, y, z, w):
		'''
		Convert a quaternion into euler angles (roll, pitch, yaw)
		roll is rotation around x in radians (counterclockwise)
		pitch is rotation around y in radians (counterclockwise)
		yaw is rotation around z in radians (counterclockwise)
		'''
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll_x = math.atan2(t0, t1)

		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch_y = math.asin(t2)

		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_z = math.atan2(t3, t4)

		return roll_x, pitch_y, yaw_z # in radians

	def reset(self):
		# self.stop_bot()
		# rospy.wait_for_service('/gazebo/reset_simulation')
		# try:
		# 	self.pause()
		# 	self.reset_simulation_proxy()
		# 	self.unpause()
		# 	# rospy.sleep(1)
		# 	print('Simulation reset')
		# except rospy.ServiceException as exc:
		# 	print("Reset Service did not process request: " + str(exc))
		# self.reset_pose()

		y = random.uniform(-1, 1)
		x = random.choice([-1, 1])
		self.target[0], self.target[1] = [1, y]#random.choice([[x, y], [y, x]]) # 

		print("Reset target to : [{:.2f}, {:.2f}]".format(self.target[0], self.target[1]))
		head_to_target = self.get_heading(self.pose, self.target)
		yaw = random.uniform(head_to_target - THETA0, head_to_target + THETA0)

		obs = [(self.target[0] - self.pose[0])/GRID, (self.target[1] - self.pose[1])/GRID, head_to_target - self.pose[2]]
		obs = [round(x, 2) for x in obs]
		self.ep_steps = 0
		return np.concatenate((np.array(obs), self.sector_scan))

	def get_distance(self,x1,x2):
		return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

	def get_heading(self, x1,x2):
		return math.atan2((x2[1] - x1[1]), (x2[0] - x1[0]))

	def get_reward(self):
		yaw_car = self.pose[2]
		head_to_target = self.get_heading(self.pose, self.target)

		alpha = head_to_target - yaw_car
		ld = self.get_distance(self.pose, self.target)
		crossTrackError = math.sin(alpha) * ld

		headingError = abs(alpha)
		alongTrackError = abs(self.pose[0] - self.target[0]) + abs(self.pose[1] - self.target[1])
		return -1*(alongTrackError - np.min(self.sector_scan))
		# return -1*(abs(crossTrackError)**2 + alongTrackError + 5*headingError/1.57)/6

	def reset_pose(self):
		rospy.wait_for_service('/gazebo/reset_simulation')
		try:
			self.pause()
			self.reset_simulation_proxy()
			self.unpause()
			# rospy.sleep(1)
			print('Simulation reset')
		except rospy.ServiceException as exc:
			print("Reset Service did not process request: " + str(exc))

	def check_goal(self):
		done = False

		# print(abs(self.pose[0]), abs(self.pose[1]), end = '\r')

		if (abs(self.pose[0]) < GRID) or (abs(self.pose[1]) < GRID):
			if (abs(self.pose[0] - self.target[0]) < THRESHOLD and abs(self.pose[1] - self.target[1]) < THRESHOLD) :
				reward = 100
				print("Goal Reached")
				self.stop_bot()	
				self.reset_pose()
				done = True
			else :				
				reward = self.get_reward()
				if np.min(self.sector_scan) < OBS_THRESH :
					print("Collision Detected")
					reward = -100
					self.stop_bot()
					self.reset_pose()
					done = True
		else:
			self.stop_bot()
			self.reset_pose()
			done = True
			reward = -10
			print("Outside range")
			
			

		if self.ep_steps > MAX_EP_LEN :
			print("Reached max episode length")
			self.stop_bot()
			self.reset_pose()
			done = True

		return done, reward

	def step(self, discrete_action):
		reward = 0
		done = False
		info = {}
		self.ep_steps += 1

		action = self.actSpace[discrete_action]
		
		self.action = [round(x, 2) for x in action]
		msg = Twist()
		msg.linear.x = self.action[0]
		msg.angular.z = self.action[1]
		# print("Lin Vel : ", msg.linear.x , "\tAng Vel : ", msg.angular.z)
		self.pub.publish(msg)
		time.sleep(0.05)

		head_to_target = self.get_heading(self.pose, self.target)

		done, reward = self.check_goal()

		obs = [(self.target[0] - self.pose[0])/GRID, (self.target[1] - self.pose[1])/GRID, head_to_target - self.pose[2]]
		obs = [round(x, 2) for x in obs]
		return np.concatenate((np.array(obs), self.sector_scan)), reward, done, info  

	def stop_bot(self):
		print("Stopping Bot...")
		msg = Twist()
		msg.linear.x = 0.
		msg.linear.y = 0.
		msg.linear.z = 0.
		msg.angular.x = 0.
		msg.angular.y = 0.
		msg.angular.z = 0.
		self.pub.publish(msg)
		rospy.sleep(0.5)

	def close(self):
		pass

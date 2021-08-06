#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, Image, LaserScan
import numpy as np
from tf.transformations import euler_from_quaternion

class CallbackTest():
	def __init__(self):	
		
		# self.sub1 = rospy.Subscriber("/odom", Odometry, self.pose_callback)
		self.sub2 = rospy.Subscriber("/camera/depth/image_raw", Image, self.cam_callback)
		# self.sub3 = rospy.Subscriber("scan", LaserScan, self.scan_callback)
		self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
		self.pose = np.array([0., 0., 0.])

	def pose_callback(self, pose_data) :
		orient = pose_data.pose.pose.orientation
		q = (orient.x, orient.y, orient.z, orient.w)
		euler = euler_from_quaternion([q[0], q[1], q[2], q[3]])
		self.pose = np.array([pose_data.pose.pose.position.x, pose_data.pose.pose.position.y, euler[2]])
		print(self.pose)

	def cam_callback(self, cam_data) :
		print(len(cam_data.data))

	def scan_callback(self, scan_data) :
		print(len(scan_data.ranges))

if __name__ == '__main__':
	try:
		rospy.init_node('test', anonymous=True)
		env = CallbackTest()
		rospy.spin()
	except rospy.ROSInterruptException:
		pass



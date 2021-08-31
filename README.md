# Reinforcement Learning with Turtlebot in Gazebo
# TurtleBot3 with Intel RealSense D435
<img src="https://github.com/ROBOTIS-GIT/emanual/blob/master/assets/images/platform/turtlebot3/logo_turtlebot3.png" width="100">
- [ROBOTIS e-Manual for TurtleBot3](http://turtlebot3.robotis.com/)
- To install the Realsense Plugin from source follow steps from https://github.com/intel/gazebo-realsense, or as given below

# Reinforcement Learning with Stable Baselines
```
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone 
git clone 
cd ..
catkin_make
. devel/setup.bash
```

## Setup Virtual Environment for RL
```
python3 -m venv gymenv
source gymenv/bin/activate
pip3 install --upgrade pip
pip3 install pyyaml rospkg numpy tensorboard 
pip3 install <compatible version of torch https://pytorch.org/>
```
## Start training RL model
```
roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch

source gymenv/bin/activate
rosrun turtlebot3_gazebo sbtraining.py
```
# Reinforcement Learning with Custom RL Algorithm

#### Dependencies
```
turtlebot3
turtlebot3_msgs
turtlebot3_descriptions
```

## Setup Virtual Environment for RL
```
python3 -m venv gymenv
source gymenv/bin/activate
pip3 install --upgrade pip
pip3 install pyyaml rospkg numpy gym matplotlib tensorboard scikit-build cmake stable-baselines3 scipy
pip3 install <compatible version of torch https://pytorch.org/>
```
## Start training RL model
```
roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch

source gymenv/bin/activate
rosrun turtlebot3_gazebo training.py
```

### RealSense Camera Gazebo Plugin
- This Gazebo plugin simulates a RealSense camera by publishing the 4 main RealSense streams: Depth, Infrared, Infrared2 and Color. It is associated to a
RealSense model that is providade in ./models

#### Build #

1. Create a build folder under /src and make using CMAKE as follows:

    ```
    mkdir build
    cd build
    cmake ..
    make
    ```

#### Install #

The plugin binaries will be installed so that Gazebo finds them. Also the
needed models will be copied to the default gazebo models folder.

    sudo make install
    
#### Launch Simulation with D435 #

```
roslaunch turtlebot3_gazebo turtlebot3_house.launch
```

## Wiki for turtlebot3_simulations Packages
- http://wiki.ros.org/turtlebot3_simulations (metapackage)
- http://wiki.ros.org/turtlebot3_fake
- http://wiki.ros.org/turtlebot3_gazebo

## Open Source related to TurtleBot3
- [turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3)
- [turtlebot3_msgs](https://github.com/ROBOTIS-GIT/turtlebot3_msgs)
- [turtlebot3_simulations](https://github.com/ROBOTIS-GIT/turtlebot3_simulations)
- [turtlebot3_applications_msgs](https://github.com/ROBOTIS-GIT/turtlebot3_applications_msgs)
- [turtlebot3_applications](https://github.com/ROBOTIS-GIT/turtlebot3_applications)
- [turtlebot3_autorace](https://github.com/ROBOTIS-GIT/turtlebot3_autorace)
- [turtlebot3_deliver](https://github.com/ROBOTIS-GIT/turtlebot3_deliver)
- [hls_lfcd_lds_driver](https://github.com/ROBOTIS-GIT/hls_lfcd_lds_driver)
- [turtlebot3_manipulation](https://github.com/ROBOTIS-GIT/turtlebot3_manipulation.git)
- [turtlebot3_manipulation_simulations](https://github.com/ROBOTIS-GIT/turtlebot3_manipulation_simulations.git)
- [robotis_manipulator](https://github.com/ROBOTIS-GIT/robotis_manipulator)
- [open_manipulator_msgs](https://github.com/ROBOTIS-GIT/open_manipulator_msgs)
- [open_manipulator](https://github.com/ROBOTIS-GIT/open_manipulator)
- [open_manipulator_simulations](https://github.com/ROBOTIS-GIT/open_manipulator_simulations)
- [open_manipulator_perceptions](https://github.com/ROBOTIS-GIT/open_manipulator_perceptions)
- [dynamixel_sdk](https://github.com/ROBOTIS-GIT/DynamixelSDK)
- [OpenCR-Hardware](https://github.com/ROBOTIS-GIT/OpenCR-Hardware)
- [OpenCR](https://github.com/ROBOTIS-GIT/OpenCR)

## Documents and Videos related to TurtleBot3
- [ROBOTIS e-Manual for TurtleBot3](http://turtlebot3.robotis.com/)
- [ROBOTIS e-Manual for OpenManipulator](http://emanual.robotis.com/docs/en/platform/openmanipulator/)
- [ROBOTIS e-Manual for Dynamixel SDK](http://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/overview/)
- [Website for TurtleBot Series](http://www.turtlebot.com/)
- [e-Book for TurtleBot3](https://community.robotsource.org/t/download-the-ros-robot-programming-book-for-free/51/)
- [Videos for TurtleBot3 ](https://www.youtube.com/playlist?list=PLRG6WP3c31_XI3wlvHlx2Mp8BYqgqDURU)

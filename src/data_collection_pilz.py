#!/usr/bin/env python3
# Author Willow Mandil || 01/07/2022

import os
import sys
import cv2
import copy
import time
import math
import rospy
import random
import datetime
import message_filters
import moveit_msgs.msg
import moveit_commander
import geometry_msgs.msg

import numpy as np
import pandas as pd

from math import pi
from cv_bridge import CvBridge
from std_msgs.msg import String, Bool, Int16MultiArray
from shape_msgs.msg import SolidPrimitive
from xela_server.msg import XStream
from sensor_msgs.msg import JointState, Image
from moveit_msgs.msg import CollisionObject, DisplayTrajectory, MotionPlanRequest, Constraints, PositionConstraint, JointConstraint
from geometry_msgs.msg import PoseStamped, Pose
from actionlib_msgs.msg import GoalStatusArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from moveit_commander.conversions import pose_to_list

class FrankaRobot(object):
    def __init__(self):
        super(FrankaRobot, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('FrankaRobotWorkshop', anonymous=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)
        self.planning_frame = self.move_group.get_planning_frame()
        self.eef_link = self.move_group.get_end_effector_link()
        self.move_group.set_end_effector_link("panda_link8")
        self.group_names = self.robot.get_group_names()
        self.bridge = CvBridge()

        self.move_group.set_planning_pipeline_id("pilz_industrial_motion_planner") 
        self.move_group.set_planner_id("LIN")                      # Set to be the straight line planner
        print(self.move_group.get_interface_description().name)    # Print the planner being used.

        self.move_group.set_max_velocity_scaling_factor(0.2)       # scaling down velocity
        self.move_group.set_max_acceleration_scaling_factor(0.05)  # scaling down acceleration

        self.pushing_z_height        = 0.19
        self.starting_depth          = 0.3 # was 35
        self.finish_depth            = self.starting_depth + 0.05 + 0.2  # (0.5)
        self.starting_position_width = [-0.35, 0.35]
        self.pushing_orientation     = [-0.9238638957839016, 0.3827149349905697, -0.0020559535525728366, 0.0007440814108405214]

        self.joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"] 
        # self.joint_home = [-0.00045263667338251855, -0.7897728340714605, -0.0008685607692354328, -2.347460109880495, 0.0008903473842785589, 1.572969910038366, 0.7851288378097758]
        self.joint_home = [-0.02942486957764782, -0.6074509349001009, -0.009107291408774911, -2.6551141966728364, -0.012705765512177402, 2.047778068507159, 0.759312559799446]

        self.resets           = 100
        self.pushes_per_reset = 10

        # datasaving:
        self.datasave_folder = "/home/willow/Robotics/Datasets/PRI/household_object_dataset/"
        self.tactile_sub     = message_filters.Subscriber('/xServTopic', XStream)
        self.robot_sub       = message_filters.Subscriber('/joint_states', JointState)
        self.image_color_sub_side = message_filters.Subscriber('/camera/side/color/image_raw', Image)
        self.image_depth_sub_side = message_filters.Subscriber('/camera/side/color/depth_raw', Image)
        self.image_color_sub_top = message_filters.Subscriber('/camera/top/color/image_raw', Image)
        self.image_depth_sub_top = message_filters.Subscriber('/camera/top/color/depth_raw', Image)
        self.image_color_sub_ceiling = message_filters.Subscriber('/camera/ceiling/color/image_raw', Image)
        self.image_depth_sub_ceiling = message_filters.Subscriber('/camera/ceiling/color/depth_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.robot_sub, self.tactile_sub, self.image_color_sub_side, self.image_depth_sub_side, self.image_color_sub_top, self.image_depth_sub_top, self.image_color_sub_ceiling, self.image_depth_sub_ceiling],
                                                               queue_size=1, slop=0.1, allow_headerless=True)

    def pushing_actions(self):
        start_position, start_ori = self.get_robot_task_state()

        total_pushes = 0
        failure_cases = 0
        for j in range(self.resets):
            for i in range(self.pushes_per_reset):
                print("resets: {},   push: {},  total_pushes: {},   failure cases: {}".format(j, i, total_pushes, failure_cases))
                not_broken = self.go_home()
                if not_broken == False:
                    print("Error, stopping ")
                    break

                # 1. Move to random start position:
                self.move_group.set_planner_id("LIN")                      # Set to be the straight line planner
                start_y_position = random.uniform(self.starting_position_width[0], self.starting_position_width[1])
                start_pose = self.move_group.get_current_pose().pose
                start_pose.position.x = self.starting_depth
                start_pose.position.y = start_y_position
                start_pose.position.z = self.pushing_z_height
                start_pose.orientation.x = self.pushing_orientation[0]
                start_pose.orientation.y = self.pushing_orientation[1]
                start_pose.orientation.z = self.pushing_orientation[2]
                start_pose.orientation.w = self.pushing_orientation[3]

                target = self.move_group.set_pose_target(start_pose)
                trajectory = self.move_group.plan(target)
                
                if trajectory[0] == False:
                    self.go_home()
                    failure_cases += 1
                    continue
                self.move_group.go(target, wait=True)
                # check if trajectory executed:
                pos, ori = self.get_robot_task_state()
                if pos[2] > self.pushing_z_height + 0.03:
                    print("error")
                    continue


                # 2. Execute pushing action to second random position:
                need_new = True
                while need_new:
                    finish_y_position = random.uniform(self.starting_position_width[0], self.starting_position_width[1])

                    self.move_group.set_max_velocity_scaling_factor(0.5)       # scaling down velocity
                    self.move_group.set_max_acceleration_scaling_factor(0.3)   # scaling down acceleration              
                    start_position, start_ori = self.get_robot_task_state()    # calculate required angle of end effector:
                    euler_ori = euler_from_quaternion(start_ori)
                    euler_ori = list(euler_ori)
                    rotational_change = math.atan((start_y_position - finish_y_position) / (self.finish_depth - self.starting_depth))
                    euler_ori[2] += rotational_change
                    joint_goal = self.move_group.get_current_joint_values()
                    joint_goal[-1] += rotational_change

                    if joint_goal[-1] < 2.6 and joint_goal[-1] > -2.6:
                        need_new = False

                self.move_group.go(joint_goal, wait=True)
                self.move_group.stop()


                # 3. Make pushing action:
                self.move_group.set_planner_id("LIN")                      # Set to be the straight line planner
                self.move_group.set_max_velocity_scaling_factor(0.2)       # scaling down velocity
                self.move_group.set_max_acceleration_scaling_factor(0.05)  # scaling down acceleration
                _, final_ori_quat = self.get_robot_task_state()
                finish_pose = self.move_group.get_current_pose().pose
                finish_pose.position.x = self.finish_depth
                finish_pose.position.y = finish_y_position
                finish_pose.position.z = self.pushing_z_height
                finish_pose.orientation.x = final_ori_quat[0]
                finish_pose.orientation.y = final_ori_quat[1]
                finish_pose.orientation.z = final_ori_quat[2]
                finish_pose.orientation.w = final_ori_quat[3]
                target = self.move_group.set_pose_target(finish_pose)
                trajectory = self.move_group.plan(target)
                if trajectory[0] == False:
                    self.go_home()
                    failure_cases += 1
                    continue
                time_for_trajectory = float(str(trajectory[1].joint_trajectory.points[-1].time_from_start.secs) + "." +str(trajectory[1].joint_trajectory.points[-1].time_from_start.nsecs))
                self.move_group.go(target, wait=False)

                self.data_saver(time_for_trajectory)
                total_pushes += 1
            
            robot.reset_objects()
            self.go_home()
            if not_broken == False:
                print("Error, stopping ")
                break

    def data_saver(self, time_for_trajectory):
        rate                = rospy.Rate(10)
        self.robot_states   = []
        self.color_image_side = []
        self.depth_image_side = []
        self.color_image_top = []
        self.depth_image_top = []
        self.color_image_ceiling = []
        self.depth_image_ceiling = []
        self.tactile_states = []
        self.tactile_states_hf = []
        self.prev_i, self.i = 0, 1

        self.ts.registerCallback(self.read_robot_data)
        t0 = time.time()
        while not rospy.is_shutdown() and time.time() - t0 < time_for_trajectory:
            print(time_for_trajectory, "    ----     ", time.time() - t0, end="\r")
            self.i += 1
            rate.sleep()
        t1 = time.time()
        self.rate = (len(self.robot_states)) / (t1-t0)
        self.save_data()

    def read_robot_data(self, robot_joint_data, tactile_data, color_image_side, depth_image_side, color_image_top, depth_image_top, color_image_ceiling, depth_image_ceiling):
        if self.i != self.prev_i:
            self.prev_i = self.i
            ee_state = self.move_group.get_current_pose().pose            
            self.robot_states.append([robot_joint_data, ee_state])
            self.color_image_side.append(color_image_side)
            self.depth_image_side.append(depth_image_side)
            self.color_image_top.append(color_image_top)
            self.depth_image_top.append(depth_image_top)
            self.color_image_ceiling.append(color_image_ceiling)
            self.depth_image_ceiling.append(depth_image_ceiling)
            self.tactile_states.append(tactile_data.data[0])

    def format_data_for_saving(self):
        self.robot_states_formated = []
        self.tactile_states_formated = []
        self.tactile_states_formated_HF = []

        for data_sample_index in range(len(self.robot_states)):
            robot_joint_data = self.robot_states[data_sample_index][0]
            ee_state = self.robot_states[data_sample_index][1]
            self.robot_states_formated.append(list(robot_joint_data.position) + list(robot_joint_data.velocity) + list(robot_joint_data.effort) + 
                                                [ee_state.position.x, ee_state.position.y, ee_state.position.z,
                                                 ee_state.orientation.x, ee_state.orientation.y, ee_state.orientation.z, ee_state.orientation.w])

            # low freq tactile data:
            tactile_data = self.tactile_states[data_sample_index]
            tactile_vector = np.zeros(48)
            for index, i in enumerate(range(0, 48, 3)):
                tactile_vector[i]     = int(tactile_data.xyz[index].x)
                tactile_vector[i + 1] = int(tactile_data.xyz[index].y)
                tactile_vector[i + 2] = int(tactile_data.xyz[index].z)
            self.tactile_states_formated.append(tactile_vector)

    def save_data(self):
        # create new folder for this experiment:
        folder = str(self.datasave_folder + '/data_sample_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        mydir = os.mkdir(folder)

        self.format_data_for_saving()
        T0 = pd.DataFrame(self.robot_states_formated)

        robot_states_col = ["position_panda_joint1", "position_panda_joint2", "position_panda_joint3", "position_panda_joint4", "position_panda_joint5", "position_panda_joint6", "position_panda_joint7",
        "velocity_panda_joint1", "velocity_panda_joint2", "velocity_panda_joint3", "velocity_panda_joint4", "velocity_panda_joint5", "velocity_panda_joint6", "velocity_panda_joint7",
        "effort_panda_joint1", "panda_joint2", "effort_panda_joint3", "effort_panda_joint4", "panda_joint5", "effort_panda_joint6", "effort_panda_joint7",
        "ee_state_position_x", "ee_state_position_y", "ee_state_position_z", "ee_state_orientation_x", "ee_state_orientation_y", "ee_state_orientation_z", "ee_state_orientation_w"]

        T0.to_csv(folder + '/robot_state.csv', header=robot_states_col, index=False)
        np.save(folder + '/color_image_side.npy', np.array([self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough') for image in self.color_image_side]))
        np.save(folder + '/depth_image_side.npy', np.array([self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough') for image in self.depth_image_side]))
        np.save(folder + '/color_image_top.npy', np.array([self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough') for image in self.color_image_top]))
        np.save(folder + '/depth_image_top.npy', np.array([self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough') for image in self.depth_image_top]))
        np.save(folder + '/color_image_ceiling.npy', np.array([self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough') for image in self.color_image_ceiling]))
        np.save(folder + '/depth_image_ceiling.npy', np.array([self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough') for image in self.depth_image_ceiling]))
        np.save(folder + '/tactile_states.npy', np.array(self.tactile_states_formated))
        # np.save(folder + '/tactile_states_HF.npy', np.array(self.tactile_states_formated_HF))

    def go_home(self):
        self.move_group.set_max_velocity_scaling_factor(0.5)       # scaling down velocity
        self.move_group.set_max_acceleration_scaling_factor(0.5)   # scaling down acceleration
        self.move_group.set_planner_id("PTP")
        a = self.move_group.go(self.joint_home, wait=True)
        self.move_group.set_max_velocity_scaling_factor(0.2)       # scaling down velocity
        self.move_group.set_max_acceleration_scaling_factor(0.05)  # scaling down acceleration
        return a

    def get_robot_task_state(self):
        robot_ee_pose = self.move_group.get_current_pose().pose
        return [robot_ee_pose.position.x, robot_ee_pose.position.y, robot_ee_pose.position.z], [robot_ee_pose.orientation.x, robot_ee_pose.orientation.y, robot_ee_pose.orientation.z, robot_ee_pose.orientation.w]

    def create_pose(self, start_position, orientation):
        pose = PoseStamped()
        pose.header.frame_id = '/panda_link0'
        pose.pose.position.x = start_position[0]
        pose.pose.position.y = start_position[1]
        pose.pose.position.z = start_position[2]
        pose.pose.orientation.x = orientation[0]
        pose.pose.orientation.y = orientation[1]
        pose.pose.orientation.z = orientation[2]
        pose.pose.orientation.w = orientation[3]
        return pose


if __name__ == '__main__':
    robot = FrankaRobot()
    robot.pushing_actions()
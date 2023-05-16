#!/usr/bin/env python3

import os
import sys
import rospy
import imageio
import numpy as np
import moveit_commander
import tf.transformations
import matplotlib.pyplot as plt

from franka_msgs.msg import FrankaState
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl
from interactive_markers.interactive_marker_server import InteractiveMarkerServer, InteractiveMarkerFeedback

robot_pose = PoseStamped()
pose_pub = None
# [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
position_limits = [[0.2, 0.6], [-0.6, 0.6], [0.05, 0.9]]
map_translation = [0.3, -0.2, 0.0]
NUM_ANGLE_STEPS = 10


class Robot:
    def __init__(self, link_name):
        self.moving = False
        self.angle_moving = False
        self.position_index = 0
        self.angle_moves = 0
        self.initial_pose_found = False
        self.link_name = link_name

        # get start state of robot:
        state_sub = rospy.Subscriber("franka_state_controller/franka_states", FrankaState, self.franka_state_callback_start)
        self.pose_pub = rospy.Publisher("equilibrium_pose", PoseStamped, queue_size=10)
        while not self.initial_pose_found:
            rospy.sleep(1)
        state_sub.unregister()

        print(self.robot_pose)
        self.pose_pub.publish(self.robot_pose)

        # initialize the map:
        self.weight_map = WeightMap(400, 400)
        self.current_position = (self.robot_current_pose[0][0] ,self.robot_current_pose[0][1])
        self.weight_map.update(self.robot_to_map_translation(self.current_position))

        state_sub = rospy.Subscriber("franka_state_controller/franka_states", FrankaState, self.franka_state_callback)
        self.rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.rate.sleep()

    def franka_state_callback_start(self, msg):
        self.robot_pose = PoseStamped()
        initial_quaternion = tf.transformations.quaternion_from_matrix(np.transpose(np.reshape(msg.O_T_EE, (4, 4))))
        initial_quaternion = initial_quaternion / np.linalg.norm(initial_quaternion)
        self.robot_pose.pose.orientation.x = initial_quaternion[0]
        self.robot_pose.pose.orientation.y = initial_quaternion[1]
        self.robot_pose.pose.orientation.z = initial_quaternion[2]
        self.robot_pose.pose.orientation.w = initial_quaternion[3]
        self.robot_pose.pose.position.x = msg.O_T_EE[12]
        self.robot_pose.pose.position.y = msg.O_T_EE[13]
        self.robot_pose.pose.position.z = msg.O_T_EE[14]

        self.initial_pose_found = True
        self.robot_current_pose = [[self.robot_pose.pose.position.x, self.robot_pose.pose.position.y, self.robot_pose.pose.position.z], [self.robot_pose.pose.orientation.x, self.robot_pose.pose.orientation.y, self.robot_pose.pose.orientation.z, self.robot_pose.pose.orientation.w]]

    def franka_state_callback(self, msg):
        self.robot_pose = PoseStamped()
        initial_quaternion = tf.transformations.quaternion_from_matrix(np.transpose(np.reshape(msg.O_T_EE, (4, 4))))
        initial_quaternion = initial_quaternion / np.linalg.norm(initial_quaternion)
        self.robot_pose.pose.orientation.x = initial_quaternion[0]
        self.robot_pose.pose.orientation.y = initial_quaternion[1]
        self.robot_pose.pose.orientation.z = initial_quaternion[2]
        self.robot_pose.pose.orientation.w = initial_quaternion[3]
        self.robot_pose.pose.position.x = msg.O_T_EE[12]
        self.robot_pose.pose.position.y = msg.O_T_EE[13]
        self.robot_pose.pose.position.z = msg.O_T_EE[14]
        self.robot_current_pose = [[self.robot_pose.pose.position.x, self.robot_pose.pose.position.y, self.robot_pose.pose.position.z], [self.robot_pose.pose.orientation.x, self.robot_pose.pose.orientation.y, self.robot_pose.pose.orientation.z, self.robot_pose.pose.orientation.w]]

        self.initial_pose_found = True

        if not self.moving and not self.angle_moving:
            print("Calculating next position")
            next_position = self.weight_map.get_next_target()
            next_position = self.map_to_robot_translation(next_position)
            self.positions, self.angles = self.interpolate_movement(self.current_position, next_position, self.robot_current_pose[1])
            self.position_index = 0
            self.angle_moves = 0
            self.angle_moving = True
            return

        if self.angle_moving:
            print("angle moving: ", self.angle_moves)
            print(self.angles)
            self.move_to_next_position(self.positions[0], self.angles[self.angle_moves])
            self.angle_moves +=1
            if self.angle_moves == NUM_ANGLE_STEPS:
                self.angle_moving = False
                self.moving = True
                self.angle_moves = 0
            return

        if self.moving:
            print("moving: ", self.position_index)
            self.move_to_next_position(self.positions[self.position_index], self.angles[-1])

            self.weight_map.update(self.robot_to_map_translation((self.robot_current_pose[0][0], self.robot_current_pose[0][1])))

            self.current_position = (self.robot_current_pose[0][0], self.robot_current_pose[0][1])

            self.position_index += 1
            if self.position_index == len(self.positions):
                self.moving = False
            return

        plt.imshow(self.weight_map.probabilities, cmap='hot', interpolation='nearest')
        plt.colorbar()
        frame_filename = f"/home/willmandil/catkin_ws/src/tabletop_pushing/robot_position_map/inverse_prob_frame_{self.position_index}.png"
        plt.savefig(frame_filename)
        plt.close()

        self.rate.sleep()


    def interpolate_movement(self, start_position, end_position, start_angle_quat):
        sx, sy = start_position[0], start_position[1]
        ex, ey = end_position[0], end_position[1]
        # Calculate the distance between the start and end positions
        distance = np.sqrt((sx - ex)**2 + (sy - ey)**2)
        # Calculate the number of steps to take
        num_steps = int(distance / 0.015)
        if num_steps == 0:
            return [start_position]
        # Calculate the step size
        step_size = 1.0 / num_steps
        # Calculate the x and y step sizes
        x_step = (ex - sx) * step_size
        y_step = (ey - sy) * step_size
        # Create a list of positions to move to
        positions = []
        for i in range(num_steps):
            positions.append((sx + i * x_step, sy + i * y_step))

        # Calculate the angle between the start and end positions

        roll, pitch, yaw = tf.transformations.euler_from_quaternion(start_angle_quat)
        start_angle = yaw
        finish_angle = np.arctan2(ey - sy, ex - sx)

        angle_step_size = 1.0 / NUM_ANGLE_STEPS
        angles = []
        for i in range(NUM_ANGLE_STEPS):
            angles.append(start_angle + i * (finish_angle - start_angle) * angle_step_size)

        return positions, angles

    def move_to_next_position(self, position, angle):
        new_pose = PoseStamped()
        new_pose.header.frame_id = link_name
        new_pose.header.stamp = rospy.Time(0)
        new_pose.pose.position.x = position[0]
        new_pose.pose.position.y = position[1]
        new_pose.pose.position.z = new_pose.pose.position.z

        if angle > 2.5:
            angle = 2.5
        elif angle < -2.5:
            angle = -2.5

        # convert quaternion to euler
        roll, pitch, yaw = tf.transformations.euler_from_quaternion([1, 0, 0, 0])
        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, angle)

        new_pose.pose.orientation.x = quaternion[0]
        new_pose.pose.orientation.y = quaternion[1]
        new_pose.pose.orientation.z = quaternion[2]
        new_pose.pose.orientation.w = quaternion[3]

        self.pose_pub.publish(new_pose)

    def map_to_robot_translation(self, map_position):
        robot_x = (map_position[0] * 0.001) + map_translation[0]
        robot_y = (map_position[1] * 0.001) + map_translation[1]
        return (robot_x, robot_y)

    def robot_to_map_translation(self, robot_position):  # works with map_translation = [0.2, -0.2, 0.0]
        map_x = int((robot_position[0] - map_translation[0]) * 1000)
        map_y = int((robot_position[1] - map_translation[1]) * 1000)
        return (map_x, map_y)


class WeightMap:
    def __init__(self, size_x=400, size_y=400, initial_weight=1.0, update_weight=0.1, decay_rate=0.05, radius= 25):
        self.size_x = size_x
        self.size_y = size_y
        self.map = np.full((size_x, size_y), initial_weight)
        self.decay_rate = decay_rate
        self.update_weight = update_weight
        self.radius = radius
        self.probabilities = np.zeros((size_x, size_y))

    def update(self, position):
        position = (int(position[0]), int(position[1]))

        # Increase the weight at the current position and surrounding cells within radius
        for i in range(max(0, position[0]-self.radius), min(self.size_x, position[0]+self.radius+1)):
            for j in range(max(0, position[1]-self.radius), min(self.size_y, position[1]+self.radius+1)):
                distance = np.sqrt((i - position[0])**2 + (j - position[1])**2)
                if distance <= self.radius:
                    weight_increase = (1 - distance/self.radius) * self.update_weight
                    self.map[i, j] += weight_increase
        
        # Decrease the weight at the current position
        self.map += self.decay_rate

        # Ensure the weight does not go below zero
        self.map[position] = max(self.map[position], 0)

        # ensure the weight does not go above 1
        # self.map[position] = min(self.map[position], 1)

    def get_next_target(self):
        # Normalize the weights to make them probabilities
        # probabilities = self.map / np.sum(self.map)
        inverse_probabilities = -self.map / np.sum(-self.map)
        # Flatten the probabilities to 1D
        probabilities_1d = inverse_probabilities.flatten()
        # Choose a random index based on the probabilities
        target_index_1d = np.random.choice(len(probabilities_1d), p=probabilities_1d)
        # Convert the 1D index back to 2D coordinates
        target_x, target_y = np.unravel_index(target_index_1d, (self.size_x, self.size_y))
        self.probabilities = inverse_probabilities
        return (target_x, target_y)

if __name__ == "__main__":
    rospy.init_node("equilibrium_pose_node")
    link_name = rospy.get_param("~link_name")

    robot = Robot(link_name)
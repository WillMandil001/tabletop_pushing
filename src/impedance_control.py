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
initial_pose_found = False
pose_pub = None
# [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
position_limits = [[0.2, 0.6], [-0.6, 0.6], [0.05, 0.9]]
map_translation = [0.3, -0.2, 0.0]
NUM_ANGLE_STEPS = 10
ADD_TIME = 5


class FrankaRobot(object):
    def __init__(self):
        super(FrankaRobot, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        self.move_group.set_end_effector_link("panda_link8")

    def get_robot_task_state(self):
        robot_ee_pose = self.move_group.get_current_pose().pose
        return [[robot_ee_pose.position.x, robot_ee_pose.position.y, robot_ee_pose.position.z], [robot_ee_pose.orientation.x, robot_ee_pose.orientation.y, robot_ee_pose.orientation.z, robot_ee_pose.orientation.w]]


def franka_state_callback(msg):
    initial_quaternion = tf.transformations.quaternion_from_matrix(np.transpose(np.reshape(msg.O_T_EE, (4, 4))))
    initial_quaternion = initial_quaternion / np.linalg.norm(initial_quaternion)
    robot_pose.pose.orientation.x = initial_quaternion[0]
    robot_pose.pose.orientation.y = initial_quaternion[1]
    robot_pose.pose.orientation.z = initial_quaternion[2]
    robot_pose.pose.orientation.w = initial_quaternion[3]
    robot_pose.pose.position.x = msg.O_T_EE[12]
    robot_pose.pose.position.y = msg.O_T_EE[13]
    robot_pose.pose.position.z = msg.O_T_EE[14]
    global initial_pose_found
    initial_pose_found = True


class WeightMap:
    def __init__(self, size_x=400, size_y=400, initial_weight=0.0, update_weight=1.0, decay_rate=0.001, radius= 50):
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
                    self.map[i, j] -= weight_increase
        
        # Decrease the weight at the current position
        self.map += self.decay_rate

    def get_next_target(self):
        self.map = self.map + np.abs(np.min(self.map))
        probabilities = self.map / np.sum(self.map)

        # Flatten the probabilities to 1D
        probabilities_1d = probabilities.flatten()

        # Choose a random index based on the probabilities
        target_index_1d = np.random.choice(len(probabilities_1d), p=probabilities_1d)
        # Convert the 1D index back to 2D coordinates
        target_x, target_y = np.unravel_index(target_index_1d, (self.size_x, self.size_y))
        self.probabilities = probabilities
        return (target_x, target_y)


def interpolate_movement(start_position, end_position, start_angle_quat):
    sx, sy = start_position[0], start_position[1]
    ex, ey = end_position[0], end_position[1]
    # Calculate the distance between the start and end positions
    distance = np.sqrt((sx - ex)**2 + (sy - ey)**2)
    # Calculate the number of steps to take
    num_steps = int(distance / 0.005)  # 0.015
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


def move_to_next_position(position, angle):
    new_pose.header.frame_id = link_name
    new_pose.header.stamp = rospy.Time(0)
    new_pose.pose.position.x = position[0]
    new_pose.pose.position.y = position[1]
    new_pose.pose.position.z = new_pose.pose.position.z

    if angle > 2.5:
        angle = 2.5
    elif angle < -2.5:
        angle = -2.1

    # convert quaternion to euler
    roll, pitch, yaw = tf.transformations.euler_from_quaternion([1, 0, 0, 0])
    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, angle)

    new_pose.pose.orientation.x = quaternion[0]
    new_pose.pose.orientation.y = quaternion[1]
    new_pose.pose.orientation.z = quaternion[2]
    new_pose.pose.orientation.w = quaternion[3]

    pose_pub.publish(new_pose)


def map_to_robot_translation(map_position):
    robot_x = (map_position[0] * 0.001) + map_translation[0]
    robot_y = (map_position[1] * 0.001) + map_translation[1]
    return (robot_x, robot_y)


def robot_to_map_translation(robot_position):  # works with map_translation = [0.2, -0.2, 0.0]
    map_x = int((robot_position[0] - map_translation[0]) * 1000)
    map_y = int((robot_position[1] - map_translation[1]) * 1000)
    return (map_x, map_y)


if __name__ == "__main__":
    rospy.init_node("equilibrium_pose_node")
    state_sub = rospy.Subscriber("franka_state_controller/franka_states", FrankaState, franka_state_callback)
    listener = tf.TransformListener()
    link_name = rospy.get_param("~link_name")

    robot = FrankaRobot()

    # Get initial pose for the robot
    while not initial_pose_found:
        rospy.sleep(1)
    state_sub.unregister()

    pose_pub = rospy.Publisher("equilibrium_pose", PoseStamped, queue_size=10)

    new_pose = PoseStamped()
    new_pose.header.frame_id = link_name
    new_pose.header.stamp = rospy.Time(0)
    new_pose.pose.position.x = 0.5
    new_pose.pose.position.y = 0.0
    new_pose.pose.position.z = 0.03
    new_pose.pose.orientation.x = 1
    new_pose.pose.orientation.y = 0
    new_pose.pose.orientation.z = 0
    new_pose.pose.orientation.w = 0

    i = 0
    rate = rospy.Rate(1)
    while not rospy.is_shutdown() and i < 5:
        print(i)    
        pose_pub.publish(new_pose)
        i += 1
        rate.sleep()

    robot_current_pose = robot.get_robot_task_state()

    rospy.sleep(2)

    weight_map = WeightMap(400, 400)
    current_position = (robot_current_pose[0][0] ,robot_current_pose[0][1])
    weight_map.update(robot_to_map_translation(current_position))

    moving = False
    angle_moving = False
    pushing_count = 0
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        print("moving: ", moving)

        if not moving and not angle_moving:
            robot_current_pose = robot.get_robot_task_state()
            current_position = (robot_current_pose[0][0], robot_current_pose[0][1])
            next_position = weight_map.get_next_target()
            next_position = map_to_robot_translation(next_position)
            positions, angles = interpolate_movement(current_position, next_position, robot_current_pose[1])
            position_index = 0
            angle_moves = 0
            additional_time = 0
            angle_moving = True

        if angle_moving:
            print("angle_moving: ", angle_moves,  angles)
            move_to_next_position(positions[0], angles[angle_moves])
            angle_moves +=1
            if angle_moves == NUM_ANGLE_STEPS:
                angle_moving = False
                moving = True
                angle_moves = 0
                rospy.sleep(2)

        if moving:
            print("moving: ", position_index)
            move_to_next_position(positions[position_index], angles[-1])

            robot_current_pose = robot.get_robot_task_state()
            weight_map.update(robot_to_map_translation((robot_current_pose[0][0], robot_current_pose[0][1])))

            current_position = (robot_current_pose[0][0], robot_current_pose[0][1])

            position_index += 1
            if position_index == len(positions):
                moving = False
                plt.imshow(weight_map.probabilities, cmap='hot', interpolation='nearest')
                plt.colorbar()
                frame_filename = f"/home/willmandil/catkin_ws/src/tabletop_pushing/robot_position_map/robot_position_map_{pushing_count}.png"
                plt.savefig(frame_filename)
                plt.close()
                pushing_count += 1
                rospy.sleep(2)

        rate.sleep()




# print("===== Pose - from controller =====")
# print([new_pose.pose.position.x, new_pose.pose.position.y, new_pose.pose.position.z], tf.transformations.euler_from_quaternion([new_pose.pose.orientation.x, new_pose.pose.orientation.y, new_pose.pose.orientation.z, new_pose.pose.orientation.w]))

# print("===== Pose - from moveit     =====")
# print(robot_current_pose[0], tf.transformations.euler_from_quaternion(robot_current_pose[1]))

# ===== Pose - from controller =====
# [0.43151358861631106, 0.03405750663734303, 0.4551336034385356] (-3.02576480440015, -0.13171111486893855, 0.040845744706281356)
# ===== Pose - from moveit     =====
# [0.41851071511085786, 0.021600150732486736, 0.5569666132608113] (-2.966782402514508, -0.011871893588937912, -0.7378482053103548)
# offset: [0.0130028735054532, 0.012457355904856294, -0.1018330098222757] (-0.05898240188564102, 0.11983922127900064, 0.7786939490166362)


# ===== Pose - from controller =====
# [0.5188854870018338, -0.031004326693353176, 0.2794326227062691] (-2.9748417125182653, -0.030345447580374885, -1.2124338948446887)
# ===== Pose - from moveit     =====
# [0.5016729828389496, -0.0341299036377953, 0.38127232528479854] (-3.001935520621518, 0.0959427467407812, -1.988614667373374)
# offset: [0.0172125041628842, 0.003125576944442124, -0.10183970257852943] (0.0270938081032527, -0.12628819432115608, 0.7761807725286853)
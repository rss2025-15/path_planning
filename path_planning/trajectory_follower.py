import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from rclpy.node import Node
from visualization_msgs.msg import Marker

from tf_transformations import quaternion_from_euler, euler_from_quaternion

from .utils import LineTrajectory

import numpy as np
import math
import rclpy


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 1.0  # FILL IN #
        self.speed = 0.5  # FILL IN #
        self.wheelbase_length = .46  # FILL IN #

        self.trajectory = LineTrajectory("/followed_trajectory")
        self.initialized_traj = False

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)

        self.initialized_pose = False
        # self.init_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.init_callback, 1)

        #This estimated pose is gonna come from the localization lab
        self.estimated_robot_sub = self.create_subscription(PoseStamped, '/estimated_robot', self.pose_callback, 1)

        #adding this for vizualization of path on rviz
        self.loaded_traj_sub = self.create_subscription(PoseArray,
                                                 "/loaded_trajectory/path",
                                                 self.fake_callback,
                                                 1)

        self.start_sub = self.create_subscription(Marker,
                                                 "/loaded_trajectory/start_point",
                                                 self.fake_callback,
                                                 1)
        self.end_sub = self.create_subscription(Marker,
                                                 "/loaded_trajectory/end_pose",
                                                 self.fake_callback,
                                                 1)

        self.lookahead_pub = self.create_publisher(Marker, "/lookahead_pt", 1)
        self.reached_end = False

    # def init_callback(self, init_msg):
    #     2
    #     # self.initialized_pose = True

    def pose_callback(self, estimated_robot_msg):
        orientation = euler_from_quaternion([
        estimated_robot_msg.pose.orientation.x,
        estimated_robot_msg.pose.orientation.y,
        estimated_robot_msg.pose.orientation.z,
        estimated_robot_msg.pose.orientation.w
    ])[-1]

        robot_pose = np.array([
            estimated_robot_msg.pose.position.x,
            estimated_robot_msg.pose.position.y,
            orientation
        ])

        # if self.initialized_traj and self.initialized_pose:
        if self.initialized_traj:
            # self.get_logger().info(f'in the loop')

            # FINDING CLOSEST SEGMENT
            # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment/1501725#1501725
            dist_to_segments = np.empty(len(self.trajectory.points)-1)
            # self.get_logger().info(f'current distances : {dist_to_segments}')
            for i in range(len(self.trajectory.points)-1):
                s_1 = np.array(self.trajectory.points[i])
                s_2 = np.array(self.trajectory.points[i+1])
                norm_segment = np.linalg.norm(s_2-s_1)
                if norm_segment == 0:
                    dist_to_segments[i] = np.linalg.norm([robot_pose[0] - s_1[0], robot_pose[1] - s_2[1]])
                    # self.get_logger().info(f'published a distance of zero')
                else:
                    # t = ((robot_pose[0] - s_1[0])*(s_2[0] - s_1[0]) + (robot_pose[1] - s_1[1])*(s_2[1] - s_1[1]))/np.linalg.norm(s_2-s_1)
                    t = ((robot_pose[0] - s_1[0])*(s_2[0] - s_1[0]) + (robot_pose[1] - s_1[1])*(s_2[1] - s_1[1])) / (np.linalg.norm(s_2 - s_1) ** 2)


                    t = np.clip(t, 0., 1.)

                    hehe = np.array([s_1[0] + t*(s_2[0] - s_1[0]), s_1[1] + t*(s_2[1] - s_1[1])])
                    dist_to_segments[i] = np.linalg.norm([hehe[0] - robot_pose[0], hehe[1] - robot_pose[1]])
                    # self.get_logger().info(f'published a distance of {np.linalg.norm([hehe[0] - robot_pose[0], hehe[1] - robot_pose[1]])}')

            # CHOOSING SEGMENT THAT FALLS ON LOOKAHEAD
            # https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm/86428#86428
            lookahead_angle = None
            segments_checked = 0
            too_far = True
            # self.get_logger().info(f"dist_to_segments {dist_to_segments}")
            while lookahead_angle is None and segments_checked < len(self.trajectory.points)-1:
                #for no assignment error
                angle_in_robot_frame = 0.0
                i = np.argmin(dist_to_segments) # closest segment currently

                segments_checked += 1
                # self.get_logger().info(f"distance to the shortest segment was {dist_to_segments[i]}")
                s_2 = np.array(self.trajectory.points[i+1])
                s_1 = np.array(self.trajectory.points[i])

                # self.get_logger().info(f'the first point on trajectory was: {s_1}')
                # self.get_logger().info(f'the second point on trajectory was: {s_1}')

                segment_v = s_2 - s_1

                # WE HAVE DECIDED WE NEED TO LOOK AT BOTH CLOSEST AND SECOND CLOSEST SEGMENTS!!!

                a = np.dot(segment_v, segment_v)
                b = 2*np.dot(segment_v, s_1 - robot_pose[0:2])
                c = np.dot(s_1, s_1) + np.dot(robot_pose[0:2], robot_pose[0:2]) - 2*np.dot(s_1, robot_pose[0:2]) - self.lookahead**2
                disc = b**2 - 4*a*c

                if disc < 0: # no intersection like ever even if the segment was extended
                    # too_far = True
                    continue

                # parametrized intersection pts
                t_pts = np.array([(-b + math.sqrt(disc))/(2*a), (-b - math.sqrt(disc))/(2*a)])

                  #we only want to consider the first index of the intersection since that gives us the forward distance
                forward_t = t_pts[0]
                #if forward t not in range we dont want it
                if forward_t <= 0 or forward_t >=1:
                    dist_to_segments[i] = float('inf')
                    continue


                too_far = False

                pts = s_1 + forward_t*(segment_v)

                # if i == len(self.trajectory.points)-2:
                dist_to_end = np.linalg.norm([self.trajectory.points[-1][0] - robot_pose[0], self.trajectory.points[-1][1] - robot_pose[1]])
                self.get_logger().info(f'distance to end is {dist_to_end}')
                if dist_to_end < 1:
                    self.reached_end = True



                #lets find look ahead angle
                #this angle is in the world frame
                lookahead_angle = math.atan2(pts[1]-robot_pose[1], pts[0] - robot_pose[0])
                lookahead_angle_in_degrees = lookahead_angle*180/math.pi

                robot_angle_in_degrees = robot_pose[2]*180/math.pi
                angle_in_robot_frame = self.relative_angle_rad(lookahead_angle, robot_pose[2])
                angle_in_robot_frame_degrees = angle_in_robot_frame * 180/math.pi
                # self.get_logger().info(f"current look ahead is fixed to {lookahead_angle_in_degrees} degrees")
                # self.get_logger().info(f"current the robot is headed to {robot_angle_in_degrees} degrees")
                # self.get_logger().info(f"The angle of the point in robot's frame is {angle_in_robot_frame_degrees} degrees")


            # # ACTUAL PURE PURSUIT
            # self.get_logger().info(f"lookahead angle{lookahead_angle}")
            self.get_logger().info(f"too far parameter is set to {too_far}")
            if math.sin(angle_in_robot_frame) == 0:
                turn_radius = 1000
            else:
                turn_radius = self.lookahead / (2*math.sin(angle_in_robot_frame)) if not too_far else 0.
            steer_angle = math.atan(self.wheelbase_length/turn_radius) if not too_far else 0.
            # self.cmd_speed = max(1.0-math.exp(-self.exp_speed_coeff*(lookahead-self.parking_distance)),self.close_speed) # this is from panos code

            self.get_logger().info(f"currently the car is being steered at the steering angle {steer_angle}")
            self.get_logger().info(f"reached end parameter is set to {self.reached_end}")

            self.cmd_speed = 1.0 if not self.reached_end else 0.0
            self.drive_cmd(steer_angle, self.cmd_speed)




    def normalize_angle_rad(self, angle):
        """ Normalize angle to [-π, π) radians """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def relative_angle_rad(self, object_angle, robot_angle):
        return self.normalize_angle_rad(object_angle - robot_angle)
    def drive_cmd(self, steer, speed = 1.0):
        drive_cmd_drive = AckermannDriveStamped()
        drive_cmd_drive.drive.speed = speed
        drive_cmd_drive.drive.steering_angle = steer
        drive_cmd_drive.drive.steering_angle_velocity = 0.0
        drive_cmd_drive.drive.acceleration = 0.0
        drive_cmd_drive.drive.jerk = 0.0
        drive_cmd_drive.header.stamp = self.get_clock().now().to_msg()
        self.drive_pub.publish(drive_cmd_drive)

    # def stop_cmd(self):
    #     stop_cmd_drive = AckermannDriveStamped()
    #     stop_cmd_drive.drive.speed = 0.0
    #     stop_cmd_drive.drive.steering_angle = 0.0
    #     stop_cmd_drive.drive.steering_angle_velocity = 0.0
    #     stop_cmd_drive.drive.acceleration = 0.0
    #     stop_cmd_drive.drive.jerk = 0.0
    #     stop_cmd_drive.header.stamp = self.get_clock().now().to_msg()
    #     self.drive_pub.publish(stop_cmd_drive)


    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True


    def fake_callback(self, extra):
       return


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()

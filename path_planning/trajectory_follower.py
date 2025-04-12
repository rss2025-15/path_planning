import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from visualization_msgs.msg import Marker

from tf_transformations import quaternion_from_euler, euler_from_quaternion

from .utils import LineTrajectory

import numpy as np
import math


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 1.  # FILL IN #
        self.speed = 1.  # FILL IN #
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
        self.init_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.init_callback, 1)
        # for getting robot pose in map frame -- change this to localization output for real world
        self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_callback, 1)
        # this might not work -- if not listen to transform of map to baselink


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


    def init_callback(self, init_msg):
        self.initialized_pose = True

    def pose_callback(self, odometry_msg):
        orientation = euler_from_quaternion([odometry_msg.pose.pose.orientation.x, odometry_msg.pose.pose.orientation.y, odometry_msg.pose.pose.orientation.z, odometry_msg.pose.pose.orientation.w])[-1]
        robot_pose = np.array([odometry_msg.pose.pose.position.x, odometry_msg.pose.pose.position.y, orientation])

        if self.initialized_traj and self.initialized_pose:
            # FINDING CLOSEST SEGMENT
            # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment/1501725#1501725
            dist_to_segments = np.empty(len(self.trajectory.points)-1)

            for i in range(len(self.trajectory.points)-1):
                s_1 = np.array(self.trajectory.points[i])
                s_2 = np.array(self.trajectory.points[i+1])
                norm_segment = np.linalg.norm(s_2-s_1)
                if norm_segment == 0: dist_to_segments[i] = np.linalg.norm([robot_pose[0] - s_1[0], robot_pose[1] - s_2[1]])
                else:
                    t = ((robot_pose[0] - s_1[0])*(s_2[0] - s_1[0]) + (robot_pose[1] - s_1[1])*(s_2[1] - s_1[1]))/np.linalg.norm(s_2-s_1)
                    t = np.clip(t, 0., 1.)

                    hehe = np.array([s_1[0] + t*(s_2[0] - s_1[0]), s_1[1] + t*(s_2[1] - s_1[1])])
                    dist_to_segments[i] = np.linalg.norm([hehe[0] - robot_pose[0], hehe[1] - robot_pose[1]])

            # CHOOSING SEGMENT THAT FALLS ON LOOKAHEAD
            # https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm/86428#86428
            lookahead_angle = None
            segments_checked = 0
            too_far = False
            # self.get_logger().info(f"dist_to_segments {dist_to_segments}")
            while lookahead_angle is None and segments_checked < len(self.trajectory.points)-1:

                i = np.argmin(dist_to_segments) # closest segment currently
                segments_checked += 1
                # self.get_logger().info(f"{i}")
                s_2 = np.array(self.trajectory.points[i+1])
                s_1 = np.array(self.trajectory.points[i])
                segment_v = s_2 - s_1

                # WE HAVE DECIDED WE NEED TO LOOK AT BOTH CLOSEST AND SECOND CLOSEST SEGMENTS!!!

                a = np.dot(segment_v, segment_v)
                b = 2*np.dot(segment_v, s_1 - robot_pose[0:2])
                c = np.dot(s_1, s_1) + np.dot(robot_pose[0:2], robot_pose[0:2]) - 2*np.dot(s_1, robot_pose[0:2]) - self.lookahead**2
                disc = b**2 - 4*a*c

                if disc < 0: # no intersection like ever even if the segment was extended
                    too_far = True
                    break

                # parametrized intersection pts
                t_pts = np.array([(-b + math.sqrt(disc))/(2*a), (-b - math.sqrt(disc))/(2*a)])
                t_intersected_pts = t_pts[(t_pts >= 0) & (t_pts <= 1)]
                self.get_logger().info(f"t_intersected_pts{t_intersected_pts}")

                if len(t_intersected_pts) == 0: # no intersection within segment
                    dist_to_segments[i] = float('inf')
                    continue

                pts = s_1 + t_intersected_pts[-1]*(segment_v)
                self.get_logger().info(f"pts{pts}")
                # angles = np.arctan2(pts[:,1], pts[:,0])
                # lookahead_angle = angles[np.argmin(np.abs(angles - robot_pose[2]))]
                lookahead_angle = math.atan2(pts[0] - robot_pose[0], pts[1]-robot_pose[1])

            # ACTUAL PURE PURSUIT
            turn_radius = self.lookahead / (2*math.sin(lookahead_angle)) if not too_far else 0.
            steer_angle = math.atan(self.wheelbase_length/turn_radius) if not too_far else 0.
            # self.cmd_speed = max(1.0-math.exp(-self.exp_speed_coeff*(lookahead-self.parking_distance)),self.close_speed) # this is from panos code
            self.cmd_speed = 1.0
            self.drive_cmd(steer_angle, self.cmd_speed)
            self.get_logger().info(f'steering {steer_angle} >:)')

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

    def fake_callback():
       return


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()

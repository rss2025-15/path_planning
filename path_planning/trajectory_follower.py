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

        self.lookahead = .5  # FILL IN #
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
        # self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_callback, 1)
        # this might not work -- if not listen to transform of map to baselink

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


    def init_callback(self, init_msg):
        self.initialized_pose = True

    def pose_callback(self, estimated_robot_msg):
        # self.get_logger().info(f'{type(estimated_robot_msg)} is the message type' )
        # orientation = euler_from_quaternion([odometry_msg.pose.pose.orientation.x, odometry_msg.pose.pose.orientation.y, odometry_msg.pose.pose.orientation.z, odometry_msg.pose.pose.orientation.w])[-1]
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

        # self.get_logger().info(f'current robot pose: {robot_pose}')
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
            segments_got = 0
            too_far = False
            t_intersected_pts = np.empty((0,))
            t_intersected_coeffs = []
            while segments_checked < len(self.trajectory.points)-1 and segments_got < 2:
                self.get_logger().info(f"dist to segments {dist_to_segments}")
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
                    # too_far = True
                    self.get_logger().info(f"too far is true")
                    break

                # parametrized intersection pts
                t_pts = np.array([(-b + math.sqrt(disc))/(2*a), (-b - math.sqrt(disc))/(2*a)])
                if t_pts[(t_pts >= 0) & (t_pts <= 1)].any():
                    t_intersected_pts = np.append(t_intersected_pts, t_pts[(t_pts >= 0) & (t_pts <= 1)][0])
                    t_intersected_coeffs.append((s_1, segment_v))
                    self.get_logger().info(f"t_intersected_pts {t_intersected_pts}")
                    self.get_logger().info(f"t_intersected_coeffs {t_intersected_coeffs}")

                segments_got += 1
                
                dist_to_segments[i] = float('inf')
                
            if t_intersected_pts.any():
                pt = t_intersected_coeffs[-1][0] + t_intersected_pts[-1]*(t_intersected_coeffs[-1][1])

                map_wrt_robot = self.compute_transform_from_to(tuple(robot_pose), (0, 0, 0))
                lookahead_pt = self.compose_transforms(map_wrt_robot, (pt[0], pt[1], 0))
                

                self.get_logger().info(f"lookahead pt in robot frame{lookahead_pt}")
                lookahead_angle = math.atan2(lookahead_pt[1], lookahead_pt[0])
                self.get_logger().info(f"lookahead angle {lookahead_angle}")    

                # ACTUAL PURE PURSUIT
                turn_radius = self.lookahead / (2*math.sin(lookahead_angle))
                steer_angle = math.atan(self.wheelbase_length/turn_radius)
                # self.cmd_speed = max(1.0-math.exp(-self.exp_speed_coeff*(lookahead-self.parking_distance)),self.close_speed) # this is from panos code
                self.cmd_speed = 1.0
                self.drive_cmd(steer_angle, self.cmd_speed)
                self.get_logger().info(f'steering {steer_angle} >:)')
            else:
                self.drive_cmd(0.0, 1.0)
                self.get_logger().info(f'too far, steering {0.0} >:)')

    def compute_transform_from_to(self, from_pos, to_pos):
        from_x, from_y, from_theta = from_pos
        to_x, to_y, to_theta = to_pos
        dx = np.cos(from_theta)*(to_x-from_x)+np.sin(from_theta)*(to_y-from_y)
        dy = -np.sin(from_theta)*(to_x-from_x)+np.cos(from_theta)*(to_y-from_y)
        dtheta = to_theta - from_theta
        return (dx, dy, dtheta)

    def compose_transforms(self, t1, t2):
        t1_dx, t1_dy, t1_dtheta = t1
        t2_dx, t2_dy, t2_dtheta = t2
        t_dx = t1_dx+np.cos(t1_dtheta)*t2_dx-np.sin(t1_dtheta)*t2_dy
        t_dy = t1_dy+np.sin(t1_dtheta)*t2_dx+np.cos(t1_dtheta)*t2_dy
        t_dtheta = t1_dtheta+t2_dtheta
        return (t_dx, t_dy, t_dtheta)

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

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

from tf_transformations import quaternion_from_euler, euler_from_quaternion

from .utils import LineTrajectory

import numpy as np


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = .1  # FILL IN #
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

        # for getting robot pose in map frame -- change this to localization output for real world
        self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_callback, 1)
        # this might not work -- if not listen to transform of map to baselink

        
    def pose_callback(self, odometry_msg):
        orientation = euler_from_quaternion([odometry_msg.pose.pose.orientation.x, odometry_msg.pose.pose.orientation.y, odometry_msg.pose.pose.orientation.z, odometry_msg.pose.pose.orientation.w])[-1]
        robot_pose = np.array([odometry_msg.pose.pose.position.x, odometry_msg.pose.pose.position.y, orientation])

        if self.initialized_traj:
            # FINDING CLOSEST SEGMENT
            # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment/1501725#1501725
            dist_to_segments = np.array(len(self.trajectory.distances))

            for i in len(self.trajectory.distances):
                s_1 = self.trajectory.points[i]
                s_2 = self.trajectory.points[i+1]
                norm_segment = np.linalg.norm(s_2-s_1)
                if norm_segment == 0: dist_to_segments[i] = np.linalg.norm([robot_pose[0] - s_1[0], robot_pose[1] - s_2[1]])
                else:
                    t = ((robot_pose[0] - s_1[0])*(s_2[0] - s_1[0]) + (robot_pose[1] - s_1[1])*(s_2[1] - s_1[1]))/np.linalg.norm(s_2-s_1)
                    t = np.clip(t, 0., 1.)

                    hehe = np.array([s_1[0] + t*(s_2[0] - s_1[0]), s_1[1] + t*(s_2[1] - s_1[1])])
                    dist_to_segments[i] = np.linalg.norm([hehe[0] - robot_pose[0], hehe[1] - robot_pose[1]])

            # CHOOSING SEGMENT THAT FALLS ON LOOKAHEAD
            # https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm/86428#86428 
            lookahead_pt = None
            while lookahead_pt is None:
                i = np.argmin(dist_to_segments) # closest segment currently

                # WE HAVE DECIDED WE NEED TO LOOK AT BOTH CLOSEST AND SECOND CLOSEST SEGMENTS!!!
                
                segment_v = self.trajectory.points[i+1] - self.trajectory.points[i] # V
                dist_to_start = self.trajectory.points[i] - robot_pose[0:1] # P1-Q

                a = np.dot(segment_v, segment_v)
                b = 2*np.dot(segment_v, dist_to_start)
                c = np.dot(self.trajectory.points[i], self.trajectory.points[i]) + np.dot(robot_pose[0:1], robot_pose[0:1]) - 2*np.dot(self.trajectory.points[i], robot_pose[0:1]) - self.lookahead**2
                disc = b**2 - 4*a*c

                if disc < 0: # no intersection like ever even if the segment was extended
                    dist_to_segments[i] = float('inf')

                    # ADD SOMETHING TO MAKE THE CAR JUST DRIVE UNTIL IT IS CLOSE TO THE TRAJECTORY!!
                    continue 

                # parametrized intersection pts
                t_pts = np.array([(-b + math.sqrt(disc))/(2*a), (-b - math.sqrt(disc))/(2*a)])
                t_intersected_pts = t_pts[(t_pts >= 0) & (t_pts <= 1)] 

                if len(t_intersected_pts) == 0: # no intersection within segment
                    dist_to_segments[i] = float('inf')
                    continue 
                
                pts = self.trajectory.points[i] + t_intersected_pts*(np.linalg.norm(segment_v))
                angles = np.arctan2(pts[:,1], pts[:,0])
                # lookahead_pt = pts[np.argmin(np.abs(angles - robot_pose[2]))] # for getting lookahead pt in map
                lookahead_angle = angles[np.argmin(np.abs(angles - robot_pose[2]))]

            # ACTUAL PURE PURSUIT
            turn_radius = self.lookahead / (2*math.sin(lookahead_angle))
            steer_angle = math.atan(self.wheelbase_length/turn_radius)
            # self.cmd_speed = max(1.0-math.exp(-self.exp_speed_coeff*(lookahead-self.parking_distance)),self.close_speed) # this is from panos code
            self.cmd_speed = 1.0
            self.drive_cmd(steer_angle, self.cmd_speed)
            self.get_logger().info('steering {steer_angle} >:)')
                    
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




def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()

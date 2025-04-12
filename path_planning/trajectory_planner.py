import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
import numpy as np

class Cell:
    def __init__(self):
        self.parent_r = 0  
        self.parent_c = 0  
        self.f = float('inf')  # f = g + h
        self.g = 0  # actual cost
        self.h = float('inf')  # heuristic cost

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.map = OccupancyGrid()
        self.map_data = np.array([0, 0])
        self.downsampled_map = None
        self.downsampling_factor = 1
        self.ROWS = None
        self.COLS = None
        self.DOWNSAMPLED_ROWS = None
        self.DOWNSAMPLED_COLS = None
        self.pos = [0, 0]


    def map_cb(self, msg):
        self.map = msg
        self.ROWS = msg.info.height
        self.COLS = msg.info.width
        map_info = np.array(msg.data).reshape((self.ROWS, self.COLS)) # reshape flattened array
        self.map_data = map_info
        self.downsampled_map = map_info[::self.downsampling_factor, ::self.downsampling_factor]
        self.DOWNSAMPLED_ROWS = self.downsampled_map.shape[0]
        self.DOWNSAMPLED_COLS = self.downsampled_map.shape[1]

    def pose_cb(self, pose):
        raise NotImplementedError

    def goal_cb(self, msg):
        raise NotImplementedError

    def plan_path(self, start_point, end_point, map):
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()

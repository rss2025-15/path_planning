import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
import numpy as np
import heapq
import math
import cv2 
import imageio
import sys

np.set_printoptions(threshold=sys.maxsize)

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
        self.map_data = None
        self.origin = None
        self.downsampled_map = None
        self.downsampling_factor = 10
        self.ROWS = None
        self.COLS = None
        self.RES = None
        self.DOWNSAMPLED_ROWS = None
        self.DOWNSAMPLED_COLS = None
        self.pos = (0, 0)
        self.dilation = 5
        self.map_initialized = False

        self.get_logger().info("Trajectory planner node started")


    def map_cb(self, msg):
        self.map_initialized = True
        self.map = msg
        self.ROWS = msg.info.height
        self.COLS = msg.info.width
        self.RES = msg.info.resolution
        self.origin = (msg.info.origin.position.x, msg.info.origin.position.y, self.euler_from_quaternion(msg.info.origin.orientation)[2])
        self.get_logger().info("Map size: {} x {}".format(self.ROWS, self.COLS))

        g = np.transpose(np.reshape(msg.data, (self.ROWS, self.COLS)))
        self.map = np.where(np.logical_or(g == -1, g == 100), 1, g)
        # self.get_logger().info(f"map: {self.map}")
        self.map = cv2.dilate(self.map.astype('uint8'), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilation, self.dilation)))

        # self.downsampled_map = self.map[::self.downsampling_factor, ::self.downsampling_factor]
        self.downsampled_map = self.maxpool2d(self.map, self.downsampling_factor)
        self.DOWNSAMPLED_ROWS = self.downsampled_map.shape[0]
        self.DOWNSAMPLED_COLS = self.downsampled_map.shape[1]
        self.get_logger().info("Downsampled map size: {} x {}".format(self.DOWNSAMPLED_ROWS, self.DOWNSAMPLED_COLS))
        self.get_logger().info("Map received")

    def pose_cb(self, pose):
        self.pos = (pose.pose.pose.position.x, pose.pose.pose.position.y)
        self.get_logger().info("Pose received")

    def goal_cb(self, msg):
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info("Goal received")
        # imageio.imwrite("./src/path_planning/map.png", self.downsampled_map)
        self.plan_path(self.pos, (msg.pose.position.x, msg.pose.position.y), self.downsampled_map)

    def plan_path(self, start_point, end_point, map):
        start_time = self.get_clock().now()
        resolution = self.RES
        
        if not self.map_initialized:
            self.get_logger().warn("waiting for valid map")
            return

        # convert map to pixel coordinates
        self.get_logger().info("Converting map to pixel coordinates")
        self.get_logger().info("Planning path from {} to {}".format(start_point, end_point))
        self.get_logger().info(f"Origin: {self.origin}")

        start = self.grid_to_map(start_point, resolution)
        goal = self.grid_to_map(end_point, resolution)

        self.get_logger().info("Start: {}".format(start))
        self.get_logger().info("Goal: {}".format(goal))
        path = self.a_star(start, goal, self.downsampled_map)
        if path is None:
            self.get_logger().info("No path found")
            return
        
        self.get_logger().info("Path found")
        self.get_logger().info("Path length: {}".format(len(path)))
        self.get_logger().info("Path: {}".format(path))
        for point in path:
            self.trajectory.addPoint(self.map_to_grid(point, resolution))

        end_time = self.get_clock().now()
        elapsed_time = (end_time - start_time).nanoseconds / 1e9
        self.get_logger().info("Path planning took {} seconds".format(elapsed_time))
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


    # HELPER FUNCTIONS

    def maxpool2d(self, array, pool_size):
        """
        Downsample a 2D array using max pooling with the specified kernel size.

        Parameters
        ----------
        array : numpy.ndarray
            The input 2D array.
        pool_size : int
            The size of the 2D kernel (e.g., 2 for 2x2 pooling).

        Returns
        -------
        numpy.ndarray
            The downsampled array with the max values in each pooling window.
        """
        h, w = array.shape
        # Ensure the array dimensions are multiples of pool_size
        new_h, new_w = h // pool_size, w // pool_size

        # Trim the array to be evenly divisible by pool_size
        array = array[:new_h * pool_size, :new_w * pool_size]

        # Reshape, then take maximum along both pooling dimensions
        reshaped = array.reshape(new_h, pool_size, new_w, pool_size)
        pooled = reshaped.max(axis=(1, 3))

        return pooled

    def is_valid(self, position):
        in_bounds = (0 <= position[0] < self.DOWNSAMPLED_ROWS and 0 <= position[1] < self.DOWNSAMPLED_COLS)
        is_obstacle = (self.downsampled_map[position[0]][position[1]] == 1 or self.downsampled_map[position[0]][position[1]] == -1)
        return in_bounds and not is_obstacle
    
    def is_goal(self, position, goal):
        return position == goal

    def map_to_grid(self, pos, resolution):
        # modify when using downsampling
        x = float(pos[0]) * resolution * self.downsampling_factor
        y = float(pos[1]) * resolution * self.downsampling_factor
        x_, y_, _ = self.compose_transforms(self.compute_transform_from_to(self.origin, (0,0,0)), (x, y, 0))
        return (x_, y_)

    def grid_to_map(self, pos, resolution):
        # modify when using downsampling
        x_pixel, y_pixel, _ = self.compose_transforms(self.origin, (pos[0], pos[1], 0))
        return (int(x_pixel//(resolution * self.downsampling_factor)), int(y_pixel//(resolution * self.downsampling_factor)))
    
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
    
    def euler_from_quaternion(self, quat):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians
    
    def heuristic(self, start, goal):
        # euclidean distance
        return np.linalg.norm(np.array(start) - np.array(goal))
    
    def a_star(self, start, goal, grid):
        """everything in grid coordinates"""
        self.get_logger().info("A* path planning")
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        steps = 0

        while open_list:
            steps += 1
            _, current = heapq.heappop(open_list)
            if self.is_goal(current, goal):
                self.get_logger().info("Goal found")
                # reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path # includes start node
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_valid(neighbor):
                    continue

                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return None  # no path found

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()

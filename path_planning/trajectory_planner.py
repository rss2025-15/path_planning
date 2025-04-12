import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
import numpy as np
import heapq

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        # self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        # self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        # self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value
        self.odom_topic = "/odom"
        self.map_topic = "/map"
        self.initial_pose_topic = "/initialpose"

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
        self.pos = (0, 0)

        self.get_logger().info("Trajectory planner node started")


    def map_cb(self, msg):
        self.get_logger().info("Map received")
        self.map = msg
        self.ROWS = msg.info.height
        self.COLS = msg.info.width
        map_info = np.array(msg.data).reshape((self.ROWS, self.COLS)) # reshape flattened array
        self.map_data = map_info
        self.downsampled_map = map_info[::self.downsampling_factor, ::self.downsampling_factor]
        self.DOWNSAMPLED_ROWS = self.downsampled_map.shape[0]
        self.DOWNSAMPLED_COLS = self.downsampled_map.shape[1]

    def pose_cb(self, pose):
        self.pos = (pose.pose.pose.position.x, pose.pose.pose.position.y)
        self.get_logger().info("Pose received")
        self.trajectory.clear()

    def goal_cb(self, msg):
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info("Goal received")
        self.plan_path(self.pos, (msg.pose.position.x, msg.pose.position.y), self.map)

    def plan_path(self, start_point, end_point, map, map_data):
        map_info = map.info

        if map_info.resolution == 0.0:
            self.get_logger().warn("Map resolution is 0, waiting for valid map")
            return

        # convert map to pixel coordinates
        start = self.map_to_grid(start_point, map_info)
        goal = self.map_to_grid(end_point, map_info)

        path = self.a_star(start, goal, self.downsampled_map)
        if path is None:
            self.get_logger().info("No path found")
            return
        
        self.get_logger().info("Path found")
        self.get_logger().info("Path length: {}".format(len(path)))
        self.get_logger().info("Path: {}".format(path))
        for point in path:
            self.trajectory.addPoint(self.grid_to_map(point, map_info))
            
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


    # HELPER FUNCTIONS
    def is_valid(self, position):
        return  0 <= position[0] < self.DOWNSAMPLED_ROWS and 0 <= position[1] < self.DOWNSAMPLED_COLS

    def is_obstacle(self, grid, position):
        return not grid[position[0]][position[1]] == 0
    
    def is_goal(self, position, goal):
        return position == goal
    
    def map_to_grid(self, pos, map_info):
        grid_x = int((pos[0] - map_info.origin.position.x) / map_info.resolution)
        grid_y = int((pos[1] - map_info.origin.position.y) / map_info.resolution)
        return (grid_x, grid_y)
    
    def grid_to_map(self, grid_pos, map_info):
        x = grid_pos[0] * map_info.resolution + map_info.origin.position.x + map_info.resolution / 2.0
        y = grid_pos[1] * map_info.resolution + map_info.origin.position.y + map_info.resolution / 2.0
        return (x, y)
    
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

        while open_list:
            _, current = heapq.heappop(open_list)
            if self.is_goal(current, goal):
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
                if not self.is_valid(neighbor) or self.is_obstacle(grid, neighbor):
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

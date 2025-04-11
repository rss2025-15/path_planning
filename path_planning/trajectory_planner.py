import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
import random
import heapq
import dubins
from path_planning.visualization_tools import VisualizationTools
from visualization_msgs.msg import MarkerArray, Marker

# from https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
# for some reason tf_transformations functions don't work with the numpy version installed in my docker
def euler_from_quaternion(quat):
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



class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")
        self.declare_parameter('car_length', 1.0)
        self.declare_parameter('max_steer', 0.8)
        self.declare_parameter('dilate_radius', 1)
        self.declare_parameter('sample_goal_prob', 0.05)
        self.declare_parameter('neighbor_type', "num")
        self.declare_parameter('neighbor_num', 10)
        self.declare_parameter('neighbor_radius', 2.0)
        self.declare_parameter('num_iters', 100)
        self.declare_parameter('dist_limit', False)
        self.declare_parameter('discretize_path_progress', True)
        self.declare_parameter('progress_discretization_dist', 0.5)
        self.declare_parameter('discretize_path_publish', True)
        self.declare_parameter('publish_discretization_dist', 0.5)
        self.declare_parameter('check_discretization_dist', 0.1)
        self.declare_parameter('shortcutting_iters', 100)


        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value
        self.car_length = self.get_parameter('car_length').get_parameter_value().double_value
        self.max_steer = self.get_parameter('max_steer').get_parameter_value().double_value
        self.dilate_radius = self.get_parameter('dilate_radius').get_parameter_value().integer_value
        self.sample_goal_prob = self.get_parameter('sample_goal_prob').get_parameter_value().double_value
        self.neighbor_type = self.get_parameter('neighbor_type').get_parameter_value().string_value
        self.neighbor_num = self.get_parameter('neighbor_num').get_parameter_value().integer_value
        self.neighbor_radius = self.get_parameter('neighbor_radius').get_parameter_value().double_value
        self.num_iters = self.get_parameter('num_iters').get_parameter_value().integer_value
        self.dist_limit = self.get_parameter('dist_limit').get_parameter_value().bool_value
        self.discretize_path_progress = self.get_parameter('discretize_path_progress').get_parameter_value().bool_value
        self.progress_discretization_dist = self.get_parameter('progress_discretization_dist').get_parameter_value().double_value
        self.discretize_path_publish = self.get_parameter('discretize_path_publish').get_parameter_value().bool_value
        self.publish_discretization_dist = self.get_parameter('publish_discretization_dist').get_parameter_value().double_value
        self.check_discretization_dist = self.get_parameter('check_discretization_dist').get_parameter_value().double_value
        self.shortcutting_iters = self.get_parameter('shortcutting_iters').get_parameter_value().integer_value

        self.min_turn_radius = self.car_length/np.tan(self.max_steer)
        
        self.got_map = False

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

        self.tree_viz_pub = self.create_publisher(MarkerArray, '/tree_viz', 10)
        self.try_path_viz_pub = self.create_publisher(Marker, '/try_path_viz', 10)

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        # RRT NODES
        self.path = []
        self.nodes = []
        self.parent = []
        self.children = []
        self.cost = []
        self.num_nodes = 0
        self.goal_idx = None
    
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

    def map_to_real(self, x, y):
        x = float(x)*self.resolution
        y = float(y)*self.resolution
        x_, y_, _ = self.compose_transforms(self.origin, (x,y,0))
        return (x_, y_)

    def real_to_map(self, x, y):
        x_pixel, y_pixel, _ = self.compose_transforms(self.compose_transforms(self.origin, (0,0,0)), (x, y, 0))
        return (int(x_pixel//self.resolution), int(y_pixel//self.resolution))
    
    def taken(self, x, y):
        x_map, y_map = self.real_to_map(x,y)
        # self.get_logger().info(f'checking if {(x, y)}->{(x_map, y_map)} with grid value {self.map[x_map][y_map]} is taken')
        return x_map < 0 or x_map >= self.map_width or y_map < 0 or y_map >= self.map_height or self.map[x_map][y_map] == 1

    def map_cb(self, msg):
        self.get_logger().info('GOT MAP')
        self.got_map = True

        self.map_height = msg.info.height
        self.map_width = msg.info.width
        g = np.transpose(np.reshape(msg.data, (self.map_height, self.map_width)))
        self.map = np.where(g == -1, 1, g)
        self.origin = (msg.info.origin.position.x, msg.info.origin.position.y, euler_from_quaternion(msg.info.origin.orientation)[2])
        self.resolution = msg.info.resolution

        # process grid to make not have close-cut corners and gaps in the map
        self.map = cv2.dilate(self.map.astype('uint8'), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate_radius, self.dilate_radius)))
        # plt.imshow(self.map, cmap='binary', vmin=0, vmax=1)
        # plt.show()

    def pose_cb(self, pose):
        self.init_pose = (pose.pose.pose.position.x, pose.pose.pose.position.y, euler_from_quaternion(pose.pose.pose.orientation)[2])
        self.get_logger().info(f'INITIAL POSE: {self.init_pose}')
        self.clear_rrt()

    def goal_cb(self, msg):
        self.goal = (msg.pose.position.x, msg.pose.position.y, euler_from_quaternion(msg.pose.orientation)[2])
        self.get_logger().info(f'GOAL POSE: {self.goal}')
        self.plan_path(self.init_pose, self.goal, self.map)

    def pose_sampler(self, pose, iter_num):
        p = random.random()
        if p < self.sample_goal_prob:
            # self.get_logger().info('SAMPLING GOAL')
            return self.goal
        else:
            # increasing the scope of the sampler as iterations go
            if iter_num < self.num_iters:
                coeff = float(iter_num)/self.num_iters
                pose_x, pose_y, _ = pose
                pose_x, pose_y = self.real_to_map(pose_x, pose_y)
                min_x, min_y = pose_x - coeff*self.map_width, pose_y - coeff*self.map_height
                max_x, max_y = pose_x + coeff*self.map_width, pose_y + coeff*self.map_height
                min_x = max(min_x, 0)
                min_y = max(min_y, 0)
                max_x = min(max_x, self.map_width)
                max_y = min(max_y, self.map_height)
                return self.map_to_real(min_x+random.random()*(max_x-min_x), min_y+random.random()*(max_y-min_y))+(random.random()*2*np.pi,)
            else:
                return self.map_to_real(random.random()*self.map_width, random.random()*self.map_height)+(random.random()*2*np.pi,)
        
    def clear_rrt(self):
        self.path = []
        self.nodes = []
        self.parent = []
        self.children = []
        self.cost = []
        self.num_nodes = 0
        self.goal_idx = None
    
    def add_node(self, pose, cost=-1):
        self.nodes.append(pose)
        self.cost.append(cost)
        self.parent.append(None)
        self.children.append(set())
        self.num_nodes += 1
        return self.num_nodes-1
    
    def add_edge(self, par, child):
        if self.parent[child] is not None and self.parent[child] is not par:
            self.children[self.parent[child]].remove(child)
        self.parent[child] = par
        self.children[par].add(child)

    def pose_dist(self, pose1, pose2, include_theta=False):
        x1, y1, theta1 = pose1
        x2, y2, theta2 = pose2
        if include_theta:
            return (x1-x2)**2+(y1-y2)**2+(theta1-theta2)**2
        else:
            return (x1-x2)**2+(y1-y2)**2
        
    def update_costs(self, node_id, cost_diff):
        # DFS & visualize edges
        # self.get_logger().info(f'UPDATING COST OF {node_id}')
        self.cost[node_id] += cost_diff
        for child_id in self.children[node_id]:
            self.update_costs(child_id, cost_diff)
        
    def extend_tree(self, rand_pose):
        nearest_node_idx = None
        # find nearest node
        for i in range(self.num_nodes):
            if nearest_node_idx is None or self.pose_dist(rand_pose, self.nodes[nearest_node_idx]) > self.pose_dist(rand_pose, self.nodes[i]):
                nearest_node_idx = i

        # compute dubin(/reed-sheep) trajectory from the nearest node to the random pose we have

        path = dubins.shortest_path(self.nodes[nearest_node_idx], rand_pose, self.min_turn_radius)
        steps, _ = path.sample_many(self.check_discretization_dist)
        path = [self.nodes[self.start_id]]

        # x_list, y_list, _ = zip(*steps)
        # self.try_path_viz_pub.publish(VisualizationTools.plot_line(x_list, y_list, (0., 0., 1.)))

        last_node_idx = None
        # self.get_logger().info(f'try going from {self.nodes[nearest_node_idx]} to {rand_pose} with {len(steps)} steps')

        node_indices = []

        if not self.dist_limit:
            # no distance limit, find the largest part of the circular trajectory from the neighbor to the pose
            # that we can follow without colliding
            # and discretize it with in segment of length equal to the progress distance if desired
            for step in steps[:-1]:
                x, y, theta = step
                # self.get_logger().info(f'step {step}')
                if self.taken(x, y):
                    # self.get_logger().info(f'{(x,y)} is taken')
                    break
                # self.get_logger().info('continue')
                if last_node_idx is None or (self.discretize_path_progress and self.pose_dist(self.nodes[last_node_idx], step) > self.progress_discretization_dist):
                    # self.get_logger().info(f'add node {step}')
                    last_node_idx = nearest_node_idx
                    new_idx = self.add_node(step, self.cost[last_node_idx]+dubins.shortest_path(self.nodes[last_node_idx], step, self.min_turn_radius).path_length())
                    self.add_edge(last_node_idx, new_idx)
                    node_indices.append(new_idx)
                    last_node_idx = new_idx
        else:
            # compute the part of the circular trajectory going to the node that has distance equal to the progress distance (or where we collide)
            last_step = None
            for step in steps[:-1]:
                x, y, theta = step
                if self.taken(x, y) or self.pose_dist(step, self.nodes[nearest_node_idx]) > self.progress_discretization_dist:
                    break
                last_step = step
            if last_step is None:
                return None
            new_idx = self.add_node(last_step, self.cost[nearest_node_idx]+dubins.shortest_path(self.nodes[nearest_node_idx], last_step, self.min_turn_radius).path_length())
            self.add_edge(nearest_node_idx, new_idx)
            node_indices.append(new_idx)
        
        return node_indices

        
    def rewire_tree(self, new_pose_idx):
        h = []
        for i in range(self.num_nodes):
            if i == new_pose_idx:
                continue
            if self.neighbor_type == "num":
                heapq.heappush(h, (self.pose_dist(self.nodes[new_pose_idx], self.nodes[i]), i))
                if len(h) > self.neighbor_num:
                    heapq.heappop(h)
            else:
                # find neighbors in certain radius
                if self.pose_dist(self.nodes[new_pose_idx], self.nodes[i]) < self.neighbor_radius:
                    heapq.heappush(h, (self.pose_dist(self.nodes[new_pose_idx], self.nodes[i]), i))
        
        h_cost = []
        while len(h) > 0:
            # pop items in increasing order of distance and push them in another heap to sort them with cost
            dist, node_index = heapq.heappop(h)
            heapq.heappush(h_cost, (self.cost[node_index], node_index))
        while len(h_cost) > 0:
            # pop in increasing number of cost
            cost, node_index = heapq.heappop(h_cost)
            # check if they can be used to update distances and update tree accordingly
            
            # start -> new -> old
            path = dubins.shortest_path(self.nodes[new_pose_idx], self.nodes[node_index], self.min_turn_radius)
            steps, _ = path.sample_many(self.check_discretization_dist)
            reached = True
            for step in steps[:-1]:
                x, y, theta = step
                if self.taken(x, y):
                    reached = False
            if reached and (self.cost[new_pose_idx] + path.path_length() < cost):
                cost_diff = self.cost[new_pose_idx] + path.path_length() - cost
                self.add_edge(new_pose_idx, node_index)
                self.update_costs(node_index, cost_diff)

            # start -> old -> new
            path = dubins.shortest_path(self.nodes[node_index], self.nodes[new_pose_idx], self.min_turn_radius)
            steps, _ = path.sample_many(self.check_discretization_dist)
            reached = True
            for step in steps[:-1]:
                x, y, theta = step
                if self.taken(x, y):
                    reached = False
            if reached and (cost + path.path_length() < self.cost[new_pose_idx]):
                cost_diff = cost + path.path_length() - self.cost[new_pose_idx]
                self.add_edge(node_index, new_pose_idx)
                self.update_costs(new_pose_idx, cost_diff)

    def visualize_tree(self, node_id, mark_arr):
        # DFS & visualize edges
        x_par, y_par, _ = self.nodes[node_id]
        for child_id in self.children[node_id]:
            x_child, y_child, _ = self.nodes[child_id]
            mark_arr.markers.append(VisualizationTools.plot_line([x_par, x_child], [y_par, y_child], child_id))
            self.visualize_tree(child_id, mark_arr)


    def plan_path(self, start_point, end_point, map):
        # PATH PLANNING CODE
        # RRT*

        """
        Ideas/Notes:
        Sample poses in (x,y,theta) space
        Trajectories between such poses will be a combination of two circles
        (dubins/reeds-sheep curve or something custom with varying turning radius)
        Goal bias (sampling the goal once in a while)
        Also shortcutting afterwards to remove jerky parts of the path (sample pairs of points in the path randomly or start from the end and go backwards?)
        """
        # self.goal_idx = None
        self.clear_rrt()
        self.start_id = self.add_node(start_point, 0)
        i = 0
        while (i < self.num_iters or self.goal_idx is None):
            i+=1
            # self.get_logger().info(f'ITERATION {i}')
            pose_sample = self.pose_sampler(start_point,i)
            new_idx = self.extend_tree(pose_sample)
            if new_idx is None:
                continue
            for idx in new_idx:
                self.rewire_tree(idx)
            # try to reach goal
            for idx in new_idx:
                # self.get_logger().info(f'TRYING GOAL FROM {idx}')
                if self.pose_dist(self.nodes[idx], self.goal) < self.progress_discretization_dist:
                    path = dubins.shortest_path(self.nodes[idx], self.goal, self.min_turn_radius)
                    steps, _ = path.sample_many(self.check_discretization_dist)
                    found_goal = True
                    for step in steps[:-1]:
                        x, y, theta = step
                        if self.taken(x, y):
                            found_goal = False
                    if not found_goal:
                        continue
                    self.get_logger().info('!!!!!!FOUND GOAL!!!!!!')
                    if self.goal_idx is None:
                        self.goal_idx = self.add_node(self.goal, self.cost[idx]+dubins.shortest_path(self.nodes[idx], self.goal, self.min_turn_radius).path_length())
                        self.add_edge(idx, self.goal_idx)
                    else:
                        new_cost = self.cost[idx]+dubins.shortest_path(self.nodes[idx], self.goal, self.min_turn_radius).path_length()
                        if new_cost < self.cost[self.goal_idx]:
                            self.cost[self.goal_idx] = new_cost
                            self.add_edge(idx, self.goal_idx)
            mark_arr = MarkerArray()
            # self.visualize_tree(self.start_id, mark_arr)
            # self.tree_viz_pub.publish(mark_arr)
        
        # create path going backwards with parent pointer starting from goal and up to the start
        node_id = self.goal_idx
        # self.get_logger().info(f'GOING BACKWARDS FROM GOAL: {self.goal_idx}')
        # self.path.clear()
        while node_id is not self.start_id:
            # self.get_logger().info(f'NODE {node_id}')
            self.path.append(self.nodes[node_id])
            node_id = self.parent[node_id]
        self.path.append(self.nodes[self.start_id])
        self.path.reverse()

        # Shortcutting to make paths more straightforward (include discretization if needed)
        for i in range(self.shortcutting_iters):
            u = math.floor(random.random()*len(self.path))
            v = math.floor(random.random()*(len(self.path)-1))
            start_pose = self.path[u]
            end_pose = self.path[v]
            path = dubins.shortest_path(start_pose, end_pose, self.min_turn_radius)
            steps, _ = path.sample_many(self.check_discretization_dist)
            can_shortcut = True
            for step in steps[:-1]:
                x, y, theta = step
                if self.taken(x, y):
                    can_shortcut = False
            if not can_shortcut:
                continue
            del self.path[u+1:v]
                        
        # publish trajectory
        self.trajectory.clear()
        init_x, init_y, init_theta = self.init_pose
        self.trajectory.addPoint((init_x, init_y))
        last_pose = self.init_pose
        for pose in self.path[:-1]:
            if self.discretize_path_publish:
                # discretize
                path = dubins.shortest_path(last_pose, pose, self.min_turn_radius)
                steps, _ = path.sample_many(self.publish_discretization_dist)
                for step in steps[:-1]: # ignore the first point since it's the previous one
                    x, y, theta = step
                    self.trajectory.addPoint((x,y))
            else:
                x, y, theta = pose
                self.trajectory.addPoint((x,y))
            last_pose = pose

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()

# src/rrt/rrt.py
"""
RRT (Rapidly-exploring Random Trees) path planning implementation.
- Created by: Francisco Fonseca on July 2024
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from matplotlib import pyplot as plt
from scipy.spatial import KDTree, distance
from scipy.stats import uniform
from typing import List, Tuple, Callable, Optional, Dict
from tqdm import tqdm
from threading import Thread, Lock , current_thread, main_thread # Import threading utilities
from src.boustrophedon.boustrophedon import Boustrophedon
from src.estimation.estimation import estimate_sources_bayesian
from src.rrt.rrt_utils import (
    choose_parent, cost, line_cost, obstacle_free, rewire, near, steer, 
    nearest, add_node, trace_path_to_root, node_selection_key_distance, 
    InformativeTreeNode, TreeNode, TreeCollection
)
from src.point_source.point_source import PointSourceField
import os
from matplotlib import ticker, colors

# Lock for synchronizing access to shared resources
lock = Lock()

# Helper function to calculate suppression factor
def calculate_suppression_factor(node_point: np.ndarray, source: np.ndarray, other_sources: List[np.ndarray]) -> float:
    x_t, y_t = node_point
    x_k, y_k, intensity = source
    F_src = 0
    for other_source in other_sources:
        if not np.array_equal(source, other_source):
            x_j, y_j, _ = other_source
            mid_point = [(x_k + x_j) / 2, (y_k + y_j) / 2]
            d_t_k_j = np.linalg.norm([x_t - mid_point[0], y_t - mid_point[1]])

            # Distance suppression factor
            C_src_Dist = 2 - 1 / (1 + np.exp((d_t_k_j - 2) / 16))

            # Suppression factor
            F_src += (1 - C_src_Dist + C_src_Dist / (1 + np.exp((2 - d_t_k_j) / 16)))
    return F_src

# Gain Calculation Functions

def point_source_gain_no_penalties(self, node: InformativeTreeNode, agent_idx: int) -> float:
    def sources_gain(node: InformativeTreeNode) -> float:
        x_t, y_t = node.point
        point_source_gain = 0
        other_sources = [s for s in self.best_estimates]
        if hasattr(self, 'best_estimates') and self.best_estimates.size > 0:
            closest_source = min(self.best_estimates, key=lambda source: np.linalg.norm([self.agents_full_path[agent_idx][-1][0] - source[0], self.agents_full_path[agent_idx][-1][1] - source[1]]))
            d_src = np.linalg.norm([x_t - closest_source[0], y_t - closest_source[1]])
            F_src = calculate_suppression_factor(node.point, closest_source, other_sources)
            point_source_gain += (1 + np.exp(-(d_src - 2)**2 / 2*16)) * F_src
            return point_source_gain
        else:
            return 0

    final_gain = 0
    current_node = node
    while current_node.parent:
        final_gain += sources_gain(current_node)
        current_node = current_node.parent
    return max(final_gain, 0)

def point_source_gain_only_distance_penalty(self, node: InformativeTreeNode, agent_idx: int) -> float:
    def sources_gain(node: InformativeTreeNode) -> float:
        x_t, y_t = node.point
        point_source_gain = 0
        other_sources = [s for s in self.best_estimates]
        if hasattr(self, 'best_estimates') and self.best_estimates.size > 0:
            for source in self.best_estimates:
                d_src = np.linalg.norm([x_t - source[0], y_t - source[1]])
                F_src = calculate_suppression_factor(node.point, source, other_sources)
                point_source_gain += (1 + np.exp(-(d_src - 2)**2 / 2*16)) * F_src
            return point_source_gain * distance_penalty(node)
        else:
            return 0

    def distance_penalty(node: InformativeTreeNode) -> float:
        if node.parent:
            return np.exp(0.5 * np.linalg.norm(node.point - node.parent.point))
        return 1

    final_gain = 0
    current_node = node
    while current_node.parent:
        final_gain += sources_gain(current_node)
        current_node = current_node.parent
    return max(final_gain, 0)

def point_source_gain_distance_rotation_penalty(self, node: InformativeTreeNode, agent_idx: int) -> float:
    def sources_gain(node: InformativeTreeNode) -> float:
        x_t, y_t = node.point
        point_source_gain = 0
        other_sources = [s for s in self.best_estimates]
        if hasattr(self, 'best_estimates') and self.best_estimates.size > 0:
            for source in self.best_estimates:
                d_src = np.linalg.norm([x_t - source[0], y_t - source[1]])
                F_src = calculate_suppression_factor(node.point, source, other_sources)
                point_source_gain += (1 + np.exp(-(d_src - 2)**2 / 2*16)) * F_src
            return point_source_gain * distance_penalty(node) * rotation_penalty(node)
        else:
            return 0

    def distance_penalty(node: InformativeTreeNode) -> float:
        if node.parent:
            return np.exp(0.5 * np.linalg.norm(node.point - node.parent.point))
        return 1

    def rotation_penalty(node: InformativeTreeNode) -> float:
        if node.parent:
            theta_t = np.arctan2(node.point[1] - node.parent.point[1], node.point[0] - node.parent.point[0])
            return np.exp(0.05*(theta_t**2) / 0.1)
        return 1

    final_gain = 0
    current_node = node
    while current_node.parent:
        final_gain += sources_gain(current_node)
        current_node = current_node.parent
    return max(final_gain, 0)

def point_source_gain_all(self, node: InformativeTreeNode, agent_idx: int) -> float:
    # generate kd trees for each agent
    trees_obs_wp = []
    for i, obs_wp in enumerate(self.agents_obs_wp):
        if i != agent_idx:
            if len(obs_wp) > 0:
                trees_obs_wp.append(KDTree(obs_wp))
            else:
                trees_obs_wp.append(None)
        else:
            trees_obs_wp.append(None)

    def sources_gain(node: InformativeTreeNode) -> float:
        x_t, y_t = node.point
        point_source_gain = 0
        other_sources = [s for s in self.best_estimates]
        
                    
        if hasattr(self, 'best_estimates') and self.best_estimates.size > 0:
            for source in self.best_estimates:
                d_src = np.linalg.norm([x_t - source[0], y_t - source[1]])
                F_src = calculate_suppression_factor(node.point, source, other_sources)
                point_source_gain += (np.exp(-(d_src - 2)**2 / (32))) * F_src

            distance_penalty_val = distance_penalty(node)
            exploitation_penalty_val = exploitation_penalty(node)
            workspace_penalty_val = workspace_penalty(self, node, node.parent)
            return point_source_gain - distance_penalty_val - exploitation_penalty_val - workspace_penalty_val
        else:
            workspace_penalty_val = workspace_penalty(self, node, node.parent)
            return -workspace_penalty_val

    def distance_penalty(node: InformativeTreeNode) -> float:
        if node.parent:
            return np.exp(0.5 * np.linalg.norm(node.point - node.parent.point)**2)
        return 0

    def workspace_penalty(self, node: InformativeTreeNode, parent_node: Optional[InformativeTreeNode] = None) -> float:
        """
        Calculate a penalty for nodes outside the workspace or inside obstacles.
        
        Parameters:
        - node: The node to evaluate.
        - parent_node: The parent node of the current node, if applicable.
        
        Returns:
        - float: np.inf if the node is outside the workspace or inside an obstacle; 0 otherwise.
        """
        # Check if the node is within the workspace boundaries
        if node.point[0] < self.scenario.workspace_size[0] or node.point[0] > self.scenario.workspace_size[1] or \
        node.point[1] < self.scenario.workspace_size[2] or node.point[1] > self.scenario.workspace_size[3]:
            # print(f"Node {node.point} is outside workspace boundaries.")
            return np.inf

        # Check if the node is inside an obstacle
        for obstacle in self.scenario.obstacles:
            if obstacle["type"] == "rectangle":
                x_min, y_min = obstacle["x"], obstacle["y"]
                x_max, y_max = x_min + obstacle["width"], y_min + obstacle["height"]
                
                if x_min <= node.point[0] <= x_max and y_min <= node.point[1] <= y_max:
                    # print(f"Node {node.point} is inside an obstacle.")
                    return np.inf

        # If a parent node is provided, check if the path between the parent node and the current node intersects any obstacles
        def obstacle_free(self, point1: np.ndarray, point2: np.ndarray) -> bool:
            """Check if the line segment between point1 and point2 is free of obstacles."""
            for obstacle in self.scenario.obstacles:
                if line_intersects_obstacle(self, point1, point2, obstacle):
                    print(f"Path from {point1} to {point2} intersects an obstacle.")
                    return False
            return True

        def line_intersects_obstacle(self, point1: np.ndarray, point2: np.ndarray, obstacle: Dict) -> bool:
            """Check if a line between two points intersects a given obstacle."""
            x1, y1 = point1
            x2, y2 = point2

            if obstacle['type'] == 'rectangle':
                x_min, y_min = obstacle['x'], obstacle['y']
                x_max = x_min + obstacle['width']
                y_max = y_min + obstacle['height']

                # Check for intersections with all four sides of the rectangle
                return (line_intersects_line(x1, y1, x2, y2, x_min, y_min, x_min, y_max) or
                        line_intersects_line(x1, y1, x2, y2, x_max, y_min, x_max, y_max) or
                        line_intersects_line(x1, y1, x2, y2, x_min, y_min, x_max, y_min) or
                        line_intersects_line(x1, y1, x2, y2, x_min, y_max, x_max, y_max))

            return False

        def line_intersects_line(x1, y1, x2, y2, x3, y3, x4, y4):
            """Check if the line segment from (x1, y1) to (x2, y2) intersects with the line segment from (x3, y3) to (x4, y4)."""
            # Calculate direction of the lines
            d1 = (x2 - x1, y2 - y1)
            d2 = (x4 - x3, y4 - y3)
            
            # Solve for intersection
            denominator = d1[0] * d2[1] - d1[1] * d2[0]
            if denominator == 0:
                return False  # Lines are parallel
            
            # Calculate intersection point
            t = ((x3 - x1) * d2[1] - (y3 - y1) * d2[0]) / denominator
            u = ((x3 - x1) * d1[1] - (y3 - y1) * d1[0]) / denominator
            
            # Check if intersection point is on both segments
            return 0 <= t <= 1 and 0 <= u <= 1
        if parent_node:
            if not obstacle_free(self, parent_node.point, node.point):
                # print(f"Path from {parent_node.point} to {node.point} intersects an obstacle.")
                return np.inf
        return 0



    def exploitation_penalty(node: InformativeTreeNode) -> float:
        if len(self.agents_obs_wp[agent_idx]) > 0:
            n_obs_wp = 0
            for i in range(len(self.agent_positions)):
                n_obs_wp += len([obs_wp for obs_wp in self.agents_obs_wp[i] if np.linalg.norm(node.point - obs_wp) < self.d_waypoint_distance])
            return np.exp(0.05 * n_obs_wp**2)
        return 0

    final_gain = 0
    current_node = node
    while current_node.parent:
        final_gain += sources_gain(current_node)
        current_node = current_node.parent
    return final_gain

# Tree Generation Functions

def rrt_tree_generation(self, budget_portion: float, agent_idx: int) -> None:
    distance_travelled = 0
    while distance_travelled < budget_portion:
        random_point = np.random.rand(2) * (self.scenario.workspace_size[1] - self.scenario.workspace_size[0], self.scenario.workspace_size[3] - self.scenario.workspace_size[2]) + (self.scenario.workspace_size[0], self.scenario.workspace_size[2])
        nearest_node = min(self.tree_nodes[agent_idx], key=lambda node: node_selection_key_distance(node, random_point))
        new_point = steer(nearest_node, random_point, d_max_step=self.d_waypoint_distance)

        if obstacle_free(nearest_node.point, new_point):
            new_node = InformativeTreeNode(new_point, nearest_node)
            add_node(self.tree_nodes[agent_idx], new_node, nearest_node)
            distance_travelled += np.linalg.norm(new_node.point - nearest_node.point)
            new_node.information = point_source_gain_no_penalties(self, new_node, agent_idx)
            if distance_travelled >= budget_portion:
                break

        self.agents_trees[agent_idx].add(self.agents_roots[agent_idx])

def rrt_star_tree_generation(self, budget_portion: float, agent_idx: int) -> None:
    distance_travelled = 0
    while distance_travelled < budget_portion:
        x_rand = np.random.rand(2) * (self.scenario.workspace_size[1] - self.scenario.workspace_size[0], self.scenario.workspace_size[3] - self.scenario.workspace_size[2]) + (self.scenario.workspace_size[0], self.scenario.workspace_size[2])
        x_nearest = nearest(self.tree_nodes[agent_idx], x_rand)
        x_new = steer(x_nearest, x_rand, d_max_step=1.0)
        
        if obstacle_free(x_nearest.point, x_new):
            x_new = InformativeTreeNode(x_new)
            X_near = near(x_new, self.tree_nodes[agent_idx], self.d_waypoint_distance)
            
            c_min = cost(x_nearest) + line_cost(x_nearest, x_new)
            x_min = x_nearest
            
            for x_near in X_near:
                if obstacle_free(x_near.point, x_new) and cost(x_near) + line_cost(x_near, x_new) < c_min:
                    c_min = cost(x_near) + line_cost(x_near, x_new)
                    x_min = x_near
            
            add_node(self.tree_nodes[agent_idx], x_new, x_min)
            distance_travelled += np.linalg.norm(x_new.point - x_min.point)
            
            rewire(X_near, x_new)
            if distance_travelled >= budget_portion:
                break
    self.agents_trees[agent_idx].add(self.agents_roots[agent_idx])

def rig_tree_generation(self, budget_portion: float, agent_idx: int, gain_function: Callable) -> None:
    distance_travelled = 0
    while distance_travelled < budget_portion:
        random_point = np.random.rand(2) * (self.scenario.workspace_size[1] - self.scenario.workspace_size[0], self.scenario.workspace_size[3] - self.scenario.workspace_size[2]) + (self.scenario.workspace_size[0], self.scenario.workspace_size[2])
        nearest_node = nearest(self.tree_nodes[agent_idx], random_point)
        new_point = steer(nearest_node, random_point, d_max_step=self.d_waypoint_distance)

        if obstacle_free(nearest_node.point, new_point):
            new_node = InformativeTreeNode(new_point)    
            X_near = near(new_node, self.tree_nodes[agent_idx], self.d_waypoint_distance)
            new_node = choose_parent(X_near, nearest_node, new_node)
            add_node(self.tree_nodes[agent_idx], new_node, nearest_node)
            distance_travelled += np.linalg.norm(new_node.point - nearest_node.point)
            
            new_node.information = gain_function(self, new_node, agent_idx)

            rewire(X_near, new_node)
            if distance_travelled >= budget_portion:
                break

        self.agents_trees[agent_idx].add(self.agents_roots[agent_idx])

# Information Update Functions

def gp_information_update(self) -> None:
    self.measurements = sum((agent_measurements for agent_measurements in self.agents_measurements), [])
    self.obs_wp = sum((agent_wp for agent_wp in self.agents_obs_wp), [])
    self.full_path = sum((agent_path for agent_path in self.agents_full_path), [])
    if len(self.measurements) > 0 and len(self.measurements) % 5 == 0:
        min_len = min(len(self.measurements), len(self.obs_wp))
        self.measurements = self.measurements[:min_len]
        self.obs_wp = self.obs_wp[:min_len]
        self.scenario.update(self.obs_wp, np.log10(np.maximum(self.measurements, 1e-6)))
        if hasattr(self, 'best_estimates') and self.best_estimates.size > 0:
            estimates, _, bic = estimate_sources_bayesian(
                self.obs_wp, self.measurements, self.lambda_b,
                self.max_sources, self.n_samples, self.s_stages, self.scenario
            )
            if bic > self.best_bic:
                # print(f"\nEstimated {len(estimates)//3} sources: {estimates}")
                # print(f"\nBEST BIC FROM GP: {bic}")
                self.best_bic = bic
                self.best_estimates = estimates.reshape((-1, 3))

def source_metric_information_update(self) -> None:
    # print("\nUpdating information\n")
    self.measurements = sum((agent_measurements for agent_measurements in self.agents_measurements), [])
    self.obs_wp = sum((agent_wp for agent_wp in self.agents_obs_wp), [])
    self.full_path = sum((agent_path for agent_path in self.agents_full_path), [])
    if  len(self.measurements) > 0 and len(self.measurements) % 5 == 0:
        min_len = min(len(self.measurements), len(self.obs_wp))
        self.measurements = self.measurements[:min_len]
        self.obs_wp = self.obs_wp[:min_len]
        self.scenario.update(self.obs_wp, np.log10(np.maximum(self.measurements, 1e-6)))
        
        estimates, _, bic = estimate_sources_bayesian(
            self.obs_wp, self.measurements, self.lambda_b,
            self.max_sources, self.n_samples, self.s_stages, self.scenario
        )
        # print(f"\nEstimated {len(estimates)//3} sources: {estimates}")
        # print(f"\nBEST BIC: {bic}")
        self.best_bic = bic
        self.best_estimates = estimates.reshape((-1, 3))

# Path Selection Functions

def random_path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None) -> List[np.ndarray]:
    leaf_nodes = [node for node in self.tree_nodes[agent_idx] if not node.children]
    if not leaf_nodes:
        return []
    selected_leaf = np.random.choice(leaf_nodes)
    return trace_path_to_root(selected_leaf)

def informative_source_metric_path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None, current_budget: Optional[float] = None) -> List[np.ndarray]: 
    # Get current root 
    current_root = self.agents_roots[agent_idx]
    if not current_root:
        return []

    current_leafs = []
    
    # From the root we can get the leaf nodes 
    def get_leafs(node):
        if not node.children:
            current_leafs.append(node)
        for child in node.children:
            get_leafs(child)

    get_leafs(current_root)

    if not current_leafs:
        return []

    # Remove the current position from the list of leaf nodes if it exists
    if current_position is not None:
        current_leafs = [node for node in current_leafs if not np.array_equal(node.point, current_position)]
        if not current_leafs:
            return [current_position]

    if self.best_estimates.size == 0:
        selected_node = np.random.choice(current_leafs)
        # make sure information is not -inf
        while selected_node.information == -np.inf:
            selected_node = np.random.choice(current_leafs)
    else:
        # choose the node with highest information gain
        selected_node = max(current_leafs, key=lambda node: node.information)
    # print(f"Selected node information: {selected_node.information}")

    possible_path = trace_path_to_root(selected_node)
    
    # Ensure agent does not get stuck in the same position
    if len(possible_path) == 1 and np.array_equal(possible_path[0], current_position):
        current_leafs = [node for node in current_leafs if not np.array_equal(node.point, selected_node.point)]
        if current_leafs:
            selected_node = np.random.choice(current_leafs)
            possible_path = trace_path_to_root(selected_node)

    # Make sure there is enough budget to do the path, if not stop where the budget ends
    if current_budget is not None and current_position is not None:
        path = []
        for node in possible_path:
            if current_budget - np.linalg.norm(node - current_position) < self.d_waypoint_distance:
                path.append(node)
                break
            path.append(node)
            #self.information_update() 
            current_budget -= np.linalg.norm(node - current_position) if not path else np.linalg.norm(node - path[-1])
        return path
    return possible_path

def bias_beta_path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None, current_budget: Optional[float] = None) -> List[np.ndarray]:
    """
    Selects a path for the agent based on mutual information gain.
    """
    # Find leaf nodes in the tree for the current agent
    leaf_nodes = [node for node in self.tree_nodes[agent_idx] if not node.children]
    leaf_points = np.array([node.point for node in leaf_nodes])

    # Compute the covariance matrix for leaf points (K_pi)
    K_pi = self.scenario.gp.kernel(leaf_points)
    
    # Compute the cross-covariance matrix between leaf points and observation points (K_pio)
    observed_points = np.array(self.agents_obs_wp[agent_idx])
    if observed_points.size > 0:
        K_pio = self.scenario.gp.kernel(leaf_points, observed_points)
    else:
        K_pio = np.zeros((len(leaf_points), 0))  # No observations yet

    # Calculate mutual information for each leaf point
    mutual_information_values = np.array([
        mutual_information_gain(K_pi, K_pio, beta_t=self.beta_t) for _ in leaf_points
    ])
    def distance_penalty(node: InformativeTreeNode) -> float:
        if node.parent:
            return np.exp(0.5 * np.linalg.norm(node.point - node.parent.point)**2)
        return 0

    def workspace_penalty(node: InformativeTreeNode) -> float:
        # if inside workspace 0, if not np.inf
        if node.point[0] < self.scenario.workspace_size[0] or node.point[0] > self.scenario.workspace_size[1] or node.point[1] < self.scenario.workspace_size[2] or node.point[1] > self.scenario.workspace_size[3]:
            return np.inf
        return 0
    
    def exploitation_penalty(node: InformativeTreeNode) -> float:
        if len(self.agents_obs_wp[agent_idx]) > 0:
            n_obs_wp = 0
            for i in range(len(self.agent_positions)):
                n_obs_wp += len([obs_wp for obs_wp in self.agents_obs_wp[i] if np.linalg.norm(node.point - obs_wp) < self.d_waypoint_distance])
            return np.exp(0.05 * n_obs_wp**2)
        return 0
    # apply to the values the penalty functions of distance, workspace and exploitation
    for i, node in enumerate(leaf_nodes):
        mutual_information_values[i] -= distance_penalty(node)
        mutual_information_values[i] -= workspace_penalty(node)
        mutual_information_values[i] -= exploitation_penalty(node)

    
    # Select the leaf node that maximizes the mutual information
    max_mi_idx = np.argmax(mutual_information_values)
    selected_leaf = leaf_nodes[max_mi_idx]
    
    # Trace back the path to the root for the selected leaf
    possible_path = trace_path_to_root(selected_leaf)
    
    # Ensure agent does not get stuck in the same position
    if len(possible_path) == 1 and np.array_equal(possible_path[0], current_position):
        current_leafs = [node for node in current_leafs if not np.array_equal(node.point, selected_node.point)]
        if current_leafs:
            selected_node = np.random.choice(current_leafs)
            possible_path = trace_path_to_root(selected_node)

    # Make sure there is enough budget to do the path, if not stop where the budget ends
    if current_budget is not None and current_position is not None:
        path = []
        for node in possible_path:
            if current_budget - np.linalg.norm(node - current_position) < self.d_waypoint_distance:
                path.append(node)
                break
            path.append(node)
            #self.information_update() 
            current_budget -= np.linalg.norm(node - current_position) if not path else np.linalg.norm(node - path[-1])
        return path
    return possible_path

def mutual_information_gain(K_pi, K_pio, beta_t: float) -> float:
    """
    Calculate the mutual information gain using a modified version that includes a weighting factor (beta_t).
    
    K_pi: Covariance matrix for the path points.
    K_pio: Cross-covariance matrix between path points and observation points.
    beta_t: Weighting factor to control exploration vs. exploitation.
    """
    # Handle the case when there are no observed points
    if K_pio.shape[1] == 0:
        return 0  # No mutual information to gain from unobserved points
    
    # Modified covariance matrix to include exploration-exploitation factor
    modified_covariance = np.eye(K_pi.shape[0]) + beta_t * np.dot(K_pio, K_pio.T)
    return 0.5 * np.log(np.linalg.det(modified_covariance))


# Base Class and Specific Implementations

class InformativeRRTBaseClass:
    def __init__(self, scenario: PointSourceField, beta_t: float = 5.0, budget: float = 375, d_waypoint_distance: float = 2.5, num_agents: int = 1,
                 n_samples: int = 25, s_stages: int = 10, lambda_b: float = 1, max_sources: int = 3, budget_iter: int = 10, stage_lambda: float = 0.875, **kwargs):
        self.scenario = scenario
        self.beta_t = beta_t
        self.budget = [budget] * num_agents
        self.d_waypoint_distance = d_waypoint_distance
        self.num_agents = num_agents
        self.budget_iter = budget_iter
        self.name = None
        self.trees = TreeCollection()
        self.uncertainty_reduction = []
        self.time_taken = 0
        self.new_scenario = None
        self.Z_pred = None
        self.best_bic = -np.inf

        self.n_samples = n_samples
        self.s_stages = s_stages
        self.lambda_b = lambda_b
        self.max_sources = max_sources
        self.assignments = [-1] * num_agents  # Initialize assignments
        self.stage_lambda = stage_lambda

        if self.num_agents > 1:
            self.agent_positions = [np.array([self.scenario.workspace_size[0] + 0.5 + i * (self.scenario.workspace_size[1] - 1) / (num_agents - 1), self.scenario.workspace_size[2] + 0.5]) for i in range(num_agents)]
        else:
            self.agent_positions = [np.array([0.5, 0.5])]
        self.agents_trees = [TreeCollection() for _ in range(num_agents)]
        self.agents_obs_wp = [[] for _ in range(num_agents)]
        self.agents_measurements = [[] for _ in range(num_agents)]
        self.agents_full_path = [[] for _ in range(num_agents)]
        self.tree_nodes = [[] for _ in range(num_agents)]
        self.agents_roots = [None] * num_agents

    def assign_agents_to_sources(self):
        if self.best_estimates.size == 0:
            return
        # print("\nAssigning agents to sources\n")
        source_positions = [source[:2] for source in self.best_estimates]
        agent_positions = self.agent_positions

        num_sources = len(source_positions)
        num_agents = len(agent_positions)

        if num_sources >= num_agents:
            # If there are more sources or equal sources than agents, each agent gets the closest source
            self.assignments = np.argmin(distance.cdist(agent_positions, source_positions), axis=1)
        else:
            # If there are more agents than sources, closest agents get sources, and others explore
            dist_matrix = distance.cdist(agent_positions, source_positions)
            sorted_indices = np.argsort(dist_matrix, axis=1)
            assigned_sources = set()
            self.assignments = [-1] * num_agents  # -1 means agent is exploring

            for agent_idx in range(num_agents):
                for source_idx in sorted_indices[agent_idx]:
                    if source_idx not in assigned_sources:
                        self.assignments[agent_idx] = source_idx
                        assigned_sources.add(source_idx)
                        break

            # # Agents not assigned to any source will explore
            # for agent_idx in range(num_agents):
            #     if self.assignments[agent_idx] == -1:
            #         print(f"Agent {agent_idx} is exploring")
            #     else:
            #         print(f"Agent {agent_idx} is assigned to source {self.assignments[agent_idx]}")

    def initialize_trees(self, start_position: np.ndarray, agent_idx: int) -> None:
        self.agents_roots[agent_idx] = InformativeTreeNode(start_position)
        self.tree_nodes[agent_idx] = [self.agents_roots[agent_idx]]
        self.trees.add(self.agents_roots[agent_idx])

    def plot_current_state(self, iteration: int, save_dir: str) -> None:
        if current_thread() is main_thread():
            pass
        else:
            return
        fig, axs = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)

        max_log_value = np.ceil(np.log10(np.max(self.scenario.g_truth))) if np.max(self.scenario.g_truth) != 0 else 1
        levels = np.logspace(0, max_log_value, int(max_log_value) + 1)
        cmap = plt.get_cmap('Greens_r', len(levels) - 1)

        # Plot ground truth
        cs_true = axs[0].contourf(self.scenario.X, self.scenario.Y, self.scenario.g_truth, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
        fig.colorbar(cs_true, ax=axs[0], format=ticker.LogFormatterMathtext())
        axs[0].set_title('Ground Truth')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].set_facecolor(cmap(0))

        # Plot predicted field
        cs_pred = axs[1].contourf(self.scenario.X, self.scenario.Y, self.Z_pred, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
        fig.colorbar(cs_pred, ax=axs[1], format=ticker.LogFormatterMathtext())
        axs[1].set_title('Predicted Field')

        # Plot paths and waypoints
        colors_path = ['b', 'c', 'm', 'y', 'k', 'w']
        for i in range(self.num_agents):
            current_path = np.array(self.agents_full_path[i]).reshape(-1, 2)
            if current_path.size > 0:
                axs[1].plot(current_path[:, 0], current_path[:, 1], label=f'Agent {i+1} Path', color=colors_path[i % len(colors_path)], linewidth=1)
            current_wp = np.array(self.agents_obs_wp[i]).reshape(-1, 2)
            if current_wp.size > 0:
                axs[1].plot(current_wp[:, 0], current_wp[:, 1], 'ro', markersize=2)

        # Plot sources
        for source in self.scenario.sources:
            axs[1].plot(source[0], source[1], 'rX', markersize=10, label='Source')

        if hasattr(self, 'best_estimates'):
            for est_source in self.best_estimates:
                axs[1].plot(est_source[0], est_source[1], 'yx', markersize=10, label='Estimated Source')

        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.suptitle(f'Iteration {iteration}', fontsize=16)

        # Save plot to file with unique name
        plt.savefig(os.path.join(save_dir, f'iteration_{iteration}.png'))
        plt.close(fig)
    
    # Thread function for each agent
    def agent_thread(self, agent_idx: int, budget_portion: float, max_budget: float) -> None:
        # while self.budget[agent_idx] > 0:
        current_budget = budget_portion
        while current_budget > self.d_waypoint_distance and self.budget[agent_idx] > 0:
            self.tree_generation(max_budget, agent_idx)
            budget_factor = self.budget[agent_idx] / max_budget
            if budget_factor > self.stage_lambda:
                path = bias_beta_path_selection(self, agent_idx, self.agents_full_path[agent_idx][-1] if self.agents_full_path[agent_idx] else None, current_budget)
            else:
                path = self.path_selection(agent_idx, self.agents_full_path[agent_idx][-1] if self.agents_full_path[agent_idx] else None, current_budget)

            with lock:
                budget_spent = self.calculate_budget_spent(path)
                self.update_observations_and_model(path, agent_idx)
                self.agents_full_path[agent_idx].extend(path)
                self.budget[agent_idx] -= budget_spent
                current_budget -= budget_spent
                if len(self.agents_obs_wp[agent_idx % self.num_agents]) > 0:
                    self.information_update() if budget_factor <= self.stage_lambda else gp_information_update(self)

            if len(path) > 0 and self.budget[agent_idx] > 0:
                self.initialize_trees(path[-1], agent_idx)
    # Main run function to manage threads
    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        budget_portion = [budget / self.budget_iter for budget in self.budget]
        max_budget = max(self.budget)
        for i in range(self.num_agents):
            self.initialize_trees(self.agent_positions[i], i)

        start_time = time.time()
        # save_dir = './images/' + self.name + '/step_by_step/' + str(int(time.time())) + '/'

        # # Create the directory if it does not exist
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # # Always clean existing images
        # for file in os.listdir(save_dir):
        #     os.remove(os.path.join(save_dir, file))

        iteration = 0
        with tqdm(total=sum(self.budget), desc="Running " + str(self.num_agents) + " Agent " + self.name) as pbar:
            while any(b > 0 for b in self.budget):
                self.assign_agents_to_sources()

                # Clear previous threads
                threads = []

                # Create and start threads for each agent
                for i in range(self.num_agents):
                    if self.budget[i] > 0:
                        t = Thread(target=self.agent_thread, args=(i, budget_portion[i], max_budget))
                        threads.append(t)
                        t.start()

                # Join threads to ensure all agents complete their steps before updating the global state
                for t in threads:
                    t.join()

                # Plot current state after all threads have completed their iteration
                # self.plot_current_state(iteration, save_dir)
                iteration += 1

                # Update progress bar after each iteration
                pbar.update(sum(budget_portion))

        self.time_taken = time.time() - start_time
        return self.finalize_all_agents()
    def update_observations_and_model(self, path: List[np.ndarray], agent_idx: int) -> None:
        if path:
            new_measurements = []
            new_obs_wp = []
            
            for point in path:
                # Simulate measurement at the given point
                measurement = self.scenario.simulate_measurements([point])[0]
                
                # Append to temporary lists to ensure synchronization
                new_measurements.append(measurement)
                new_obs_wp.append(point)

            # Update agent-specific lists with synchronized data
            self.agents_measurements[agent_idx].extend(new_measurements)
            self.agents_obs_wp[agent_idx].extend(new_obs_wp)

            # Update global state only after all updates
            self.obs_wp = sum((agent_wp for agent_wp in self.agents_obs_wp), [])
            self.measurements = sum((agent_measurements for agent_measurements in self.agents_measurements), [])
            self.full_path = sum((agent_path for agent_path in self.agents_full_path), [])

            # Ensure the data is in the correct format
            self.obs_wp = np.array(self.obs_wp)
            self.full_path = np.array(self.full_path).reshape(-1, 2).T

            # Update the predicted spatial field based on the new observations
            self.Z_pred, std = self.scenario.predict_spatial_field(self.obs_wp, np.array(self.measurements))

    def calculate_budget_spent(self, path: List[np.ndarray]) -> float:
        if not path:
            return 0
        path_array = np.array(path)
        if path_array.ndim > 1:
            return np.sum(np.linalg.norm(np.diff(path_array, axis=0), axis=1))
        return 0
    
    def finalize_all_agents(self) -> Tuple[np.ndarray, np.ndarray]:
        self.measurements = sum((agent_measurements for agent_measurements in self.agents_measurements), [])
        self.obs_wp = sum((agent_wp for agent_wp in self.agents_obs_wp), [])
        self.full_path = sum((agent_path for agent_path in self.agents_full_path), [])
        self.obs_wp = np.array(self.obs_wp)
        self.full_path = np.array(self.full_path).reshape(-1, 2).T

        self.Z_pred, std = self.scenario.predict_spatial_field(self.obs_wp, np.array(self.measurements))
        
        return self.Z_pred, std
    
    def tree_generation(self, budget_portion: float, agent_idx: int) -> None:
        raise NotImplementedError("tree_generation method must be implemented in the subclass")
    
    def path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None) -> List[np.ndarray]:
        raise NotImplementedError("path_selection method must be implemented in the subclass")

    def information_update(self) -> None:
        raise NotImplementedError("information_update method must be implemented in the subclass")
    
    def get_current_node(self, agent_idx: int):
        """Returns the current node for visualization purposes."""
        if self.tree_nodes[agent_idx]:
            return self.tree_nodes[agent_idx][-1]  # Assuming the last node added is the current node
        return None

    def get_chosen_branch(self, agent_idx: int):
        """Returns the chosen branch for visualization purposes."""
        if not self.tree_nodes[agent_idx]:
            return []
        
        current_node = self.get_current_node(agent_idx)
        if current_node is None:
            return []

        branch = []
        while current_node is not None:
            branch.append(current_node.point)
            current_node = current_node.parent
        branch.reverse()
        return branch

class RRT_Random_GP_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "RRT_Random_GP_Path"

    def tree_generation(self, budget_portion: float, agent_idx: int) -> None:
        rrt_tree_generation(self, budget_portion, agent_idx)

    def path_selection(self, agent_idx: int) -> List[np.ndarray]:
        return random_path_selection(self, agent_idx)

    def information_update(self) -> None:
        gp_information_update(self)

class RRTStar_Random_GP_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "RRTStar_Random_GP_Path"

    def tree_generation(self, budget_portion: float, agent_idx: int) -> None:
        rrt_star_tree_generation(self, budget_portion, agent_idx)

    def path_selection(self, agent_idx: int) -> List[np.ndarray]:
        return random_path_selection(self, agent_idx)

    def information_update(self) -> None:
        gp_information_update(self)

class RRTRIG_PointSourceInformative_SourceMetric_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_estimates = np.array([])
        
        self.name = "RRTRIG_PointSourceInformative_SourceMetric_Path"

    def tree_generation(self, budget_portion: float, agent_idx: int) -> None:
        rig_tree_generation(self, budget_portion, agent_idx, gain_function=point_source_gain_no_penalties)

    def path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None, current_budget: Optional[float] = None) -> List[np.ndarray]:
        return informative_source_metric_path_selection(self, agent_idx, current_position, current_budget)

    def information_update(self) -> None:
        source_metric_information_update(self)

class RRTRIG_PointSourceInformative_Distance_SourceMetric_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_estimates = np.array([])
        
        self.name = "RRTRIG_PointSourceInformative_Distance_SourceMetric_Path"

    def tree_generation(self, budget_portion: float, agent_idx: int) -> None:
        rig_tree_generation(self, budget_portion, agent_idx, gain_function=point_source_gain_only_distance_penalty)

    def path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None, current_budget: Optional[float] = None) -> List[np.ndarray]:
        return informative_source_metric_path_selection(self, agent_idx, current_position, current_budget)
    
    def information_update(self) -> None:
        source_metric_information_update(self)

class RRTRIG_PointSourceInformative_DistanceRotation_SourceMetric_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_estimates = np.array([])
        
        self.name = "RRTRIG_PointSourceInformative_DistanceRotation_SourceMetric_Path"

    def tree_generation(self, budget_portion: float, agent_idx: int) -> None:
        rig_tree_generation(self, budget_portion, agent_idx, gain_function=point_source_gain_distance_rotation_penalty)

    def path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None, current_budget: Optional[float] = None) -> List[np.ndarray]:
        return informative_source_metric_path_selection(self, agent_idx, current_position, current_budget)

    def information_update(self) -> None:
        source_metric_information_update(self)

class MR_IPP(InformativeRRTBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_estimates = np.array([])
        
        self.name = "MR_IPP"

    def tree_generation(self, budget_portion: float, agent_idx: int) -> None:
        rig_tree_generation(self, budget_portion, agent_idx, gain_function=point_source_gain_all)

    def path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None, current_budget: Optional[float] = None) -> List[np.ndarray]:
        return informative_source_metric_path_selection(self, agent_idx, current_position, current_budget)

    def information_update(self) -> None:
        source_metric_information_update(self)

class RRT_BiasBetaInformative_GP_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, directional_bias: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.directional_bias = directional_bias
        self.name = "RRT_BiasBetaInformative_GP_Path"

    def path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None, current_budget: Optional[float] = None) -> List[np.ndarray]:
        return bias_beta_path_selection(self, agent_idx, current_position)

    def tree_generation(self, budget_portion: float, agent_idx: int) -> None:
        rrt_tree_generation(self, budget_portion, agent_idx)

    def information_update(self) -> None:
        gp_information_update(self)

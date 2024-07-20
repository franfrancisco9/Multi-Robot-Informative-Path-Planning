# src/rrt/rrt.py
"""
RRT (Rapidly-exploring Random Trees) path planning implementation.
- Created by: Francisco Fonseca on July 2024
"""

import numpy as np
import time
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import uniform
from typing import List, Tuple, Callable, Optional
from tqdm import tqdm
from src.boustrophedon.boustrophedon import Boustrophedon
from src.estimation.estimation import estimate_sources_bayesian
from src.rrt.rrt_utils import (
    choose_parent, cost, line_cost, obstacle_free, rewire, near, steer, 
    nearest, add_node, trace_path_to_root, node_selection_key_distance, 
    InformativeTreeNode, TreeNode, TreeCollection
)
from src.point_source.point_source import PointSourceField

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
            for source in self.best_estimates:
                d_src = np.linalg.norm([x_t - source[0], y_t - source[1]])
                F_src = calculate_suppression_factor(node.point, source, other_sources)
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
                point_source_gain += (1 + np.exp(-(d_src - 2)**2 / (2 * 16))) * F_src
            distance_penalty_val = distance_penalty(node)
            rotation_penalty_val = rotation_penalty(node)
            exploitation_penalty_val = exploitation_penalty(node)
            # print(f"Point source gain: {point_source_gain}")
            # print(f"Distance penalty: {distance_penalty_val}")
            # print(f"Rotation penalty: {rotation_penalty_val}")
            # print(f"Exploitation penalty: {exploitation_penalty_val}")
            # print(f"Final gain: {point_source_gain - distance_penalty_val - rotation_penalty_val - exploitation_penalty_val}")
            return point_source_gain - exploitation_penalty_val#- distance_penalty_val - rotation_penalty_val - exploitation_penalty_val
        else:
            return 0

    def distance_penalty(node: InformativeTreeNode) -> float:
        if node.parent:
            return np.exp(0.005 * np.linalg.norm(node.point - node.parent.point))
        return 1

    def rotation_penalty(node: InformativeTreeNode) -> float:
        if node.parent:
            theta_t = np.arctan2(node.point[1] - node.parent.point[1], node.point[0] - node.parent.point[0])
            return np.exp(0.005 * (theta_t**2) / 0.1)
        return 1

    def exploitation_penalty(node: InformativeTreeNode) -> float:
        if len(self.agents_obs_wp[agent_idx]) > 0:
            n_obs_wp = 0
            for i in range(len(self.agent_positions)):
                n_obs_wp += len([obs_wp for obs_wp in self.agents_obs_wp[i] if np.linalg.norm(node.point - obs_wp) < self.d_waypoint_distance*2])
            # print(f"n_obs_wp: {n_obs_wp}")
            return np.exp(0.5 * n_obs_wp)
        return 1

    final_gain = 0
    current_node = node
    while current_node.parent:
        final_gain += sources_gain(current_node)
        current_node = current_node.parent
    return max(final_gain, 0)

# Tree Generation Functions

def rrt_tree_generation(self, budget_portion: float, agent_idx: int) -> None:
    distance_travelled = 0
    while distance_travelled < budget_portion:
        random_point = np.random.rand(2) * (self.scenario.workspace_size[1] - self.scenario.workspace_size[0], self.scenario.workspace_size[3] - self.scenario.workspace_size[2]) + self.scenario.workspace_size[0], self.scenario.workspace_size[2]
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
        x_rand = np.random.rand(2) * (self.scenario.workspace_size[1] - self.scenario.workspace_size[0], self.scenario.workspace_size[3] - self.scenario.workspace_size[2]) + self.scenario.workspace_size[0], self.scenario.workspace_size[2]
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
    if len(self.measurements) % 5 == 0 and len(self.measurements) > 0:
        self.scenario.gp.fit(self.obs_wp, self.measurements)
        if hasattr(self, 'best_estimates') and self.best_estimates.size > 0:
            estimates, _, bic = estimate_sources_bayesian(
                self.full_path, self.measurements, self.lambda_b,
                self.max_sources, self.n_samples, self.s_stages, self.scenario
            )
            # print (f"Estimated {len(estimates)//3} sources: {estimates}")
            # print (f"BIC: {bic}")
            if bic > self.best_bic:
                print(f"\nEstimated {len(estimates)//3} sources: {estimates}")
                print(f"\nBEST BIC FROM GP: {bic}")
                self.best_bic = bic
                self.best_estimates = estimates.reshape((-1, 3))

def source_metric_information_update(self) -> None:
    self.measurements = sum((agent_measurements for agent_measurements in self.agents_measurements), [])
    self.full_path = sum((agent_path for agent_path in self.agents_full_path), [])
    if len(self.measurements) % 5 == 0 and len(self.measurements) > 0:
        estimates, _, bic = estimate_sources_bayesian(
            self.full_path, self.measurements, self.lambda_b,
            self.max_sources, self.n_samples, self.s_stages, self.scenario
        )
        if bic > self.best_bic:
            print(f"\nEstimated {len(estimates)//3} sources: {estimates}")
            print(f"\nBEST BIC: {bic}")
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
    """
    Generic path selection strategy for Multi-Agent Informative Source Metric RRT Path Planning algorithms.

    Parameters:
    - self: Assumes a class that inherits from InformativeRRTBaseClass
    - agent_idx: The index of the agent in the multi-agent system
    - current_position: The current position of the agent

    Returns:
    - path: The selected path for the agent as a list of points.
    """
    # get current root 
    current_root = self.agents_roots[agent_idx]
    if not current_root:
        return []
    current_leafs = []
    # from the root we can get the leaf nodes 
    def get_leafs(node):
        if not node.children:
            current_leafs.append(node)
        for child in node.children:
            get_leafs(child)

    get_leafs(current_root)

    if not current_leafs:
        return []
    if self.best_estimates.size == 0:
        selected_node = np.random.choice(current_leafs)
    else:
        # select the max unless it is the current position in that case go to the next best
        selected_node = max(current_leafs, key=lambda node: node.information)
        if np.array_equal(selected_node.point, current_position):
            current_leafs.remove(selected_node)
            if not current_leafs:
                return [current_position]
            selected_node = max(current_leafs, key=lambda node: node.information)


    possible_path = trace_path_to_root(selected_node)
    # make sure there is enough budget to do the path, if not stop where the budget ends
    if current_budget is not None and current_position is not None:
        path = []
        current_budget = current_budget
        for node in possible_path:
            if current_budget - np.linalg.norm(node - current_position) < 0:
                path.append(node)
                break
            path.append(node)
            self.information_update() 
            current_budget -= np.linalg.norm(node - current_position) if path == [] else np.linalg.norm(node - path[-1])
        return path
    return possible_path

def bias_beta_path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None) -> List[np.ndarray]:
    leaf_nodes = [node for node in self.tree_nodes[agent_idx] if not node.children]
    leaf_points = np.array([node.point for node in leaf_nodes])
    
    mu, stds = self.scenario.gp.predict(leaf_points, return_std=True)
    mu_normalized = (mu - np.mean(mu)) / np.std(mu) if np.std(mu) != 0 else mu
    stds_normalized = (stds - np.mean(stds)) / np.std(stds) if np.std(stds) != 0 else stds

    self.uncertainty_reduction.append(np.mean(stds))
    acquisition_values = mu_normalized + self.beta_t * stds_normalized
    # give bonus to the closest best estimate nearest to current position 
    if current_position is not None and hasattr(self, 'best_estimates') and self.best_estimates.size > 0:
        distances = [np.linalg.norm(current_position - estimate[:2]) for estimate in self.best_estimates]
        bonus = 1 - np.tanh(np.array(distances) / 2)
        for i, estimate in enumerate(self.best_estimates):
            acquisition_values += bonus[i] * np.exp(-np.linalg.norm(leaf_points - estimate[:2], axis=1))
    max_acq_idx = np.argmax(acquisition_values)
    selected_leaf = leaf_nodes[max_acq_idx]
    
    path = []
    current_node = selected_leaf
    while current_node is not None:
        path.append(current_node.point)
        current_node = current_node.parent
    path.reverse()
    return path

# Base Class and Specific Implementations

class InformativeRRTBaseClass:
    def __init__(self, scenario: PointSourceField, beta_t: float = 5.0, budget: float = 375, d_waypoint_distance: float = 2.5, num_agents: int = 1,
                 n_samples: int = 25, s_stages: int = 10, lambda_b: float = 1, max_sources: int = 3, budget_iter: int = 10, **kwargs):
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

    def initialize_trees(self, start_position: np.ndarray, agent_idx: int) -> None:
        self.agents_roots[agent_idx] = InformativeTreeNode(start_position)
        self.tree_nodes[agent_idx] = [self.agents_roots[agent_idx]]
        self.trees.add(self.agents_roots[agent_idx])

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        budget_portion = [budget / self.budget_iter for budget in self.budget]

        for i in range(self.num_agents):
            self.initialize_trees(self.agent_positions[i], i)
        start_time = time.time()
        closest = None
        with tqdm(total=sum(self.budget), desc="Running " + str(self.num_agents) + " Agent " + self.name) as pbar:
            while any(b > 0 for b in self.budget):
                for i in range(self.num_agents):
                    if self.budget[i] > 0:
                        self.tree_generation(self.budget[i], i)
                        # if budget spent greater or equal than 2/3 of the budget, then do boustroped path selection
                        if self.budget[i] <= 7/8 * budget_portion[i] * self.budget_iter:
                            # first go to the closest best estimat
                            # if closest is None:
                            #     closest = np.argmin([np.linalg.norm(self.agents_full_path[i][-1] - estimate[:2]) for estimate in self.best_estimates])
                            #     print(f"Closest: {closest}")
                            #     self.agents_full_path[i].append(self.best_estimates[closest][:2])
                            #     self.update_observations_and_model([self.best_estimates[closest][:2]], i)
                            path = bias_beta_path_selection(self, i, self.agents_full_path[i][-1] if self.agents_full_path[i] else None)
                            # # create a scneario around the urrent last best estimate with + 5 -5 in x and y
                            # last_estimate = self.best_estimates[-1] if hasattr(self, 'best_estimates') and self.best_estimates.size > 0 else None
                            # if last_estimate is not None:
                            #     scenario = PointSourceField(num_sources=1, workspace_size=(last_estimate[0] - 5, last_estimate[0] + 5, last_estimate[1] - 5, last_estimate[1] + 5),
                            #                                 intensity_range=self.scenario.intensity_range)
                            #     scenario.update_source(0, *last_estimate)
                            #     boustrophedon = Boustrophedon(scenario=scenario, budget=self.budget[i], line_spacing=self.d_waypoint_distance)
                            #     boustrophedon.run()
                            #     path = boustrophedon.full_path
                            #     path = [np.array([x, y]) for x, y in path.T]
                            #     budget_spent = self.calculate_budget_spent(path)
                            #     self.update_observations_and_model(path, i)
                            #     self.agents_full_path[i].extend(path)
                            #     self.budget[i] -= budget_spent
                            #     pbar.update(budget_spent)
                            #     self.initialize_trees(path[-1], i)
                            # self.update_observations_and_model(path, i)
                            # self.agents_full_path[i].extend(path)
                            # budget_spent = self.calculate_budget_spent(path)
                            # self.budget[i] -= budget_spent
                            # pbar.update(budget_spent)
                            # gp_information_update(self)
                            # continue
                        else:
                            path = self.path_selection(i, self.agents_full_path[i][-1] if self.agents_full_path[i] else None)
                        budget_spent = self.calculate_budget_spent(path)
                        self.update_observations_and_model(path, i)
                        self.agents_full_path[i].extend(path)
                        self.budget[i] -= budget_spent

                        pbar.update(budget_spent)
                        if len(path) > 0 and self.budget[i] > 0:
                            self.initialize_trees(path[-1], i)
                        self.information_update() if self.budget[i] > 1/8 * budget_portion[i] * self.budget_iter else gp_information_update(self)
        self.time_taken = time.time() - start_time
        return self.finalize_all_agents()

    def update_observations_and_model(self, path: List[np.ndarray], agent_idx: int) -> None:
        if path:
            for point in path:
                measurement = self.scenario.simulate_measurements([point])[0]
                self.agents_measurements[agent_idx].append(measurement)
                self.agents_obs_wp[agent_idx].append(point)

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

        if hasattr(self, 'best_estimates') and self.best_estimates.size > 0:
            num_sources = len(self.best_estimates)
            self.new_scenario = PointSourceField(num_sources=num_sources, workspace_size=self.scenario.workspace_size,
                                        intensity_range=self.scenario.intensity_range)
            for i, estimate in enumerate(self.best_estimates):
                self.new_scenario.update_source(i, *estimate)
            std = np.zeros_like(self.scenario.g_truth)
            self.Z_pred = self.new_scenario.g_truth
        else:
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

class RRTRIG_PointSourceInformative_All_SourceMetric_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_estimates = np.array([])
        
        self.name = "RRTRIG_PointSourceInformative_All_SourceMetric_Path"

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

# src/rrt/rrt.py

"""
RRT (Rapidly-exploring Random Trees) path planning implementation.
"""

import numpy as np
import time
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import uniform
from typing import List, Tuple, Callable, Optional
from tqdm import tqdm
from src.estimation.estimation import estimate_sources_bayesian
from src.rrt.rrt_utils import (
    choose_parent, cost, line_cost, obstacle_free, rewire, near, steer, 
    nearest, add_node, trace_path_to_root, node_selection_key_distance, 
    InformativeTreeNode, TreeNode, TreeCollection
)
from src.point_source.point_source import PointSourceField

# Gain Calculation Functions

def point_source_gain_no_penalties(self, node: InformativeTreeNode, agent_idx: int) -> float:
    def sources_gain(node: InformativeTreeNode) -> float:
        x_t, y_t = node.point
        point_source_gain = 0
        if hasattr(self, 'best_estimates') and self.best_estimates.size > 0:
            for source in self.best_estimates:
                x_k, y_k, intensity = source
                d_src = np.linalg.norm([x_t - x_k, y_t - y_k])
                point_source_gain += intensity / d_src**2
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
        if hasattr(self, 'best_estimates') and self.best_estimates.size > 0:
            for source in self.best_estimates:
                x_k, y_k, intensity = source
                d_src = np.linalg.norm([x_t - x_k, y_t - y_k])
                point_source_gain += intensity / d_src**2
            return point_source_gain * distance_penalty(node)
        else:
            return 0

    def distance_penalty(node: InformativeTreeNode) -> float:
        if node.parent:
            return np.exp(-500 * np.linalg.norm(node.point - node.parent.point))
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
        if hasattr(self, 'best_estimates') and self.best_estimates.size > 0:
            for source in self.best_estimates:
                x_k, y_k, intensity = source
                d_src = np.linalg.norm([x_t - x_k, y_t - y_k])
                point_source_gain += intensity / d_src**2
            return point_source_gain * distance_penalty(node) * rotation_penalty(node)
        else:
            return 0

    def distance_penalty(node: InformativeTreeNode) -> float:
        if node.parent:
            return np.exp(-500 * np.linalg.norm(node.point - node.parent.point))
        return 1

    def rotation_penalty(node: InformativeTreeNode) -> float:
        if node.parent:
            theta_t = np.arctan2(node.point[1] - node.parent.point[1], node.point[0] - node.parent.point[0])
            return np.exp(-10 * (theta_t**2) / 0.1)
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
        if hasattr(self, 'best_estimates') and self.best_estimates.size > 0:
            for source in self.best_estimates:
                x_k, y_k, intensity = source
                d_src = np.linalg.norm([x_t - x_k, y_t - y_k])
                point_source_gain += intensity / d_src**2
            return point_source_gain * distance_penalty(node) * rotation_penalty(node) * exploitation_penalty(node)
        else:
            return 0

    def distance_penalty(node: InformativeTreeNode) -> float:
        if node.parent:
            return np.exp(-10 * np.linalg.norm(node.point - node.parent.point))
        return 1

    def rotation_penalty(node: InformativeTreeNode) -> float:
        if node.parent:
            theta_t = np.arctan2(node.point[1] - node.parent.point[1], node.point[0] - node.parent.point[0])
            return np.exp(-10*(theta_t**2) / 0.1)
        return 1

    def exploitation_penalty(node: InformativeTreeNode) -> float:
        if len(self.agents_obs_wp[agent_idx]) > 0:
            n_obs_wp = 0
            for i in range(len(self.agent_positions)):
                if i != agent_idx:
                    n_obs_wp += len([obs_wp for obs_wp in self.agents_obs_wp[i] if np.linalg.norm(node.point - obs_wp) < self.d_waypoint_distance])
            return np.exp(-5 * n_obs_wp)
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
        random_point = np.random.rand(2) * self.scenario.workspace_size
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
        x_rand = np.random.rand(2) * self.scenario.workspace_size
        x_nearest = nearest(self.tree_nodes[agent_idx], x_rand)
        x_new = steer(x_nearest, x_rand, d_max_step=1.0)
        
        if obstacle_free(x_nearest.point, x_new):
            x_new = InformativeTreeNode(x_new)
            X_near = near(x_new, self.tree_nodes[agent_idx], self.d_waypoint_distance)
            
            c_min = cost(x_nearest) + line_cost(x_nearest, x_new)
            x_min = x_nearest
            
            for x_near in X_near:
                if obstacle_free(x_near, x_new) and cost(x_near) + line_cost(x_near, x_new) < c_min:
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
        random_point = np.random.rand(2) * self.scenario.workspace_size
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
    if len(self.measurements) % 10 == 0 and len(self.measurements) > 0:
        self.scenario.gp.fit(self.obs_wp, self.measurements)

def source_metric_information_update(self) -> None:
    self.measurements = sum((agent_measurements for agent_measurements in self.agents_measurements), [])
    self.full_path = sum((agent_path for agent_path in self.agents_full_path), [])
    if len(self.measurements) % 5 == 0 and len(self.measurements) > 0:
        estimates, _, bic = estimate_sources_bayesian(
            self.full_path, self.measurements, self.lambda_b,
            self.max_sources, self.n_samples, self.s_stages, self.scenario
        )
        if bic > self.best_bic:
            self.best_bic = bic
            self.best_estimates = estimates.reshape((-1, 3))

# Path Selection Functions

def random_path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None) -> List[np.ndarray]:
    leaf_nodes = [node for node in self.tree_nodes[agent_idx] if not node.children]
    if not leaf_nodes:
        return []
    selected_leaf = np.random.choice(leaf_nodes)
    return trace_path_to_root(selected_leaf)

def informative_source_metric_path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None) -> List[np.ndarray]:   
    """
    Generic path selection strategy for Multi-Agent Informative Source Metric RRT Path Planning algorithms.

    Parameters:
    - self: Assumes a class that inherits from InformativeRRTBaseClass.
    - agent_idx: The index of the agent in the multi-agent system.
    """
    all_nodes = self.tree_nodes[agent_idx]
    if not all_nodes:
        return []

    if self.best_estimates.size == 0:
        selected_node = np.random.choice(all_nodes)
    else:
        selected_node = max(all_nodes, key=lambda node: node.information)

    if current_position is not None:
        current_node = min(all_nodes, key=lambda node: np.linalg.norm(node.point - current_position))
        path_to_current = trace_path_to_root(current_node)
        path_to_selected = trace_path_to_root(selected_node)
        common_node = None
        for node1 in reversed(path_to_current):
            for node2 in reversed(path_to_selected):
                if np.array_equal(node1, node2):
                    common_node = node1
                    break
            if common_node is not None:
                break
        if common_node is not None:
            index_current = next(i for i, node in enumerate(path_to_current) if np.array_equal(node, common_node))
            index_selected = next(i for i, node in enumerate(path_to_selected) if np.array_equal(node, common_node))
            first_part = []
            for node in reversed(path_to_current[index_current-1:]):
                first_part.append(node) 
            second_part = []
            for node in path_to_selected[index_selected:]:
                second_part.append(node)
            path = first_part + second_part
        else:
            path = path_to_current + path_to_selected
        return [node for node in path]

    return [node for node in trace_path_to_root(selected_node)]

def bias_beta_path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None) -> List[np.ndarray]:
    leaf_nodes = [node for node in self.tree_nodes[agent_idx] if not node.children]
    leaf_points = np.array([node.point for node in leaf_nodes])
    
    mu, stds = self.scenario.gp.predict(leaf_points, return_std=True)
    mu_normalized = (mu - np.mean(mu)) / np.std(mu) if np.std(mu) != 0 else mu
    stds_normalized = (stds - np.mean(stds)) / np.std(stds) if np.std(stds) != 0 else stds

    self.uncertainty_reduction.append(np.mean(stds))
    acquisition_values = mu_normalized + self.beta_t * stds_normalized
    
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

        self.n_samples = n_samples
        self.s_stages = s_stages
        self.lambda_b = lambda_b
        self.max_sources = max_sources

        if self.num_agents > 1:
            self.agent_positions = [np.array([0.5 + i * (self.scenario.workspace_size[0] - 1) / (num_agents - 1), 0.5]) for i in range(num_agents)]
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
        with tqdm(total=sum(self.budget), desc="Running " + str(self.num_agents) + " Agent " + self.name) as pbar:
            while any(b > 0 for b in self.budget):
                for i in range(self.num_agents):
                    if self.budget[i] > 0:
                        self.tree_generation(budget_portion[i], i)
                        path = self.path_selection(i, self.agents_full_path[i][-1] if self.agents_full_path[i] else None)
                        budget_spent = self.calculate_budget_spent(path)
                        self.update_observations_and_model(path, i)
                        self.agents_full_path[i].extend(path)
                        self.budget[i] -= budget_spent

                        pbar.update(budget_spent)
                        if len(path) > 0 and self.budget[i] > 0:
                            self.initialize_trees(path[-1], i)
                self.information_update()
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
                                        intensity_range=(1e4, 1e5))
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
        self.best_bic = -np.inf
        self.name = "RRTRIG_PointSourceInformative_SourceMetric_Path"

    def tree_generation(self, budget_portion: float, agent_idx: int) -> None:
        rig_tree_generation(self, budget_portion, agent_idx, gain_function=point_source_gain_no_penalties)

    def path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None) -> List[np.ndarray]:
        return informative_source_metric_path_selection(self, agent_idx, current_position)

    def information_update(self) -> None:
        source_metric_information_update(self)

class RRTRIG_PointSourceInformative_Distance_SourceMetric_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_estimates = np.array([])
        self.best_bic = -np.inf
        self.name = "RRTRIG_PointSourceInformative_Distance_SourceMetric_Path"

    def tree_generation(self, budget_portion: float, agent_idx: int) -> None:
        rig_tree_generation(self, budget_portion, agent_idx, gain_function=point_source_gain_only_distance_penalty)

    def path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None) -> List[np.ndarray]:
        return informative_source_metric_path_selection(self, agent_idx, current_position)
    
    def information_update(self) -> None:
        source_metric_information_update(self)

class RRTRIG_PointSourceInformative_DistanceRotation_SourceMetric_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_estimates = np.array([])
        self.best_bic = -np.inf
        self.name = "RRTRIG_PointSourceInformative_DistanceRotation_SourceMetric_Path"

    def tree_generation(self, budget_portion: float, agent_idx: int) -> None:
        rig_tree_generation(self, budget_portion, agent_idx, gain_function=point_source_gain_distance_rotation_penalty)

    def path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None) -> List[np.ndarray]:
        return informative_source_metric_path_selection(self, agent_idx, current_position)

    def information_update(self) -> None:
        source_metric_information_update(self)

class RRTRIG_PointSourceInformative_All_SourceMetric_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_estimates = np.array([])
        self.best_bic = -np.inf
        self.name = "RRTRIG_PointSourceInformative_All_SourceMetric_Path"

    def tree_generation(self, budget_portion: float, agent_idx: int) -> None:
        rig_tree_generation(self, budget_portion, agent_idx, gain_function=point_source_gain_all)

    def path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None) -> List[np.ndarray]:
        return informative_source_metric_path_selection(self, agent_idx, current_position)

    def information_update(self) -> None:
        source_metric_information_update(self)

class RRT_BiasBetaInformative_GP_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, directional_bias: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.directional_bias = directional_bias
        self.name = "RRT_BiasBetaInformative_GP_Path"

    def path_selection(self, agent_idx: int, current_position: Optional[np.ndarray] = None) -> List[np.ndarray]:
        return bias_beta_path_selection(self, agent_idx, current_position)

    def tree_generation(self, budget_portion: float, agent_idx: int) -> None:
        rrt_tree_generation(self, budget_portion, agent_idx)

    def information_update(self) -> None:
        gp_information_update(self)

import numpy as np
import time
from matplotlib import pyplot as plt
import threading
from scipy.spatial import KDTree
from scipy.stats import uniform
from informative import BaseInformative
from tqdm import tqdm
from path_planning_utils import estimate_sources_bayesian
from rrt_utils import choose_parent, cost, line_cost, obstacle_free, rewire, \
                      near, steer, nearest, add_node, trace_path_to_root, \
                      node_selection_key_distance, InformativeTreeNode, TreeNode, TreeCollection
from radiation import RadiationField

# Gain Calculation Functions

def point_source_gain_no_penalties(self, node, agent_idx):
    def sources_gain(node):
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
    if final_gain < 0:
        return 0
    return final_gain

def point_source_gain_only_distance_penalty(self, node, agent_idx):
    def sources_gain(node):
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

    def distance_penalty(node):
        if node.parent:
            return np.exp(-500 * np.linalg.norm(node.point - node.parent.point))
        return 1

    final_gain = 0
    current_node = node
    while current_node.parent:
        final_gain += sources_gain(current_node) 
        current_node = current_node.parent

    if final_gain < 0:
        return 0
    return final_gain

def point_source_gain_distance_rotation_penalty(self, node, agent_idx):
    def sources_gain(node):
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

    def distance_penalty(node):
        if node.parent:
            return np.exp(-500 * np.linalg.norm(node.point - node.parent.point))
        return 1

    def rotation_penalty(node):
        if node.parent:
            theta_t = np.arctan2(node.point[1] - node.parent.point[1], node.point[0] - node.parent.point[0])
            return np.exp(-10*(theta_t**2) / 0.1)
        return 1

    final_gain = 0
    current_node = node
    while current_node.parent:
        final_gain += sources_gain(current_node) 
        current_node = current_node.parent

    if final_gain < 0:
        return 0
    return final_gain
def point_source_gain_all(self, node, agent_idx):
    def sources_gain(node):
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

    def distance_penalty(node):
        if node.parent:
            return np.exp(-500 * np.linalg.norm(node.point - node.parent.point))
        return 1

    def rotation_penalty(node):
        if node.parent:
            theta_t = np.arctan2(node.point[1] - node.parent.point[1], node.point[0] - node.parent.point[0])
            return np.exp(-10*(theta_t**2) / 0.1)
        return 1

    def exploitation_penalty(node):
        if len(self.agents_obs_wp[agent_idx]) > 0:
            n_obs_wp = 0
            for i in range(len(self.agent_positions)):
                if i != agent_idx:
                    n_obs_wp += len([obs_wp for obs_wp in self.agents_obs_wp[i] if np.linalg.norm(node.point - obs_wp) < self.d_waypoint_distance])
            return np.exp(-10000 * n_obs_wp)
        return 1

    final_gain = 0
    current_node = node
    while current_node.parent:
        final_gain += sources_gain(current_node) 
        current_node = current_node.parent

    if final_gain < 0:
        return 0
    return final_gain
# Tree Generation Functions

def rrt_tree_generation(self, budget_portion, agent_idx):
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

def rrt_star_tree_generation(self, budget_portion, agent_idx):
    distance_travelled = 0
    while distance_travelled < budget_portion:
        x_rand = np.random.rand(2) * self.scenario.workspace_size
        x_nearest = nearest(self.tree_nodes[agent_idx], x_rand)
        x_new = steer(x_nearest, x_rand, d_max_step=1.0)
        
        if obstacle_free(x_nearest, x_new):
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

def rig_tree_generation(self, budget_portion, agent_idx, gain_function=point_source_gain_no_penalties):
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

def gp_information_update(self):
    self.measurements = sum((agent_measurements for agent_measurements in self.agents_measurements), [])
    self.obs_wp = sum((agent_wp for agent_wp in self.agents_obs_wp), [])
    if len(self.measurements) % 10 == 0 and len(self.measurements) > 0:
        self.scenario.gp.fit(self.obs_wp, self.measurements)

def source_metric_information_update(self):
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

def random_path_selection(self, agent_idx, current_position=None):
    leaf_nodes = [node for node in self.tree_nodes[agent_idx] if not node.children]
    if not leaf_nodes:
        return []
    selected_leaf = np.random.choice(leaf_nodes)
    return trace_path_to_root(selected_leaf)

def informative_source_metric_path_selection(self, agent_idx, current_position=None):   
    """
    Generic path selection strategy for Multi-Agent Informative Source Metric RRT Path Planning algorithms

    Parameters:
    - self: Assumes a class that inherits from InformativeRRTBaseClass
    - agent_idx: The index of the agent in the multi-agent system

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
        # Ensure we trace the path from the current node to the selected node without jumping to the root
        path_to_current = trace_path_to_root(current_node)
        path_to_selected = trace_path_to_root(selected_node)
        # print("Path to Current: ", [node for node in path_to_current])
        # print("Path to Selected: ", [node for node in path_to_selected])
        # Find the highest index common node by comparing node points
        common_node = None
        for node1 in reversed(path_to_current):
            for node2 in reversed(path_to_selected):
                if np.array_equal(node1, node2):
                    common_node = node1
                    break
            if common_node is not None:
                break
        if common_node is not None:
            # print("Common Node: ", common_node)
            index_current = next(i for i, node in enumerate(path_to_current) if np.array_equal(node, common_node))
            index_selected = next(i for i, node in enumerate(path_to_selected) if np.array_equal(node, common_node))
            first_part = []
            for node in reversed(path_to_current[index_current-1:]):
                first_part.append(node) 
            second_part = []
            for node in path_to_selected[index_selected:]:
                second_part.append(node)
            # print("First Part: ", first_part)
            # print("Second Part: ", second_part)
            path = first_part + second_part
        else:
            path = path_to_current + path_to_selected
        return [node for node in path]

    return [node for node in trace_path_to_root(selected_node)]


def bias_beta_path_selection(self, agent_idx, current_position=None):
    leaf_nodes = [node for node in self.tree_nodes[agent_idx] if not node.children]
    leaf_points = np.array([node.point for node in leaf_nodes])
    
    mu, stds = self.scenario.gp.predict(leaf_points, return_std=True)
    if np.std(mu) == 0:
        mu_normalized = mu
    else:
        mu_normalized = (mu - np.mean(mu)) / np.std(mu)
    if np.std(stds) == 0:
        stds_normalized = stds
    else: 
        stds_normalized = (stds - np.mean(stds)) / np.std(stds)

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

class InformativeRRTBaseClass():
    def __init__(self, scenario, beta_t=5.0, budget=375, d_waypoint_distance=2.5, num_agents=1,
                 n_samples=25, s_stages=10, lambda_b=1, max_sources=3, budget_iter=10, **kwargs):
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

    def initialize_trees(self, start_position, agent_idx):
        self.agents_roots[agent_idx] = InformativeTreeNode(start_position)
        self.tree_nodes[agent_idx] = [self.agents_roots[agent_idx]]
        self.trees.add(self.agents_roots[agent_idx])

    def run(self):
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

    def update_observations_and_model(self, path, agent_idx):
        if path:
            for point in path:
                measurement = self.scenario.simulate_measurements([point])[0]
                self.agents_measurements[agent_idx].append(measurement)
                self.agents_obs_wp[agent_idx].append(point)

    def calculate_budget_spent(self, path):
        if not path:
            return 0
        path_array = np.array(path)
        if path_array.ndim > 1:
            return np.sum(np.linalg.norm(np.diff(path_array, axis=0), axis=1))
        return 0
    
    def finalize_all_agents(self):
        self.measurements = sum((agent_measurements for agent_measurements in self.agents_measurements), [])
        self.obs_wp = sum((agent_wp for agent_wp in self.agents_obs_wp), [])
        self.full_path = sum((agent_path for agent_path in self.agents_full_path), [])
        self.obs_wp = np.array(self.obs_wp)
        self.full_path = np.array(self.full_path).reshape(-1, 2).T

        if hasattr(self, 'best_estimates') and self.best_estimates.size > 0:
            num_sources = len(self.best_estimates)
            self.new_scenario = RadiationField(num_sources=num_sources, workspace_size=self.scenario.workspace_size,
                                        intensity_range=(1e4, 1e5))
            for i, estimate in enumerate(self.best_estimates):
                self.new_scenario.update_source(i, *estimate)
            print("\nBest Estimates: ", self.best_estimates, "\n")
            std = np.zeros_like(self.scenario.g_truth)
            self.Z_pred = self.new_scenario.g_truth
        else:
            self.Z_pred, std = self.scenario.predict_spatial_field(self.obs_wp, np.array(self.measurements))
        
        return self.Z_pred, std
    
    def tree_generation(self, budget_portion, agent_idx):
        raise NotImplementedError("tree_generation method must be implemented in the subclass")
    
    def path_selection(self, agent_idx, current_position=None):
        raise NotImplementedError("path_selection method must be implemented in the subclass")

    def information_update(self):
        raise NotImplementedError("information_update method must be implemented in the subclass")

class RRT_Random_GP_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "RRT_Random_GP_Path"

    def tree_generation(self, budget_portion, agent_idx):
        rrt_tree_generation(self, budget_portion, agent_idx)

    def path_selection(self, agent_idx):
        return random_path_selection(self, agent_idx)

    def information_update(self):
        gp_information_update(self)

class RRTStar_Random_GP_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "RRTStar_Random_GP_Path"

    def tree_generation(self, budget_portion, agent_idx):
        rrt_star_tree_generation(self, budget_portion, agent_idx)

    def path_selection(self, agent_idx):
        return random_path_selection(self, agent_idx)

    def information_update(self):
        gp_information_update(self)

class RRTRIG_PointSourceInformative_SourceMetric_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_estimates = np.array([])
        self.best_bic = -np.inf
        self.name = "RRTRIG_PointSourceInformative_SourceMetric_Path"

    def tree_generation(self, budget_portion, agent_idx):
        rig_tree_generation(self, budget_portion, agent_idx, gain_function=point_source_gain_no_penalties)

    def path_selection(self, agent_idx, current_position=None):
        return informative_source_metric_path_selection(self, agent_idx, current_position)

    def information_update(self):
        source_metric_information_update(self)

class RRTRIG_PointSourceInformative_Distance_SourceMetric_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_estimates = np.array([])
        self.best_bic = -np.inf
        self.name = "RRTRIG_PointSourceInformative_Distance_SourceMetric_Path"

    def tree_generation(self, budget_portion, agent_idx):
        rig_tree_generation(self, budget_portion, agent_idx, gain_function=point_source_gain_only_distance_penalty)

    def path_selection(self, agent_idx, current_position=None):
        return informative_source_metric_path_selection(self, agent_idx, current_position)
    
    def information_update(self):
        source_metric_information_update(self)

class RRTRIG_PointSourceInformative_DistanceRotation_SourceMetric_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_estimates = np.array([])
        self.best_bic = -np.inf
        self.name = "RRTRIG_PointSourceInformative_DistanceRotation_SourceMetric_Path"

    def tree_generation(self, budget_portion, agent_idx):
        rig_tree_generation(self, budget_portion, agent_idx, gain_function=point_source_gain_distance_rotation_penalty)

    def path_selection(self, agent_idx, current_position=None):
        return informative_source_metric_path_selection(self, agent_idx, current_position)

    def information_update(self):
        source_metric_information_update(self)

class RRTRIG_PointSourceInformative_All_SourceMetric_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_estimates = np.array([])
        self.best_bic = -np.inf
        self.name = "RRTRIG_PointSourceInformative_All_SourceMetric_Path"

    def tree_generation(self, budget_portion, agent_idx):
        rig_tree_generation(self, budget_portion, agent_idx, gain_function=point_source_gain_all)

    def path_selection(self, agent_idx, current_position=None):
        return informative_source_metric_path_selection(self, agent_idx, current_position)

    def information_update(self):
        source_metric_information_update(self)

class RRT_BiasBetaInformative_GP_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, directional_bias=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.directional_bias = directional_bias
        self.name = "RRT_BiasBetaInformative_GP_Path"

    def path_selection(self, agent_idx, current_position=None):
        return bias_beta_path_selection(self, agent_idx, current_position)

    def tree_generation(self, budget_portion, agent_idx):
        rrt_tree_generation(self, budget_portion, agent_idx)

    def information_update(self):
        gp_information_update(self)


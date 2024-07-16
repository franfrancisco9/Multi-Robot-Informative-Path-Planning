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
                
                # Calculate the kth sources suppression factor F_src^(t,k)(node_t)
                F_src = 0
                for other_source in self.best_estimates:
                    if not np.array_equal(source, other_source):
                        x_j, y_j, _ = other_source
                        mid_point = [(x_k + x_j) / 2, (y_k + y_j) / 2]
                        d_t_k_j = np.linalg.norm([x_t - mid_point[0], y_t - mid_point[1]])
                        
                        # Distance suppression factor
                        C_src_Dist = 2 - 1 / (1 + np.exp((d_t_k_j - 2) / 16))
                        
                        # Suppression factor
                        F_src += (1 - C_src_Dist + C_src_Dist / (1 + np.exp((2 - d_t_k_j) / 16)))
                
                # Add the point source gain
                point_source_gain += (1 + np.exp(-(d_src - 2)**2 / 2*16)) * F_src
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
                
                # Calculate the kth sources suppression factor F_src^(t,k)(node_t)
                F_src = 0
                for other_source in self.best_estimates:
                    if not np.array_equal(source, other_source):
                        x_j, y_j, _ = other_source
                        mid_point = [(x_k + x_j) / 2, (y_k + y_j) / 2]
                        d_t_k_j = np.linalg.norm([x_t - mid_point[0], y_t - mid_point[1]])
                        
                        # Distance suppression factor
                        C_src_Dist = 2 - 1 / (1 + np.exp((d_t_k_j - 2) / 16))
                        
                        # Suppression factor
                        F_src += (1 - C_src_Dist + C_src_Dist / (1 + np.exp((2 - d_t_k_j) / 16)))
                
                # Add the point source gain
                point_source_gain += (1 + np.exp(-(d_src - 2)**2 / 2*16)) * F_src
            return point_source_gain * distance_penalty(node) 
        else:
            return 0

    def distance_penalty(node):
        if node.parent:
            return np.exp(0.5 * np.linalg.norm(node.point - node.parent.point))
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
                
                # Calculate the kth sources suppression factor F_src^(t,k)(node_t)
                F_src = 0
                for other_source in self.best_estimates:
                    if not np.array_equal(source, other_source):
                        x_j, y_j, _ = other_source
                        mid_point = [(x_k + x_j) / 2, (y_k + y_j) / 2]
                        d_t_k_j = np.linalg.norm([x_t - mid_point[0], y_t - mid_point[1]])
                        
                        # Distance suppression factor
                        C_src_Dist = 2 - 1 / (1 + np.exp((d_t_k_j - 2) / 16))
                        
                        # Suppression factor
                        F_src += (1 - C_src_Dist + C_src_Dist / (1 + np.exp((2 - d_t_k_j) / 16)))
                
                # Add the point source gain
                point_source_gain += (1 + np.exp(-(d_src - 2)**2 / 2*16)) * F_src
            return point_source_gain * distance_penalty(node) * rotation_penalty(node) 
        else:
            return 0

    def distance_penalty(node):
        if node.parent:
            return np.exp(0.5 * np.linalg.norm(node.point - node.parent.point))
        return 1

    def rotation_penalty(node):
        if node.parent:
            theta_t = np.arctan2(node.point[1] - node.parent.point[1], node.point[0] - node.parent.point[0])
            return np.exp(0.05*(theta_t**2) / 0.1)
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
                
                # Calculate the kth sources suppression factor F_src^(t,k)(node_t)
                F_src = 0
                for other_source in self.best_estimates:
                    if not np.array_equal(source, other_source):
                        x_j, y_j, _ = other_source
                        mid_point = [(x_k + x_j) / 2, (y_k + y_j) / 2]
                        d_t_k_j = np.linalg.norm([x_t - mid_point[0], y_t - mid_point[1]])
                        
                        # Distance suppression factor
                        C_src_Dist = 2 - 1 / (1 + np.exp((d_t_k_j - 2) / 16))
                        
                        # Suppression factor
                        F_src += (1 - C_src_Dist + C_src_Dist / (1 + np.exp((2 - d_t_k_j) / 16)))
                
                # Add the point source gain
                point_source_gain += (1 + np.exp(-(d_src - 2)**2 / (2 * 16))) * F_src

            #print(f"Point Source Gain before penalties: {point_source_gain}")
            final_gain = point_source_gain * exploitation_penalty(node) * distance_penalty(node) * rotation_penalty(node)
            #print(f"Point Source Gain after penalties: {final_gain}")
            return final_gain
        else:
            return 0

    def distance_penalty(node):
        if node.parent:
            penalty = np.exp(0.5 * np.linalg.norm(node.point - node.parent.point))  # Reduced exponent base
            #print(f"Distance Penalty: {penalty}")
            return penalty
        return 1

    def rotation_penalty(node):
        if node.parent:
            theta_t = np.arctan2(node.point[1] - node.parent.point[1], node.point[0] - node.parent.point[0])
            penalty = np.exp(0.05 * (theta_t**2) / 0.1)  # Adjusted factor
            #print(f"Rotation Penalty: {penalty}")
            return penalty
        return 1

    def exploitation_penalty(node):
        if len(self.agents_obs_wp[agent_idx]) > 0:
            n_obs_wp = 0
            for i in range(len(self.agent_positions)):
                for obs_wp in self.agents_obs_wp[i]:
                    distance = np.linalg.norm(node.point - obs_wp)
                    if distance < self.d_waypoint_distance*10:
                        n_obs_wp += 1
            penalty = np.exp(5 * n_obs_wp)
            #print(f"Exploitation Penalty: {penalty}, Number of Nearby Observations: {n_obs_wp}")
            return penalty
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
    if len(self.measurements) > 0:
        print(len(self.measurements), len(self.obs_wp))
        self.scenario.gp.fit(self.obs_wp, self.measurements)

def source_metric_information_update(self):
    self.measurements = sum((agent_measurements for agent_measurements in self.agents_measurements), [])
    self.full_path = sum((agent_path for agent_path in self.agents_full_path), [])
    
    if len(self.measurements) > 0:
        estimates, _, bic = estimate_sources_bayesian(
            self.full_path, self.measurements, self.lambda_b,
            self.max_sources, self.n_samples, self.s_stages, self.scenario
        )
        # print("\nBIC: ", bic, "\n")
        # print("\nEstimates: ", estimates, "\n")
        if bic > self.best_bic:
            self.best_bic = bic
            self.best_estimates = estimates.reshape((-1, 3))
            # print("\nNew Best BIC: ", self.best_bic, "\n")
            # print("\nNew Best Estimates: ", self.best_estimates, "\n")
    # self.best_estimates = np.array([[20, 20, 100000]])
    
# Path Selection Functions

def random_path_selection(self, agent_idx, current_position=None, current_budget=None):
    leaf_nodes = [node for node in self.tree_nodes[agent_idx] if not node.children]
    if not leaf_nodes:
        return []
    selected_leaf = np.random.choice(leaf_nodes)
    return trace_path_to_root(selected_leaf)

def informative_source_metric_path_selection(self, agent_idx, current_position=None, current_budget=None):
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
            current_budget -= np.linalg.norm(node - current_position)
        # if path == []:
        #     # we still need to move to at least one node so we select the first node
        #     path.append(current_position, possible_path[0])
        return path
    return possible_path

    budget = self.budget[agent_idx]
    all_nodes = self.tree_nodes[agent_idx]
    if not all_nodes:
        return []
    
    def find_best_node(nodes, exclude_set):
        """Find the best node not in the exclude set based on information."""
        candidates = [node for node in nodes]
        # if not candidates:
        #     return None
        return max(candidates, key=lambda node: node.information)

    if current_position is None:
        current_node = np.random.choice(all_nodes)
    else:
        current_node = min(all_nodes, key=lambda node: np.linalg.norm(node.point - current_position))

    path = [current_node]
    visited_nodes = set(path)
    # current_budget = budget if budget is not None else float('inf')

    # while current_budget > 0:
    best_node = find_best_node(all_nodes, visited_nodes)
    if not best_node:
        return [node.point for node in path]

    path_to_best = trace_path_to_root(best_node)
    path_to_current = trace_path_to_root(current_node)

    # Find the highest index common node by comparing node points
    common_node = None
    for node1 in reversed(path_to_current):
        for node2 in reversed(path_to_best):
            if np.array_equal(node1.point, node2.point):
                common_node = node1
                break

    if common_node is not None:
        index_current = next(i for i, node in enumerate(path_to_current) if np.array_equal(node.point, common_node.point))
        index_best = next(i for i, node in enumerate(path_to_best) if np.array_equal(node.point, common_node.point))
        first_part = list(reversed(path_to_current[index_current + 1:]))  # skip the common node in the first part
        second_part = path_to_best[index_best:]  # skip the common node in the second part
        segment = first_part + second_part  # avoid duplicating common node
    else:
        segment = [current_node] + [best_node]  # Directly go from current position to best node if no common node is found

    for node in segment:
        if current_node != node:
            # cost_to_next = np.linalg.norm(node.point - current_node.point)
            # if current_budget - cost_to_next < 0:
            #     path.append(current_node)
            #     return [node.point for node in path]
            # current_budget -= cost_to_next
            path.append(node)
            current_node = node
            visited_nodes.add(node)

    path.append(current_node)
    return [node.point for node in path]

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
                        path = self.path_selection(i, current_position=self.agent_positions[i], current_budget=self.budget[i])
                        # path =[[i, j] for i in range(0, 41, 5) for j in range(0, 41, 5)]
                        # print("Path: ", path)
                        budget_spent = self.calculate_budget_spent(path)
                        self.update_observations_and_model(path, i)
                        self.agents_full_path[i].extend(path)
                        # print("full path: ", self.agents_full_path[i])
                        self.budget[i] -= budget_portion[i]
                        # if self.budget[i] < budget_portion[i]:
                        #     self.budget[i] = 0
                        pbar.update(budget_spent)
                        if len(path) > 0 and self.budget[i] > 0:
                            self.initialize_trees(path[-1], i)
                        self.information_update()
        self.time_taken = time.time() - start_time
        return self.finalize_all_agents()

    def update_observations_and_model(self, path, agent_idx):
        if path:
            measurements = self.scenario.simulate_measurements(path)
            measurements_log = np.log10(measurements)
            self.agents_measurements[agent_idx].extend(measurements)
            self.agents_obs_wp[agent_idx].extend(path)

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
    
    def path_selection(self, agent_idx, current_position=None, current_budget=None):
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

    def path_selection(self, agent_idx, current_position=None, current_budget=None):
        return informative_source_metric_path_selection(self, agent_idx, current_position, current_budget)

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

    def path_selection(self, agent_idx, current_position=None, current_budget=None):
        return informative_source_metric_path_selection(self, agent_idx, current_position, current_budget)
    
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

    def path_selection(self, agent_idx, current_position=None, current_budget=None):
        return informative_source_metric_path_selection(self, agent_idx, current_position, current_budget)

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

    def path_selection(self, agent_idx, current_position=None, current_budget=None):
        return informative_source_metric_path_selection(self, agent_idx, current_position, current_budget)

    def information_update(self):
        source_metric_information_update(self)

class RRT_BiasBetaInformative_GP_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, directional_bias=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.directional_bias = directional_bias
        self.name = "RRT_BiasBetaInformative_GP_Path"

    def path_selection(self, agent_idx, current_position=None, current_budget=None):
        return bias_beta_path_selection(self, agent_idx, current_position)

    def tree_generation(self, budget_portion, agent_idx):
        rrt_tree_generation(self, budget_portion, agent_idx)

    def information_update(self):
        gp_information_update(self)


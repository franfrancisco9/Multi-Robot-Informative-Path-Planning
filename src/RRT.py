import numpy as np
from scipy.spatial import KDTree
from scipy.stats import uniform
from informative import BaseInformative
from tqdm import tqdm
from path_planning_utils import estimate_sources_bayesian
from rrt_utils import choose_parent, cost, line_cost, obstacle_free, rewire, \
                      near, steer, nearest, add_node, trace_path_to_root, \
                      node_selection_key_distance, InformativeTreeNode, TreeNode, TreeCollection

# RRT Tree Generic generation strategy
def rrt_tree_generation(self, budget_portion, agent_idx):
    """
    Generic tree generation strategy for RRT Path Planning algorithms

    Parameters:
    - self: Assumes a class that inherits from InformativeRRTBaseClass
    - budget_portion: The portion of the budget allocated for this iteration
    - agent_idx: The index of the agent in the multi-agent system~

    """
    distance_travelled = 0
    while distance_travelled < budget_portion:
        random_point = np.random.rand(2) * self.scenario.workspace_size
        nearest_node = min(self.tree_nodes[agent_idx], key=lambda node: node_selection_key_distance(node, random_point))
        new_point = steer(nearest_node, random_point, d_max_step=1.0)

        if obstacle_free(nearest_node.point, new_point):
            new_node = InformativeTreeNode(new_point, nearest_node)
            add_node(self.tree_nodes[agent_idx], new_node, nearest_node)
            distance_travelled += np.linalg.norm(new_node.point - nearest_node.point)
            if distance_travelled >= budget_portion:
                break

        self.agents_trees[agent_idx].add(self.root)

# RRT Star Tree Generic generation strategy
def rrt_star_tree_generation(self, budget_portion, agent_idx):
    """
    Generic tree generation strategy for RRT* Path Planning algorithms

    Parameters:
    - self: Assumes a class that inherits from InformativeRRTBaseClass
    - budget_portion: The portion of the budget allocated for this iteration
    - agent_idx: The index of the agent in the multi-agent system~

    """
    distance_travelled = 0
    while distance_travelled < budget_portion:
        x_rand = np.random.rand(2) * self.scenario.workspace_size
        x_nearest = nearest(self.tree_nodes[agent_idx], x_rand)  # Find the nearest node in the tree
        x_new = steer(x_nearest, x_rand, d_max_step=1.0)  # Steer from x_nearest towards x_rand
        
        if obstacle_free(x_nearest, x_new):
            x_new = InformativeTreeNode(x_new)
            X_near = near(x_new, self.tree_nodes[agent_idx], self.d_waypoint_distance)
            M
            # Initialize the minimum cost and the best parent node for x_new
            c_min = cost(x_nearest) + line_cost(x_nearest, x_new)
            x_min = x_nearest
            
            # Try to find a better parent if any in the X_near set reduces the cost to reach x_new
            for x_near in X_near:
                if obstacle_free(x_near, x_new) and \
                        cost(x_near) + line_cost(x_near, x_new) < c_min:
                    c_min = cost(x_near) + line_cost(x_near, x_new)
                    x_min = x_near
            
            # Add x_new to the tree with x_min as its parent
            add_node(self.tree_nodes[agent_idx], x_new, x_min)
            distance_travelled += np.linalg.norm(x_new.point - x_min.point)
            
            # Rewiring the tree: Check if x_new is a better parent for nodes in X_near
            rewire(X_near, x_new)
            if distance_travelled >= budget_portion:
                break
    self.agents_trees[agent_idx].add(self.root)  # Consider all nodes for plotting

# RIG Tree Generic generation strategy with point source gain calculation
def rig_tree_generation(self, budget_portion, agent_idx):
    """
    Generic tree generation strategy for Multi-Agent Informative Source Metric RRT Path Planning algorithms

    Parameters:
    - self: Assumes a class that inherits from InformativeRRTBaseClass
    - budget_portion: The portion of the budget allocated for this iteration
    - agent_idx: The index of the agent in the multi-agent system~

    """
    distance_travelled = 0
    while distance_travelled < budget_portion:
        random_point = np.random.rand(2) * self.scenario.workspace_size
        nearest_node = nearest(self.tree_nodes[agent_idx], random_point)
        new_point = steer(nearest_node, random_point, d_max_step=1.0)

        if obstacle_free(nearest_node.point, new_point):
            new_node = InformativeTreeNode(new_point)    
            X_near = near(new_node, self.tree_nodes[agent_idx], self.d_waypoint_distance)
            new_node = choose_parent(X_near, nearest_node, new_node)
            add_node(self.tree_nodes[agent_idx], new_node, nearest_node)
            distance_travelled += np.linalg.norm(new_node.point - nearest_node.point)
            
            # Update the information gain for the new node
            new_node.information = point_source_gain(self, new_node, agent_idx)

            rewire(X_near, new_node)
            if distance_travelled >= budget_portion:
                break

    self.agents_trees[agent_idx].add(self.root)

# GP Information Updated Generic Strategy
def gp_information_update(self):
    """
    Generic information update strategy for RRT Path Planning algorithms

    Parameters:
    - self: Assumes a class that inherits from InformativeRRTBaseClass

    """
    self.measurements = sum((agent_measurements for agent_measurements in self.agents_measurements), [])
    self.obs_wp = sum((agent_wp for agent_wp in self.agents_obs_wp), [])
    self.scenario.gp.fit(self.obs_wp, self.measurements)

# Source Metric Information Updated Generic Strategy
def source_metric_information_update(self):
    """
    Generic information update strategy for Multi-Agent Informative Source Metric RRT Path Planning algorithms

    Parameters:
    - self: Assumes a class that inherits from InformativeRRTBaseClass

    """
    self.measurements = sum((agent_measurements for agent_measurements in self.agents_measurements), [])
    self.full_path = sum((agent_path for agent_path in self.agents_full_path), [])
    estimates, _, bic = estimate_sources_bayesian(
        self.full_path, self.measurements, self.lambda_b,
        self.max_sources, self.n_samples, self.s_stages
    )
    if bic > self.best_bic:
        self.best_bic = bic
        self.best_estimates = estimates.reshape((-1, 3))
    
# Generic Random Path Selection Strategy
def random_path_selection(self, agent_idx):
    """
    Generic path selection strategy for RRT Path Planning algorithms

    Parameters:
    - self: Assumes a class that inherits from InformativeRRTBaseClass
    - agent

    """
    leaf_nodes = [node for node in self.tree_nodes[agent_idx] if not node.children]
    if not leaf_nodes:
        return []  # Return empty list if no leaf nodes are available

    # Randomly select a leaf node
    selected_leaf = np.random.choice(leaf_nodes)

    # Trace the path from the selected leaf back to the root
    return trace_path_to_root(selected_leaf)

# Generic Informative Source metric Path Selection Strategy
def informative_source_metric_path_selection(self, agent_idx):
    """
    Generic path selection strategy for Multi-Agent Informative Source Metric RRT Path Planning algorithms

    Parameters:
    - self: Assumes a class that inherits from InformativeRRTBaseClass
    - agent_idx: The index of the agent in the multi-agent system

    """
    leaf_nodes = [node for node in self.tree_nodes[agent_idx] if not node.children]
    if not leaf_nodes:
        return []  # Return empty list if no leaf nodes are available

    # Choose path based on radiation gain
    if self.best_estimates.size == 0:
        selected_leaf = np.random.choice(leaf_nodes)  # Random selection if no estimates are available
    else:
        # use node.information as the key for selection
        selected_leaf = max(leaf_nodes, key=lambda node: node.information)
    # Trace the path from the selected leaf back to the root
    return trace_path_to_root(selected_leaf)

# Generic Bias Beta Path Selection Strategy
def bias_beta_path_selection(self, agent_idx):
    """
    Generic path selection strategy for RRT Path Planning algorithms with bias beta

    Parameters:
    - self: Assumes a class that inherits from InformativeRRTBaseClass
    - agent_idx: The index of the agent in the multi-agent system

    """
    leaf_nodes = [node for node in self.tree_nodes[agent_idx] if not node.children]
    leaf_points = np.array([node.point for node in leaf_nodes])
    
    # Use Gaussian Process to predict the mean and standard deviation for leaf points
    mu, stds = self.scenario.gp.predict(leaf_points, return_std=True)
    # Normalize mu and stds to have mean 0 and std 1
    if np.std(mu) == 0:
        mu_normalized = mu
    else:
        mu_normalized = (mu - np.mean(mu)) / np.std(mu)
    if np.std(stds) == 0:
        stds_normalized = stds
    else: 
        stds_normalized = (stds - np.mean(stds)) / np.std(stds)

    self.uncertainty_reduction.append(np.mean(stds))
    # Calculate acquisition values considering both mean (mu) and standard deviation (stds)
    acquisition_values = mu_normalized + self.beta_t * stds_normalized
    
    # Select the leaf node with the highest acquisition value
    max_acq_idx = np.argmax(acquisition_values)
    selected_leaf = leaf_nodes[max_acq_idx]
    
    # Trace back to root from the selected leaf to form a path
    path = []
    current_node = selected_leaf
    while current_node is not None:
        path.append(current_node.point)
        current_node = current_node.parent
    path.reverse()  # Reverse to start from root
    return path

# Generic point source gain calculation strategy
def point_source_gain(self, node, agent_idx):
    """
    Generic point source gain calculation strategy for Multi-Agent Informative Source Metric RRT Path Planning algorithms

    Parameters:
    - self: Assumes a class that inherits from InformativeRRTBaseClass
    - node: The node for which the point source gain is to be calculated
    - agent_idx: The index of the agent in the multi-agent system

    """
    # Gain_nodet = Gain_nodet-1 + Gain_nodet_src
    def sources_gain(node):
        x_t, y_t = node.point
        point_source_gain = 0
        for source in self.best_estimates:
            x_k, y_k, intensity = source
            d_src = np.linalg.norm([x_t - x_k, y_t - y_k])
            point_source_gain += intensity / d_src**2
        return point_source_gain
    
    # If there are already obs_wp for the other agents, give a severe penalty to the node that is within 5m of the observation point of the other agent
    for i in range(self.num_agents):
        if i != agent_idx:
            for obs_point in self.agents_obs_wp[i]:
                if np.linalg.norm(node.point - obs_point) < 5:
                    return -np.inf
    
    final_gain = 0
    current_node = node
    while current_node.parent:
        final_gain += sources_gain(current_node)
        current_node = current_node.parent
    return final_gain

# Create a generic RRT Class with to be implemented methods and the overall structure and main loop of the algorithm
class InformativeRRTBaseClass():
    """
    Base class for Informative RRT Path Planning algorithms. Allows multiple agents
    """
    def __init__(self, scenario, beta_t=5.0, budget=375, d_waypoint_distance=2.5, num_agents=1,
                 n_samples=25, s_stages=10, lambda_b=1, max_sources=3, budget_iter=10, **kwargs):
        # RRT Parameters
        self.scenario = scenario
        self.beta_t = beta_t
        self.budget = [budget] * num_agents
        self.d_waypoint_distance = d_waypoint_distance
        self.num_agents = num_agents
        self.budget_iter = budget_iter
        self.name = None
        self.trees = TreeCollection()
        self.uncertainty_reduction = []

        # Source Metric Parameters
        self.n_samples = n_samples
        self.s_stages = s_stages
        self.lambda_b = lambda_b
        self.max_sources = max_sources

        # Number of agents specific attributes
        if self.num_agents > 1:
            self.agent_positions = [np.array([0.5 + i * (self.scenario.workspace_size[0] - 1) / (num_agents - 1), 0.5]) for i in range(num_agents)]
        else:
            self.agent_positions = [np.array([0.5, 0.5])]
        self.agents_trees = [TreeCollection() for _ in range(num_agents)]
        self.agents_obs_wp = [[] for _ in range(num_agents)]
        self.agents_measurements = [[] for _ in range(num_agents)]
        self.agents_full_path = [[] for _ in range(num_agents)]
        self.tree_nodes = [[] for _ in range(num_agents)]

    def initialize_trees(self, start_position, agent_idx):
        self.root = InformativeTreeNode(start_position)
        self.trees.add(self.root)
        self.current_position = start_position
        self.tree_nodes[agent_idx] = [self.root]

    def run(self):
        """
        Generic run method for the Informative RRT Path Planning algorithm
        """
        budget_portion = [budget / self.budget_iter for budget in self.budget]
        for i in range(self.num_agents):
            self.initialize_trees(self.agent_positions[i], i)
        with tqdm(total=sum(self.budget), desc="Running " + str(self.num_agents) + " Agent " + self.name) as pbar:
            while any(b > 0 for b in self.budget):
                for i in range(self.num_agents):
                    if self.budget[i] > 0:
                        self.tree_generation(budget_portion[i], i)
                        path = self.path_selection(i)
                        budget_spent = self.calculate_budget_spent(path)
                        self.update_observations_and_model(path, i)
                        self.agents_full_path[i].extend(path)
                        self.budget[i] -= budget_spent
                        pbar.update(budget_spent)
                        if path:
                            self.initialize_trees(path[-1], i)
                self.information_update()

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
        Z_pred, std = self.scenario.predict_spatial_field(self.obs_wp , np.array(self.measurements))
        return Z_pred, std
    
    def tree_generation(self, budget_portion, agent_idx):
        raise NotImplementedError("tree_generation method must be implemented in the subclass")
    
    def path_selection(self, agent_idx):
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
        rig_tree_generation(self, budget_portion, agent_idx)

    def path_selection(self, agent_idx):
        return informative_source_metric_path_selection(self, agent_idx)

    def information_update(self):
        source_metric_information_update(self)

class RRT_BiasBetaInformative_GP_PathPlanning(InformativeRRTBaseClass):
    def __init__(self, *args, directional_bias=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.directional_bias = directional_bias
        self.name = "RRT_BiasBetaInformative_GP_Path"

    def path_selection(self, agent_idx):
        return bias_beta_path_selection(self, agent_idx)


    def tree_generation(self, budget_portion, agent_idx):
        rrt_tree_generation(self, budget_portion, agent_idx)

    def information_update(self):
        gp_information_update(self)

        
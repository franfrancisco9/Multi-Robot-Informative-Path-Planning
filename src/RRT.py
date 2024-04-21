import numpy as np
from scipy.spatial import KDTree
from scipy.stats import uniform
from informative import BaseInformative
from tqdm import tqdm
from path_planning_utils import TreeNode, TreeCollection, estimate_sources_bayesian


class BaseRRTPathPlanning(BaseInformative):
    def __init__(self, *args, budget_iter=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.budget_iterations = budget_iter
        self.full_path = []
        self.measurements = []
        self.trees = TreeCollection()
        self.uncertainty_reduction = []
        self.name = "BaseRRTPath"

    def initialize_tree(self, start_position):
        self.root = TreeNode(start_position)
        self.current_position = start_position
        self.tree_nodes = [self.root]

    def node_selection_key(self, node, target_point):
        return np.linalg.norm(node.point - target_point)
    
    def generate_tree(self, budget_portion):
        distance_travelled = 0
        while distance_travelled < budget_portion:
            random_point = np.random.rand(2) * self.scenario.workspace_size
            nearest_node = min(self.tree_nodes, key=lambda node: self.node_selection_key(node, random_point))
            direction = random_point - nearest_node.point
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
            step_size = min(self.d_waypoint_distance, norm)
            new_point = nearest_node.point + direction * step_size

            if self.is_within_workspace(new_point):
                new_node = TreeNode(new_point, nearest_node)
                nearest_node.add_child(new_node)
                self.tree_nodes.append(new_node)
                distance_travelled += np.linalg.norm(new_point - nearest_node.point)
                if distance_travelled >= budget_portion:
                    break
        # add the tree to the list of trees
        self.trees.add(self.root)

    def update_observations_and_model(self, path):
        for point in path:
            measurement = self.scenario.simulate_measurements([point])[0]
            self.measurements.append(measurement)
            self.obs_wp.append(point)

        self.scenario.gp.fit(np.array(self.obs_wp), np.log10(self.measurements))

    def run(self):
        budget_portion = self.budget / self.budget_iterations
        start_position = np.array([0.5, 0.5])  # Initial start position
        self.initialize_tree(start_position)

        # use tdqm to show progress bar for budget iterations
        with tqdm(total=self.budget, desc="Running " + self.name) as pbar:
            while self.budget > 0:
                # print(f"Remaining budget: {self.budget}")
                self.generate_tree(budget_portion)
                path = self.select_path()
                # calcualte budget spent in path 
                budget_spent = 0
                for i in range(1, len(path)):
                    budget_spent += np.linalg.norm(path[i] - path[i-1])
                # print(f"Budget spent in path: {budget_spent}")
                # Only the first point of the path (closest to the root) should update the model
                # to simulate the agent moving along this path.
                self.update_observations_and_model(path)
                pbar.update(budget_spent)
                self.budget -= budget_spent
                self.full_path.extend(path)
                # The next tree starts from the end of the chosen path.
                if path:
                    self.initialize_tree(path[-1])
                else:
                    # If no path was selected (shouldn't happen in practice), break the loop to avoid infinite loop.
                    break
            
        self.obs_wp = np.array(self.obs_wp)
        self.full_path = np.array(self.full_path).reshape(-1, 2).T
        Z_pred, std = self.scenario.predict_spatial_field(self.obs_wp, np.array(self.measurements))
        return Z_pred, std

    def is_within_workspace(self, point):
        return np.all(point >= 0) & np.all(point <= self.scenario.workspace_size)
    
    def select_path(self):
        # Random path selection as the baseline behavior
        leaf_nodes = [node for node in self.tree_nodes if not node.children]
        if leaf_nodes:
            selected_leaf = np.random.choice(leaf_nodes)
            # Trace back to root from the selected leaf
            path = []
            current_node = selected_leaf
            while current_node is not None:
                path.append(current_node.point)
                current_node = current_node.parent
            path.reverse()  # Reverse to start from the root
            return path
        else:
            return []

class BaseRRTStarPathPlanning(BaseRRTPathPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "BaseRRTStarPath"
        self.near_radius = self.d_waypoint_distance  # Radius for finding nearby nodes for rewiring
    
    def generate_tree(self, budget_portion):
        distance_travelled = 0
        while distance_travelled < budget_portion:
            x_rand = self.sample_free()  # Random sampling within the workspace
            x_nearest = self.nearest(x_rand)  # Find the nearest node in the tree
            x_new = self.steer(x_nearest, x_rand)  # Steer from x_nearest towards x_rand
            
            if self.obstacle_free(x_nearest, x_new):
                X_near = self.near(x_new)  # Find nearby nodes for potential rewiring
                
                # Initialize the minimum cost and the best parent node for x_new
                c_min = self.cost(x_nearest) + self.line_cost(x_nearest, x_new)
                x_min = x_nearest
                
                # Try to find a better parent if any in the X_near set reduces the cost to reach x_new
                for x_near in X_near:
                    if self.obstacle_free(x_near, x_new) and \
                            self.cost(x_near) + self.line_cost(x_near, x_new) < c_min:
                        c_min = self.cost(x_near) + self.line_cost(x_near, x_new)
                        x_min = x_near
                
                # Add x_new to the tree with x_min as its parent
                self.add_node(x_new, x_min)
                distance_travelled += np.linalg.norm(x_new.point - x_min.point)
                
                # Rewiring the tree: Check if x_new is a better parent for nodes in X_near
                for x_near in X_near:
                    if self.obstacle_free(x_new, x_near) and \
                            self.cost(x_new) + self.line_cost(x_new, x_near) < self.cost(x_near):
                        x_parent = x_near.parent
                        self.rewire(x_near, x_new)
                
                if distance_travelled >= budget_portion:
                    break
        self.trees.add(self.root)  # Consider all nodes for plotting

    def sample_free(self):
        """Randomly samples a free point within the workspace."""
        return np.random.rand(2) * self.scenario.workspace_size

    def nearest(self, x_rand):
        """Finds the nearest node in the tree to the randomly sampled point."""
        return min(self.tree_nodes, key=lambda node: np.linalg.norm(node.point - x_rand))

    def steer(self, x_nearest, x_rand, step_size=1.0):
        """Steers from x_nearest towards x_rand."""
        direction = x_rand - x_nearest.point
        distance = np.linalg.norm(direction)
        direction = direction / distance if distance > 0 else direction
        new_point = x_nearest.point + min(step_size, distance) * direction
        return TreeNode(new_point)

    def obstacle_free(self, x1, x2):
        """Checks if the path between two nodes is free of obstacles."""
        # Implement obstacle checking logic here
        return True  # Placeholder implementation

    def near(self, x_new):
        """Finds nodes in the vicinity of x_new for potential rewiring."""
        # Implement logic to find nearby nodes within a certain radius
        return [node for node in self.tree_nodes if np.linalg.norm(node.point - x_new.point) < self.near_radius]

    def cost(self, node):
        """Calculates the cost to reach a given node from the root."""
        cost = 0
        while node.parent is not None:
            cost += np.linalg.norm(node.point - node.parent.point)
            node = node.parent
        return cost

    def line_cost(self, x1, x2):
        """Calculates the cost of a direct line between two nodes."""
        return np.linalg.norm(x1.point - x2.point)

    def add_node(self, x_new, x_parent):
        """Adds a new node to the tree."""
        x_parent.children.append(x_new)
        x_new.parent = x_parent
        self.tree_nodes.append(x_new)

    def rewire(self, x_near, x_new):
        """Rewires x_near to have x_new as its parent if it reduces the path cost."""
        x_near.parent.children.remove(x_near)
        x_near.parent = x_new
        x_new.children.append(x_near)

class BiasBetaRRTPathPlanning(BaseRRTPathPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "BiasBetaRRTPath"

    def select_path(self):
        """
        Overriding the method to consider both bias towards regions of high uncertainty and
        the beta_t parameter to manage the exploration-exploitation trade-off.
        """
        leaf_nodes = [node for node in self.tree_nodes if not node.children]
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


    def evaluate_directional_bias(self, leaf_points, improvement):
        """Evaluate directional scores for each leaf node based on current directional bias."""
        vectors_to_leafs = leaf_points - self.current_position
        unit_vectors = vectors_to_leafs / np.linalg.norm(vectors_to_leafs, axis=1, keepdims=True)
        # Calculate dot product between each unit vector and the directional bias
        scores = np.dot(unit_vectors, self.directional_bias)
        # Invert scores if the last direction was not an improvement
        if not improvement:
            scores = -scores
        return scores

    def update_directional_bias(self, path):
        """Update the directional bias based on the most recent path chosen."""
        if len(path) > 1:
            # Calculate the direction of the last path taken
            direction = np.array(path[-1]) - np.array(path[0])
            norm = np.linalg.norm(direction)
            if norm > 0:
                self.directional_bias = direction / norm
            else:
                self.directional_bias = None
        else:
            self.directional_bias = None

    def trace_path_to_root(self, selected_leaf):
        """Trace back the path from a selected leaf node to the root."""
        path = []
        current_node = selected_leaf
        while current_node:
            path.append(current_node.point)
            current_node = current_node.parent
        path.reverse()
        return path

class InformativeRRTPathPlanning(BaseRRTPathPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "InformativeRRTPath"
        # Override the node selection strategy
        self.node_selection_key = self.informative_node_selection_key

    def select_path(self):
        leaf_nodes = [node for node in self.tree_nodes if not node.children]
        leaf_points = np.array([node.point for node in leaf_nodes])
        _, stds = self.scenario.gp.predict(leaf_points, return_std=True)
        self.uncertainty_reduction.append(np.mean(stds))
        max_std_idx = np.argmax(stds)
        selected_leaf = leaf_nodes[max_std_idx]

        # Trace back to root from the selected leaf
        path = []
        current_node = selected_leaf
        
        while current_node is not None:
            path.append(current_node.point)
            current_node = current_node.parent
        path.reverse()  # Reverse to start from root
        return path 
    
    def informative_node_selection_key(self, node, random_point):
        """Key function for selecting nodes based on predicted mu values."""
        # This example uses predicted mu values as the key
        mu, std = self.scenario.gp.predict(np.array([node.point]), return_std=True)
        if np.std(mu) == 0:
            mu_normalized = mu
        else:
            mu_normalized = (mu - np.mean(mu)) / np.std(mu)
        if np.std(std) == 0:
            std_normalized = std
        else:
            std_normalized = (std - np.mean(std)) / np.std(std)
        value = mu_normalized + self.beta_t * std_normalized
        return -value

class InformativeRRTStarPathPlanning(BaseRRTStarPathPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "InformativeRRTStarPath"
        # Override the node selection strategy
        self.node_selection_key = self.informative_node_selection_key

    def informative_node_selection_key(self, node, random_point):
        """Key function for selecting nodes based on predicted mu values."""
        node_selection = InformativeRRTPathPlanning.informative_node_selection_key
        return node_selection(self, node, random_point)
    
    def select_path(self):
        path_selection = InformativeRRTPathPlanning.select_path
        return path_selection(self)
    
class InformativeTreeNode(TreeNode):
    def __init__(self, point, parent=None):
        super().__init__(point, parent)
        self.information = 0  # Initialize the information gain attribute

class InformativeSourceMetricRRTPathPlanning(BaseInformative):
    def __init__(self, *args, budget_iter=10, lambda_b=1, max_sources=1, n_samples=20, s_stages=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.budget_iterations = budget_iter
        self.lambda_b = lambda_b
        self.max_sources = max_sources
        self.n_samples = n_samples
        self.s_stages = s_stages
        self.best_estimates = np.array([])
        self.best_bic = -np.inf
        self.name = "InformativeSourceMetricRRTPath"
        self.measurements = []
        self.full_path = []
        self.trees = TreeCollection()
        self.uncertainty_reduction = []
        
    def generate_rig_tree(self, budget_portion):
        distance_travelled = 0
        while distance_travelled < budget_portion:
            random_point = np.random.rand(2) * self.scenario.workspace_size
            nearest_node = self.nearest(random_point)
            new_node = self.steer(nearest_node, random_point)

            if self.obstacle_free(nearest_node.point, new_node.point):
                X_near = self.near(new_node)
                new_node = self.choose_parent(X_near, nearest_node, new_node)
                nearest_node.children.append(new_node)
                self.tree_nodes.append(new_node)
                distance_travelled += np.linalg.norm(new_node.point - nearest_node.point)
                
                # Update the information gain for the new node
                new_node.information = self.radiation_gain(new_node)

                self.rewire(X_near, new_node)
                if distance_travelled >= budget_portion:
                    break

        self.trees.add(self.root)
    
    def steer(self, nearest_node, target_point, step_size=1.0):
        direction = target_point - nearest_node.point
        distance = np.linalg.norm(direction)
        direction /= max(distance, 1e-8)  # Avoid division by zero
        new_point = nearest_node.point + direction * min(distance, step_size)
        return InformativeTreeNode(new_point)

    def nearest(self, target_point):
        return min(self.tree_nodes, key=lambda node: np.linalg.norm(node.point - target_point))

    def near(self, new_node):
        return [node for node in self.tree_nodes if np.linalg.norm(node.point - new_node.point) < self.d_waypoint_distance]

    def choose_parent(self, X_near, x_nearest, x_new):
        c_min = self.cost(x_nearest) + self.line_cost(x_nearest, x_new)
        x_min = x_nearest

        for x_near in X_near:
            if self.obstacle_free(x_near.point, x_new.point) and \
                    self.cost(x_near) + self.line_cost(x_near, x_new) < c_min:
                c_min = self.cost(x_near) + self.line_cost(x_near, x_new)
                x_min = x_near

        x_new.parent = x_min
        return x_new

    def rewire(self, X_near, x_new):
        for x_near in X_near:
            if self.obstacle_free(x_new.point, x_near.point) and \
                    self.cost(x_new) + self.line_cost(x_new, x_near) < self.cost(x_near):
                x_near.parent = x_new

    def obstacle_free(self, start, end):
        return True  # Simplify for now

    def cost(self, node):
        cost = 0
        while node.parent:
            cost += np.linalg.norm(node.point - node.parent.point)
            node = node.parent
        return cost

    def line_cost(self, x1, x2):
        return np.linalg.norm(x1.point - x2.point)
    
    def initialize_tree(self, start_position):
        self.root = InformativeTreeNode(start_position)
        self.current_position = start_position
        self.tree_nodes = [self.root]

    def run(self):
        budget_portion = self.budget / self.budget_iterations
        # Initialize with the starting position
        start_position = np.array([0.5, 0.5])
        self.initialize_tree(start_position)
        with tqdm(total=self.budget, desc="Running " + self.name) as pbar:
            while self.budget > 0:
                self.generate_rig_tree(budget_portion)
                path = self.select_path()
                # print(f"Path: {path}")
                budget_spent = 0
                for i in range(1, len(path)):
                    budget_spent += np.linalg.norm(path[i] - path[i-1])
                self.update_observations_and_model(path)
                self.full_path.extend(path)
                self.budget -= budget_spent
                pbar.update(budget_spent)
                #print(f"Remaining budget: {self.budget}")
                if path:
                    # Reinitialize tree at the last point of the current path
                    self.initialize_tree(path[-1])
                else:
                    break  # If no path is generated, exit the loop
                # evey 3 budget iterations, estimate sources
                self.estimate_sources()
            
            return self.finalize()
        
    def select_path(self):
        leaf_nodes = [node for node in self.tree_nodes if not node.children]
        if leaf_nodes:
            # Select the leaf with the highest information gain
            selected_leaf = max(leaf_nodes, key=lambda node: node.information)
            # test random selection
            # selected_leaf = np.random.choice(leaf_nodes)
            return self.trace_path_to_root(selected_leaf)
        else:
            return []
    
    def select_leaf_based_on_gain(self, leaf_nodes):
        if self.best_estimates.size == 0:
            return np.random.choice(leaf_nodes)
        radiation_gains = [self.radiation_gain(node) for node in leaf_nodes]
        return leaf_nodes[np.argmax(radiation_gains)]

    def trace_path_to_root(self, selected_leaf):
        path = []
        current_node = selected_leaf
        while current_node:
            path.append(current_node.point)
            current_node = current_node.parent
        path.reverse()
        return path

    def update_observations_and_model(self, path):
        for point in path:
            self.obs_wp.append(point)
            measurement = self.scenario.simulate_measurements([point])[0]
            self.measurements.append(measurement)

    def estimate_sources(self):
        estimates, _, bic = estimate_sources_bayesian(
            self.full_path, self.measurements, self.lambda_b,
            self.max_sources, self.n_samples, self.s_stages
        )
        if bic > self.best_bic:
            self.best_bic = bic
            self.best_estimates = estimates.reshape((-1, 3))

    def finalize(self):
        self.obs_wp = np.array(self.obs_wp)
        self.full_path = np.array(self.obs_wp).reshape(-1, 2).T
        Z_pred, std = self.scenario.predict_spatial_field(self.obs_wp , np.array(self.measurements))
        return Z_pred, std

    def radiation_gain(self, node):
        # Gain_nodet = Gain_nodet-1 + Gain_nodet_src
        def sources_gain(node):
            x_t, y_t = node.point
            radiation_gain = 0
            for source in self.best_estimates:
                x_k, y_k, intensity = source
                d_src = np.linalg.norm([x_t - x_k, y_t - y_k])
                radiation_gain += intensity / d_src**2
            suprresion_gain = 0
            # for source k 
            # F_src(t,k) (node_t) = sum_i,j!=k ^N_sources (1- C_sup^dist + (C_sup^dist)/(1+exp[(T_sup^dist-d_t(k,j))/(S_sup^dist)]))
            # where the d_t(k,j) is the distance between the node_t and midpoint of the kth and jth sources (second norm)
            # C_k^pos and C_j^pos used in distance calculation are the positions of the sources
            # T_sup^dist and S_sup^dist are the threshold and slope parameters
            # C_sup^dist is the suppression constant (lets assume 1 for now)
            for j in range(len(self.best_estimates)):
                for k in range(len(self.best_estimates)):
                    if j != k:
                        x_j, y_j, _ = self.best_estimates[j]
                        x_k, y_k, _ = self.best_estimates[k]
                        d_t_k_j = np.linalg.norm([x_t - (x_j + x_k)/2, y_t - (y_j + y_k)/2])
                        suprresion_gain += 1 - 1/(1+np.exp((d_t_k_j - 1)/1))
                        
            return radiation_gain * suprresion_gain
        final_gain = 0
        current_node = node
        while current_node.parent:
            final_gain += sources_gain(current_node)
            current_node = current_node.parent
        return final_gain

        # x_t, y_t = node.point
        # radiation_gain = 0
        # for source in self.best_estimates:
        #     x_k, y_k, intensity = source
        #     d_src = np.linalg.norm([x_t - x_k, y_t - y_k])
        #     radiation_gain += intensity / d_src**2  + self.beta_t * d_src
        #     # print(f"Radiation gain after beta: {radiation_gain}")
        # if radiation_gain < 0:
        #     radiation_gain = 0

        return radiation_gain
   
class InformativeSourceMetricRRTStartPathPlanning(BaseRRTStarPathPlanning):# F_
    def __init__(self, *args, budget_iter=10, lambda_b=1, max_sources=1, n_samples=20, s_stages=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.budget_iterations = budget_iter
        self.lambda_b = lambda_b
        self.max_sources = max_sources
        self.n_samples = n_samples
        self.s_stages = s_stages
        self.best_estimates = np.array([])
        self.best_bic = -np.inf
        self.name = "InformativeSourceMetricRRTStartPath"
        self.measurements = []
        self.full_path = []
        self.trees = TreeCollection()
        self.uncertainty_reduction = []
        
    def run(self):
        budget_portion = self.budget / self.budget_iterations
        # Initialize with the starting position
        start_position = np.array([0.5, 0.5])
        self.initialize_tree(start_position)
        with tqdm(total=self.budget, desc="Running " + self.name) as pbar:
            while self.budget > 0:
                self.generate_tree(budget_portion)
                path = self.select_path()
                # print(f"Path: {path}")
                budget_spent = 0
                for i in range(1, len(path)):
                    budget_spent += np.linalg.norm(path[i] - path[i-1])
                self.update_observations_and_model(path)
                self.full_path.extend(path)
                self.budget -= budget_spent
                pbar.update(budget_spent)
                #print(f"Remaining budget: {self.budget}")
                if path:
                    # Reinitialize tree at the last point of the current path
                    self.initialize_tree(path[-1])
                else:
                    break  # If no path is generated, exit the loop
                # evey two budget iterations, estimate sources
                
                self.estimate_sources()
            
            return self.finalize()
        
    def select_path(self):
        leaf_nodes = [node for node in self.tree_nodes if not node.children]
        leaf_points = np.array([node.point for node in leaf_nodes])
        
        # select using the best gain 
        if self.best_estimates.size == 0:
            selected_leaf = np.random.choice(leaf_nodes)
        else:
            radiation_gains = [self.radiation_gain(node) for node in leaf_nodes]
            selected_leaf = leaf_nodes[np.argmax(radiation_gains)]

        # Trace back to root from the selected leaf
        path = []
        path = self.trace_path_to_root(selected_leaf)
        return path


    def trace_path_to_root(self, selected_leaf):
        path = []
        current_node = selected_leaf
        while current_node:
            path.append(current_node.point)
            current_node = current_node.parent
        path.reverse()
        return path

    def update_observations_and_model(self, path):
        for point in path:
            self.obs_wp.append(point)
            measurement = self.scenario.simulate_measurements([point])[0]
            self.measurements.append(measurement)

    def estimate_sources(self):
        estimates, _, bic = estimate_sources_bayesian(
            self.full_path, self.measurements, self.lambda_b,
            self.max_sources, self.n_samples, self.s_stages
        )
        if bic > self.best_bic:
            self.best_bic = bic
            self.best_estimates = estimates.reshape((-1, 3))

    def finalize(self):
        self.obs_wp = np.array(self.obs_wp)
        self.full_path = np.array(self.obs_wp).reshape(-1, 2).T
        Z_pred, std = self.scenario.predict_spatial_field(self.obs_wp , np.array(self.measurements))
        return Z_pred, std

    def radiation_gain(self, node):
        # Gain_nodet = Gain_nodet-1 + Gain_nodet_src
        def sources_gain(node):
            x_t, y_t = node.point
            radiation_gain = 0
            for source in self.best_estimates:
                x_k, y_k, intensity = source
                d_src = np.linalg.norm([x_t - x_k, y_t - y_k])
                radiation_gain += intensity / d_src**2
            return radiation_gain
        # give a severe penalty to a node that is within 5m of an observation point of the other agent
        # if node in self.agents_obs_wp[0]:

        final_gain = 0
        current_node = node
        while current_node.parent:
            final_gain += sources_gain(current_node)
            current_node = current_node.parent
        return final_gain

        # x_t, y_t = node.point
        # radiation_gain = 0
        # for source in self.best_estimates:
        #     x_k, y_k, intensity = source
        #     d_src = np.linalg.norm([x_t - x_k, y_t - y_k])
        #     radiation_gain += intensity / d_src**2  + self.beta_t * d_src
        #     # print(f"Radiation gain after beta: {radiation_gain}")
        # if radiation_gain < 0:
        #     radiation_gain = 0

        return radiation_gain
   
class MultiAgentInformativeSourceMetricRRTPathPlanning(InformativeSourceMetricRRTPathPlanning):
    def __init__(self, *args, num_agents=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_agents = num_agents
        self.individual_budgets = [self.budget] * num_agents  # Can be a list if different budgets per agent
        self.agent_positions = [np.array([0.5 + i * (self.scenario.workspace_size[0] - 1) / (num_agents - 1), 0.5]) for i in range(num_agents)]
        self.agents_trees = [TreeCollection() for _ in range(num_agents)]
        self.agents_obs_wp = [[] for _ in range(num_agents)]
        self.agents_measurements = [[] for _ in range(num_agents)]
        self.agents_full_path = [[] for _ in range(num_agents)]
        self.tree_nodes = [[] for _ in range(num_agents)]
        self.name = "MultiAgentInformativeSourceMetricRRTPath"

    def initialize_trees(self, start_position, agent_idx):
        self.root = InformativeTreeNode(start_position)
        self.trees.add(self.root)
        self.current_position = start_position
        self.tree_nodes[agent_idx] = [self.root]

    def run(self):
        budget_portion = [budget / self.budget_iterations for budget in self.individual_budgets]
        for i in range(self.num_agents):
            self.initialize_trees(self.agent_positions[i], i)
        with tqdm(total=sum(self.individual_budgets), desc="Running Multi-Agent " + self.name) as pbar:
            while any(b > 0 for b in self.individual_budgets):
                for i in range(self.num_agents):
                    if self.individual_budgets[i] > 0:
                        self.generate_rig_tree(budget_portion[i], i)
                        path = self.select_path(i)
                        budget_spent = self.calculate_budget_spent(path)
                        self.update_observations_and_model(path, i)
                        self.agents_full_path[i].extend(path)
                        self.individual_budgets[i] -= budget_spent
                        pbar.update(budget_spent)
                        if path:
                            self.initialize_trees(path[-1], i)
                print(f"Remaining budgets: {self.individual_budgets}")
                self.estimate_sources()

        return self.finalize_all_agents()

    def finalize_all_agents(self):
        combined_measurements = sum((agent_measurements for agent_measurements in self.agents_measurements), [])
        combined_obs_wp = sum((agent_wp for agent_wp in self.agents_obs_wp), [])
        combined_full_path = sum((agent_path for agent_path in self.agents_full_path), [])
        # print("Finalizing all agents")
        # print(f"Combined measurements: {combined_measurements}")
        # print(f"Combined obs_wp: {combined_obs_wp}")
        # print(f"Combined full_path: {combined_full_path}")
        return self.finalize(combined_measurements, combined_obs_wp, combined_full_path)
    
    def finalize(self, measurements, obs_wp, full_path):
        self.measurements = measurements
        self.obs_wp = np.array(obs_wp)
        self.full_path = np.array(full_path).reshape(-1, 2).T
        Z_pred, std = self.scenario.predict_spatial_field(self.obs_wp , np.array(self.measurements))
        print(self.trees)
        return Z_pred, std
    
    def radiation_gain(self, node, agent_idx):
        # Gain_nodet = Gain_nodet-1 + Gain_nodet_src
        def sources_gain(node):
            x_t, y_t = node.point
            radiation_gain = 0
            for source in self.best_estimates:
                x_k, y_k, intensity = source
                d_src = np.linalg.norm([x_t - x_k, y_t - y_k])
                radiation_gain += intensity / d_src**2
            return radiation_gain
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
    
    def calculate_budget_spent(self, path):
        if not path:
            return 0
        path_array = np.array(path)
        if path_array.ndim > 1:
            return np.sum(np.linalg.norm(np.diff(path_array, axis=0), axis=1))
        return 0

    def generate_rig_tree(self, budget_portion, agent_idx):
        distance_travelled = 0
        while distance_travelled < budget_portion:
            random_point = np.random.rand(2) * self.scenario.workspace_size
            nearest_node = self.nearest(random_point, agent_idx)
            new_node = self.steer(nearest_node, random_point)

            if self.obstacle_free(nearest_node.point, new_node.point):
                X_near = self.near(new_node, agent_idx)
                new_node = self.choose_parent(X_near, nearest_node, new_node, agent_idx)
                nearest_node.children.append(new_node)
                self.tree_nodes[agent_idx].append(new_node)
                distance_travelled += np.linalg.norm(new_node.point - nearest_node.point)
                
                # Update the information gain for the new node
                new_node.information = self.radiation_gain(new_node, agent_idx)

                self.rewire(X_near, new_node, agent_idx)
                if distance_travelled >= budget_portion:
                    break

        self.agents_trees[agent_idx].add(self.root)

    def choose_parent(self, X_near, x_nearest, x_new, agent_idx):
        c_min = self.cost(x_nearest, agent_idx) + self.line_cost(x_nearest, x_new)
        x_min = x_nearest

        for x_near in X_near:
            if self.obstacle_free(x_near.point, x_new.point) and \
                    self.cost(x_near, agent_idx) + self.line_cost(x_near, x_new) < c_min and \
                    self.radiation_gain(x_near, agent_idx) > 0:
                c_min = self.cost(x_near, agent_idx) + self.line_cost(x_near, x_new)
                x_min = x_near

        x_new.parent = x_min
        return x_new
    
    def steer(self, x_nearest, x_rand, step_size=1.0):
        direction = x_rand - x_nearest.point
        distance = np.linalg.norm(direction)
        direction = direction / distance if distance > 0 else direction
        new_point = x_nearest.point + min(step_size, distance) * direction
        return TreeNode(new_point)

    def nearest(self, x_rand, agent_idx):
        return min(self.tree_nodes[agent_idx], key=lambda node: np.linalg.norm(node.point - x_rand))

    def near(self, x_new, agent_idx):
        return [node for node in self.tree_nodes[agent_idx] if np.linalg.norm(node.point - x_new.point) < self.d_waypoint_distance]

    def add_node(self, x_new, x_parent, agent_idx):
        x_parent.children.append(x_new)
        x_new.parent = x_parent
        self.tree_nodes[agent_idx].append(x_new)

    def rewire(self, x_near, x_new, agent_idx):
        for x_near in x_near:
            if self.obstacle_free(x_new.point, x_near.point) and \
                    self.cost(x_new, agent_idx) + self.line_cost(x_new, x_near) < self.cost(x_near, agent_idx):
                x_near.parent = x_new
    
    def obstacle_free(self, x1, x2):
        # Add your actual obstacle check logic here
        return True

    def cost(self, node, agent_idx):
        cost = 0
        while node.parent is not None:
            cost += np.linalg.norm(node.point - node.parent.point)
            node = node.parent
        return cost

    def line_cost(self, x1, x2):
        return np.linalg.norm(x1.point - x2.point)

    def select_path_and_update_model(self, agent_idx):
        path = self.select_path(agent_idx)
        if path:
            self.update_observations_and_model(path, agent_idx)
        return path

    def select_path(self, agent_idx):
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
        return self.trace_path_to_root(selected_leaf)

    def trace_path_to_root(self, selected_leaf):
        path = []
        current_node = selected_leaf
        while current_node:
            path.append(current_node.point)
            current_node = current_node.parent
        path.reverse()
        return path
    
    def update_observations_and_model(self, path, agent_idx):
        for point in path:
            measurement = self.scenario.simulate_measurements([point])[0]
            self.agents_measurements[agent_idx].append(measurement)
            self.agents_obs_wp[agent_idx].append(point)

    def combine_observations_and_paths(self):
        self.obs_wp = [wp for sublist in self.agents_obs_wp for wp in sublist]
        self.full_path = [path for sublist in self.agents_full_path for path in sublist]

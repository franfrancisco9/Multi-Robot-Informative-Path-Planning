import numpy as np
from scipy.spatial import KDTree
from informative import BaseInformative
from tqdm import tqdm
from path_planning_utils import TreeNode, TreeCollection


class BaseRRTPathPlanning(BaseInformative):
    def __init__(self, *args, budget_iter=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.budget_iterations = budget_iter
        self.full_path = []
        self.observations = []
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
            self.observations.append(measurement)
            self.obs_wp.append(point)

        self.scenario.gp.fit(np.array(self.obs_wp), np.log10(self.observations))

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

                # Only the first point of the path (closest to the root) should update the model
                # to simulate the agent moving along this path.
                self.update_observations_and_model(path)
                pbar.update(budget_portion)
                self.budget -= budget_portion
                self.full_path.extend(path)
                # The next tree starts from the end of the chosen path.
                if path:
                    self.initialize_tree(path[-1])
                else:
                    # If no path was selected (shouldn't happen in practice), break the loop to avoid infinite loop.
                    break
            
        self.obs_wp = np.array(self.obs_wp)
        self.full_path = np.array(self.full_path).reshape(-1, 2).T
        Z_pred, std = self.scenario.predict_spatial_field(self.obs_wp, np.array(self.observations))
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

class StrategicRRTPathPlanning(BaseRRTPathPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "StrategicRRTPath"

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
    
class BiasRRTPathPlanning(BaseRRTPathPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "BiasRRTPath"

    def select_path(self):
        leaf_nodes = [node for node in self.tree_nodes if not node.children]
        leaf_points = np.array([node.point for node in leaf_nodes])
        mus, std = self.scenario.gp.predict(leaf_points, return_std=True)
        self.uncertainty_reduction.append(np.mean(std))
        min_mu_idx = np.argmax(mus)  # Choose the leaf with the minimum expected mean (seeking sources)
        selected_leaf = leaf_nodes[min_mu_idx]

        path = []
        current_node = selected_leaf
        while current_node is not None:
            path.append(current_node.point)
            current_node = current_node.parent
        path.reverse()
        return path   
       
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

class AdaptiveRRTPathPlanning(BaseRRTPathPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "AdaptiveRRTPath"
        # Directional bias is initialized as None; it will be a unit vector pointing in the preferred direction
        self.directional_bias = None
        self.last_uncertainty = np.inf

    def select_path(self):
        # Obtain all leaf nodes and their associated points
        leaf_nodes = [node for node in self.tree_nodes if not node.children]
        leaf_points = np.array([node.point for node in leaf_nodes])
        
        # Predict standard deviations for leaf points
        _, stds = self.scenario.gp.predict(leaf_points, return_std=True)
        mean_std = np.mean(stds)
        
        # Update uncertainty reduction history
        self.uncertainty_reduction.append(mean_std)
        
        # Determine if the new direction is better or worse
        if mean_std < self.last_uncertainty:
            improvement = True
        else:
            improvement = False
        self.last_uncertainty = mean_std

        # Apply directional bias to leaf node selection if it exists
        if self.directional_bias is not None and leaf_nodes:
            directional_scores = self.evaluate_directional_bias(leaf_points, improvement)
            selected_idx = np.argmax(directional_scores)
        else:
            selected_idx = np.argmax(stds)  # Default behavior without bias
        
        selected_leaf = leaf_nodes[selected_idx]

        # Trace back to root from the selected leaf to form a path
        path = self.trace_path_to_root(selected_leaf)
        
        # Update the directional bias based on the chosen path
        self.update_directional_bias(path)
        
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

class InformativeRRTPathPlanning(StrategicRRTPathPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "InformativeRRTPath"
        # Override the node selection strategy
        self.node_selection_key = self.informative_node_selection_key
    
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
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from informative import InformativePathPlanning
from tqdm import tqdm
from path_planning_utils import TreeNode, TreeCollection

class StrategicRRTPathPlanning(InformativePathPlanning):
    def __init__(self, *args, budget_iter=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "StrategicRRTPath"
        self.full_path = []
        self.root = None
        self.observations = []
        self.trees = TreeCollection()
        self.uncertainty_reduction = []
        self.budget_iterations = budget_iter

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

    def select_path_with_highest_uncertainty(self):
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
                path = self.select_path_with_highest_uncertainty()

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

class NaiveRRTPathPlanning(StrategicRRTPathPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "NaiveRRTPath"

    def select_path_with_highest_uncertainty(self):
        leaf_nodes = [node for node in self.tree_nodes if not node.children]
        leaf_points = np.array([node.point for node in leaf_nodes])
        _, stds = self.scenario.gp.predict(leaf_points, return_std=True)
        self.uncertainty_reduction.append(np.mean(stds))
        if leaf_nodes:
            selected_leaf = np.random.choice(leaf_nodes)
            # Trace back to root from the selected leaf
            path = []
            current_node = selected_leaf
            while current_node is not None:
                path.append(current_node.point)
                current_node = current_node.parent
            path.reverse()  # Reverse to start from root
            return path
        else:
            return []

class BiasRRTPathPlanning(StrategicRRTPathPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "BiasRRTPath"

    def select_path_with_highest_uncertainty(self):
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
       
class BiasBetaRRTPathPlanning(StrategicRRTPathPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "BiasBetaRRTPath"

    def select_path_with_highest_uncertainty(self):
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

class AdaptiveRRTPathPlanning(StrategicRRTPathPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "AdaptiveRRTPath"
        # Directional bias is initialized as None; it will be a unit vector pointing in the preferred direction
        self.directional_bias = None
        self.last_uncertainty = np.inf

    def select_path_with_highest_uncertainty(self):
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



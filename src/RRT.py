import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from informative import InformativePathPlanning
from path_planning_utils import plot_tree
from tqdm import tqdm

class RRTPathPlanning(InformativePathPlanning):
    """
    Implements the Rapidly-exploring Random Tree (RRT) algorithm for path planning within a specified budget.
    Inherits from InformativePathPlanning to utilize the scenario setup and budget management.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "RRTPath"

    def initialize_tree(self):
        """
        Initializes the RRT tree with the starting position.
        """
        initial_position = np.array([0.5, 0.5])  # Start at the center or a predefined start point
        self.tree = KDTree([initial_position])
        self.obs_wp = [initial_position]
        # Initial observation
        initial_observation = self.scenario.simulate_measurements([initial_position])[0]
        self.observations = [initial_observation]

    def expand_tree(self):
        """
        Expands the RRT tree by adding new points until the budget is reached.
        """
        distance_travelled = 0
        while distance_travelled < self.budget:
            random_point = np.random.rand(2) * self.scenario.workspace_size
            nearest_idx = self.tree.query([random_point], k=1)[1][0]
            nearest_point = self.tree.data[nearest_idx]

            # Calculate direction and adjust for max step size
            direction = random_point - nearest_point
            norm = np.linalg.norm(direction)
            norm_direction = direction / norm if norm > 0 else direction
            step_size = min(self.d_waypoint_distance, norm)
            new_point = nearest_point + norm_direction * step_size

            if self.is_within_workspace(new_point):
                # Calculate actual distance travelled for this step
                step_distance = np.linalg.norm(new_point - self.obs_wp[-1])
                if distance_travelled + step_distance > self.budget:
                    # If this step exceeds the budget, break the loop
                    break
                distance_travelled += step_distance
                
                new_measurement = self.scenario.simulate_measurements([new_point])[0]
                self.tree = KDTree(np.vstack([self.tree.data, new_point]))  # Rebuild the tree
                self.obs_wp.append(new_point)
                self.observations.append(new_measurement)
        self.obs_wp = np.array(self.obs_wp)
        self.full_path = np.array(self.obs_wp).reshape(-1, 2).T
        plot_tree(self.scenario, self.tree)

    def run(self):
        """
        Executes the RRT path planning.
        """
        self.initialize_tree()
        self.expand_tree()
        waypoints = np.array(self.obs_wp)
        measurements = np.array(self.observations)
        self.scenario.gp.fit(waypoints, np.log10(measurements))
        Z_pred, std = self.scenario.predict_spatial_field(waypoints, measurements)
        return Z_pred, std

    def is_within_workspace(self, point):
        """
        Check if a point is within the boundaries of the workspace.
        """
        return np.all(point >= 0) and np.all(point <= self.scenario.workspace_size)

class BiasInformativeRRTPathPlanning(InformativePathPlanning):
    """
    This class extends InformativePathPlanning with a bias towards areas of high uncertainty,
    using a Rapidly-exploring Random Tree (RRT) approach within a specified budget.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "BiasInformativeRRTPath"
        # Initialize tree in the superclass constructor

    def initialize_tree(self):
        """
        Initialize the RRT tree with a starting position and simulate the initial measurement.
        """
        initial_position = np.array([[0.5, 0.5]])
        self.tree = KDTree(initial_position)
        self.obs_wp = [initial_position[0].tolist()]
        initial_measurement = self.scenario.simulate_measurements(initial_position)[0]
        self.observations = {str(initial_position[0]): initial_measurement}

    def generate_random_point_with_bias(self):
        """
        Generate a random point with a bias towards high uncertainty areas
        based on the current Gaussian Process model.
        """
        _, sigma = self.scenario.gp.predict(self.grid, return_std=True)
        probabilities = sigma / sigma.sum()
        index = np.random.choice(len(self.grid), p=probabilities)
        return self.grid[index]

    def expand_tree(self):
        """
        Expand the RRT tree towards areas of high information gain,
        considering the budget for path planning.
        """
        distance_travelled = 0

        while distance_travelled < self.budget:
            random_point = self.generate_random_point_with_bias()
            nearest_idx = self.tree.query([random_point], k=1)[1]
            nearest_point = self.tree.data[nearest_idx][0]

            direction = random_point - nearest_point
            direction /= np.linalg.norm(direction)
            new_point = nearest_point + direction * self.d_waypoint_distance
            # Calculate potential new distance travelled
            potential_new_distance = np.linalg.norm(new_point - self.obs_wp[-1])

            if (np.all(new_point >= 0) and np.all(new_point <= self.scenario.workspace_size) and
                    potential_new_distance <= self.budget):
                self.tree = KDTree(np.vstack([self.tree.data, new_point]))
                new_measurement = self.scenario.simulate_measurements(np.array([new_point]))[0]
                self.observations[str(new_point)] = new_measurement
                self.obs_wp.append(new_point.tolist())
                distance_travelled += potential_new_distance

        self.obs_wp = np.array(self.obs_wp)
        self.full_path = np.array(self.obs_wp).reshape(-1, 2).T
        plot_tree(self.scenario, self.tree)
    
    def run(self):
        """
        Execute the path planning process with bias towards high uncertainty areas
        and return the predicted spatial field based on the explored path.
        """
        self.initialize_tree()
        self.expand_tree()
        measurements = np.array(list(self.observations.values()))
        self.scenario.gp.fit(np.array(self.obs_wp), np.log10(measurements))
        return self.scenario.predict_spatial_field(np.array(self.obs_wp), measurements)
    
class BetaInformativeRRTPathPlanning(InformativePathPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree = None
        self.name = "BetaInformativeRRTPath"

    def initialize_tree(self):
        initial_position = np.array([[0, 0]])  # Starting position
        self.tree = KDTree(initial_position)
        self.observations = self.scenario.simulate_measurements(initial_position)
        self.scenario.gp.fit(initial_position, np.log10(self.observations))

    def generate_random_point_with_bias_and_beta(self):
        # Generate a grid of points across the workspace
        x = np.linspace(0, self.scenario.workspace_size[0], 100)
        y = np.linspace(0, self.scenario.workspace_size[1], 100)
        X, Y = np.meshgrid(x, y)
        grid = np.vstack([X.ravel(), Y.ravel()]).T
        
        # Predict mean and variance across the grid
        mu, sigma = self.scenario.gp.predict(grid, return_std=True)
        
        # Adjust acquisition by beta term
        acquisition_values = mu + self.beta_t * sigma
        
        # Normalize to get probabilities
        probabilities = np.exp(acquisition_values - np.max(acquisition_values))  # Subtract max for numerical stability
        probabilities /= np.sum(probabilities)
        
        # Choose a random index based on adjusted acquisition values
        index = np.random.choice(range(len(grid)), p=probabilities)
        
        return grid[index]

    def expand_tree(self):
        for _ in range(self.n_waypoints - 1):
            random_point = self.generate_random_point_with_bias_and_beta()
            nearest_idx = self.tree.query([random_point], k=1)[1][0]
            nearest_point = self.tree.data[nearest_idx]

            direction = random_point - nearest_point
            direction /= np.linalg.norm(direction)
            new_point = nearest_point + direction * self.d_waypoint_distance

            if np.all(new_point >= 0) and np.all(new_point <= self.scenario.workspace_size):
                self.tree = KDTree(np.vstack([self.tree.data, new_point]))
                new_measurement = self.scenario.simulate_measurements(np.array([new_point]))[0]
                self.observations = np.append(self.observations, new_measurement)
                self.scenario.gp.fit(self.tree.data, np.log10(self.observations))

        self.obs_wp = self.tree.data
        self.full_path = self.obs_wp.reshape(-1, 2).T
        plot_tree

    def run(self):
        self.initialize_tree()
        self.expand_tree()
        self.scenario.gp.fit(self.obs_wp, np.log10(self.observations))
        Z_pred, std = self.scenario.predict_spatial_field(self.obs_wp, self.observations)
        return Z_pred, std

class StrategicRRTPathPlanning(InformativePathPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "StrategicRRTPath"
        self.full_path = []
        self.root = None
        self.observations = []
        self.trees = TreeCollection()
        self.uncertainty_reduction = []

    def initialize_tree(self, start_position):
        self.root = TreeNode(start_position)
        self.current_position = start_position
        self.tree_nodes = [self.root]

    def generate_tree(self, budget_portion):
        distance_travelled = 0
        while distance_travelled < budget_portion:
            random_point = np.random.rand(2) * self.scenario.workspace_size
            nearest_node = min(self.tree_nodes, key=lambda node: np.linalg.norm(node.point - random_point))
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
        budget_portion = self.budget / 20
        start_position = np.array([0.5, 0.5])  # Initial start position
        self.initialize_tree(start_position)

        # use tdqm to show progress bar for budget iterations
        with tqdm(total=self.budget, desc="Running RRT Path Planning") as pbar:
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
                # plot_iteration(self.root, path, self.budget, self.scenario.workspace_size)
                # self.plot_uncertainty_reduction()
                # The next tree starts from the end of the chosen path.
                if path:
                    self.initialize_tree(path[-1])
                else:
                    # If no path was selected (shouldn't happen in practice), break the loop to avoid infinite loop.
                    break
            
        plot_final(self.trees, self.full_path, self.scenario.workspace_size)
        self.plot_uncertainty_reduction()
        self.obs_wp = np.array(self.obs_wp)
        self.full_path = np.array(self.full_path).reshape(-1, 2).T
        Z_pred, std = self.scenario.predict_spatial_field(self.obs_wp, np.array(self.observations))
        return Z_pred, std

    def is_within_workspace(self, point):
        return np.all(point >= 0) & np.all(point <= self.scenario.workspace_size)

    def plot_uncertainty_reduction(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.uncertainty_reduction, marker='o')
        plt.title('Reduction in Uncertainty Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Average Uncertainty (std)')
        plt.grid(True)
        plt.show()

class NaiveRRTPathPlanning(StrategicRRTPathPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "NaiveRRTPath"

    def select_path_with_highest_uncertainty(self):
        leaf_nodes = [node for node in self.tree_nodes if not node.children]
        _, stds = self.scenario.gp.predict(np.array([node.point for node in leaf_nodes]), return_std=True)
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
        mus, std = self.scenario.gp.predict(leaf_points)
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
     
class TreeNode:
    def __init__(self, point, parent=None):
        self.point = point
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

class TreeCollection(TreeNode):
    def __init__(self):
        self.trees = []

    def add(self, tree):
        self.trees.append(tree)

    def __iter__(self):
        return iter(self.trees)

    def __getitem__(self, idx):
        return self.trees[idx]

    def __len__(self):
        return len(self.trees)


def plot_tree_node(node, ax, color='blue'):
    """Recursively plot each node in the tree."""
    if node.parent:
        ax.plot([node.point[0], node.parent.point[0]], [node.point[1], node.parent.point[1]], color=color)
    for child in node.children:
        plot_tree_node(child, ax, color=color)

def plot_path(path, ax, color='red', linewidth=2):
    """Plot a path as a series of line segments."""
    for i in range(1, len(path)):
        ax.plot([path[i-1][0], path[i][0]], [path[i-1][1], path[i][1]], color=color, linewidth=linewidth)

def plot_iteration(tree_root, chosen_path, iteration, workspace_size):
    """Plot a single iteration with the tree and the chosen path."""
    fig, ax = plt.subplots()
    plot_tree_node(tree_root, ax)
    plot_path(chosen_path, ax)
    ax.set_title(f'Iteration {iteration}')
    ax.set_xlim(0, workspace_size[0])
    ax.set_ylim(0, workspace_size[1])
    plt.show()

def plot_final(all_trees, final_path, workspace_size):
    """Plot all trees and the final chosen path."""
    fig, ax = plt.subplots()
    for tree_root in all_trees:
        plot_tree_node(tree_root, ax, color='lightgray')  # Plot all trees in light gray
    plot_path(final_path, ax, color='red', linewidth=3)  # Highlight the final path
    ax.set_title('Final Path with All Trees')
    ax.set_xlim(0, workspace_size[0])
    ax.set_ylim(0, workspace_size[1])
    plt.show()


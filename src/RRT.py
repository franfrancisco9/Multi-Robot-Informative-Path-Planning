import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from informative import InformativePathPlanning
from path_planning_utils import plot_tree

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

    def run(self):
        self.initialize_tree()
        self.expand_tree()
        self.scenario.gp.fit(self.obs_wp, np.log10(self.observations))
        Z_pred, std = self.scenario.predict_spatial_field(self.obs_wp, self.observations)
        return Z_pred, std



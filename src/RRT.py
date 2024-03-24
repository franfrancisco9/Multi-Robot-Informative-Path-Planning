import numpy as np
from scipy.spatial import KDTree
from informative import InformativePathPlanning

class RRTPathPlanning(InformativePathPlanning):
    """
    An RRT-based path planning class that inherits from InformativePathPlanning.
    Focuses on exploring the environment by rapidly expanding a tree towards
    high information gain areas within a specified budget.
    """
    def __init__(self, *args, **kwargs):
        # Initialize the superclass with all provided arguments.
        super().__init__(*args, **kwargs)
        self.name = "RRTPath"  # Name of the path planning strategy.

    def initialize_tree(self):
        """Initializes the RRT tree with a starting position."""
        # Set the starting position and initialize the KDTree.
        initial_position = np.array([[0.5, 0.5]])  # Assuming a starting point within bounds.
        self.tree = KDTree(initial_position)
        self.obs_wp = [initial_position[0].tolist()]
        
        # Simulate the initial measurement at the starting position.
        initial_measurement = self.scenario.simulate_measurements(initial_position)[0]
        self.observations = [initial_measurement]

    def expand_tree(self):
        """Expands the RRT tree within the budget towards unexplored space."""
        distance_travelled = 0
        while distance_travelled < self.budget:
            # Generate a random point within the scenario's workspace.
            random_point = np.random.rand(2) * self.scenario.workspace_size
            
            # Find the nearest existing point in the tree to the random point.
            nearest_idx = self.tree.query([random_point], k=1)[1][0]
            nearest_point = self.tree.data[nearest_idx]
            
            # Calculate the direction and move towards the random point.
            direction = random_point - nearest_point
            direction /= np.linalg.norm(direction)  # Normalize the direction vector.
            new_point = nearest_point + direction * self.d_waypoint_distance
            distance_travelled += np.linalg.norm(new_point - self.obs_wp[-1])
            # Ensure the new point is within workspace boundaries.
            if np.all(new_point >= 0) and np.all(new_point <= self.scenario.workspace_size) and distance_travelled < self.budget:
                # Update the tree with the new point and simulate a measurement.
                self.tree = KDTree(np.vstack([self.tree.data, new_point]))
                new_measurement = self.scenario.simulate_measurements(np.array([new_point]))[0]
                # Update observations and waypoints lists.
                self.observations.append(new_measurement)
                self.obs_wp.append(new_point.tolist())
            elif distance_travelled >= self.budget:
                break
        # print("Distance travelled: ", distance_travelled)
        # print("Number of waypoints: ", len(self.obs_wp))
        # Convert waypoints list to a NumPy array for further processing.
        self.obs_wp = np.array(self.obs_wp)
        self.full_path = np.array(self.obs_wp).reshape(-1, 2).T  # Reshape for plotting.

    def run(self):
        """Executes the path planning process and returns the predicted spatial field."""
        # Initialize the tree and expand it within the budget.
        self.initialize_tree()
        self.expand_tree()
        
        # Fit the GP model with the observed waypoints and measurements.
        unique_positions = np.array(self.obs_wp)
        measurements = np.array(self.observations)
        self.scenario.gp.fit(unique_positions, np.log10(measurements))
        
        # Predict the spatial field based on the explored path.
        Z_pred, std = self.scenario.predict_spatial_field(unique_positions, measurements)
        return Z_pred, std


class BiasInformativeRRTPathPlanning(InformativePathPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree = None
        self.name = "EnhancedInformativeRRTPath"
    
    def initialize_tree(self):
        initial_position = np.array([[0, 0]])  # Starting position
        self.tree = KDTree(initial_position)
        self.observations = self.scenario.simulate_measurements(initial_position)
        self.scenario.gp.fit(initial_position, np.log10(self.observations))

    def generate_random_point_with_bias(self):
        # Generate a grid of points across the workspace
        x = np.linspace(0, self.scenario.workspace_size[0], 100)
        y = np.linspace(0, self.scenario.workspace_size[1], 100)
        X, Y = np.meshgrid(x, y)
        grid = np.vstack([X.ravel(), Y.ravel()]).T
        
        # Predict variance across the grid to find high-uncertainty areas
        _, sigma = self.scenario.gp.predict(grid, return_std=True)
        
        # Probability proportional to sigma (uncertainty) for selecting points
        probabilities = sigma / np.sum(sigma)
        
        # Choose a random index based on uncertainty probabilities
        index = np.random.choice(range(len(grid)), p=probabilities)
        
        return grid[index]

    def expand_tree(self):
        for _ in range(self.n_waypoints - 1):
            random_point = self.generate_random_point_with_bias()
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

        self.obs_wp = np.array(self.tree.data)
        self.full_path = self.obs_wp.reshape(-1, 2).T

    def run(self):
        self.initialize_tree()
        self.expand_tree()
        self.scenario.gp.fit(self.obs_wp, np.log10(self.observations))
        Z_pred, std = self.scenario.predict_spatial_field(self.obs_wp, self.observations)
        return Z_pred, std
    
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



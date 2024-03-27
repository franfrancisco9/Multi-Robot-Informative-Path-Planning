import numpy as np
from tqdm import tqdm

class InformativePathPlanning:
    """
    Implements Informative Path Planning to navigate a scenario within a budget, 
    balancing exploration and exploitation based on Gaussian Process predictions.
    """
    def __init__(self, scenario, beta_t=500.0, budget=375, d_waypoint_distance=2.5):
        """
        Initializes the path planner.

        Parameters:
        - scenario: Scenario object with simulation environment.
        - beta_t: Trade-off parameter between exploration and exploitation.
        - budget: Total allowed distance for path generation.
        - d_waypoint_distance: Desired distance between waypoints.
        """
        self.scenario = scenario
        self.beta_t = beta_t
        self.budget = budget
        self.d_waypoint_distance = d_waypoint_distance
        self.observations = []  # Stores measurements for each waypoint
        self.obs_wp = []  # Stores observed waypoints
        self.name = "InformativePath"
        # Precompute the grid to save computation in select_next_point
        x, y = np.linspace(0, scenario.workspace_size[0], 200), np.linspace(0, scenario.workspace_size[1], 200)
        self.grid = np.vstack(np.meshgrid(x, y)).reshape(2, -1).T

    def select_next_point(self, current_position):
        """
        Selects the next point based on the Gaussian Process model and the acquisition function.
        """
        distances = np.linalg.norm(self.grid - current_position, axis=1)
        valid_points = self.grid[(distances > 2) & (distances <= self.d_waypoint_distance)]

        if valid_points.size == 0:
            return None

        mu, sigma = self.scenario.gp.predict(valid_points, return_std=True)
        acquisition = mu + self.beta_t * sigma
        return valid_points[np.argmax(acquisition)]
    
    def generate_path(self):
        """
        Generates a path within the specified budget using the Informative Path Planning strategy.
        """
        current_position = np.array([0.5, 0.5])
        distance_travelled = self.add_observation_and_update(current_position)
        with tqdm(total=self.budget, desc="Running Informative Path Planning") as pbar:
            while distance_travelled < self.budget:
                next_point = self.select_next_point(current_position) 
                if next_point is None:
                    next_point = self.get_advanced_fallback_point(current_position)
                    if next_point is None:
                        break
                distance_travelled += self.add_observation_and_update(next_point, distance_travelled)
                current_position = next_point
                pbar.update(int(distance_travelled))
        self.obs_wp = np.array(self.obs_wp)
        self.full_path = self.obs_wp.reshape(-1, 2).T

    def add_observation_and_update(self, point, distance_travelled_so_far=0):
        """
        Adds a new waypoint, records its observation, updates the GP model, and returns the total distance travelled.
        """
        measurement = self.scenario.simulate_measurements([point])
        self.observations.append(measurement)  # Using tuple as dict key
        self.obs_wp.append(point)
        
        # Update GP model conditionally to reduce computation
        if len(self.observations) % 5 == 0:  # Example strategy: update every 5 observations
            waypoints = np.array(self.obs_wp)
            measurements = np.array(self.observations)
            self.scenario.gp.fit(waypoints, np.log10(measurements))

        return np.linalg.norm(point - self.obs_wp[-2]) if len(self.obs_wp) > 1 else 0

    def get_advanced_fallback_point(self, current_position):
        """
        Generates a fallback point when no valid next point is found by trying
        in sequence while staying within bounds.
        """
        directions = [np.array([0, self.d_waypoint_distance]), np.array([0, -self.d_waypoint_distance]),
                      np.array([-self.d_waypoint_distance, 0]), np.array([self.d_waypoint_distance, 0])]
        x_max, y_max = self.scenario.workspace_size

        for direction in directions:
            next_point = current_position + direction
            if 0.5 <= next_point[0] <= x_max - 0.5 and 0.5 <= next_point[1] <= y_max - 0.5:
                return next_point

        return current_position  # Fallback to current position if no valid points are found

    def run(self):
        """
        Executes the path planning process and returns the predicted spatial field.
        """
        self.generate_path()
        waypoints = np.array(self.obs_wp)
        measurements = np.array(self.observations)
        return self.scenario.predict_spatial_field(waypoints, measurements)

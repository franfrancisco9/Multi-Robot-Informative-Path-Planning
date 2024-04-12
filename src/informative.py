import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree
from path_planning_utils import estimate_sources_bayesian, save_plot_iteration

class BaseInformative:
    """
    Implements Base Informative to navigate a scenario within a budget, 
    balancing exploration and exploitation based on Gaussian Process predictions.
    """
    def __init__(self, scenario, beta_t=5.0, budget=375, d_waypoint_distance=2.5):
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
        self.name = "BaseInformative"
        # Precompute the grid to save computation in select_next_point
        x, y = np.linspace(0, scenario.workspace_size[0], 200), np.linspace(0, scenario.workspace_size[1], 200)
        self.grid = np.vstack(np.meshgrid(x, y)).reshape(2, -1).T
        self.grid_kdtree = KDTree(self.grid)

    def select_next_point(self, current_position):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def generate_path(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def add_observation_and_update(self, point, distance_travelled_so_far=0):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_advanced_fallback_point(self, current_position):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def run(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

class InformativePathPlanning(BaseInformative):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "InformativePath"

    def select_next_point(self, current_position):
        """
        Selects the next point based on the Gaussian Process model and the acquisition function,
        using a KD-tree for efficient neighbor searching.
        """
        # Query KD-tree for points within the valid distance range
        indices = self.grid_kdtree.query_ball_point(current_position, self.d_waypoint_distance)
        valid_points = self.grid[indices]

        # Filter out points too close to the current position (if necessary)
        valid_points = valid_points[np.linalg.norm(valid_points - current_position, axis=1) > 2]

        if valid_points.size == 0:
            return None

        # Perform GP prediction and calculate acquisition function for valid points
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
            while self.budget > 0:
                next_point = self.select_next_point(current_position) 
                if next_point is None:
                    next_point = self.get_advanced_fallback_point(current_position)
                    if next_point is None:
                        break
                distance_travelled = self.add_observation_and_update(next_point, distance_travelled)
                self.budget -= distance_travelled
                current_position = next_point
                pbar.update(distance_travelled)
                
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
        if len(self.observations) % 10 == 0:  # Example strategy: update every 5 observations
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
        # print(f"Path length: {len(self.obs_wp)}")
        waypoints = np.array(self.obs_wp)
        self.measurements = np.array(self.observations)
        return self.scenario.predict_spatial_field(waypoints, self.measurements)


class InformativeBICPathPlanning(BaseInformative):
    def __init__(self, *args, lambda_b=1, max_sources=1, n_samples=20, s_stages=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "InformativeBICPathPlanning"
        self.lambda_b = lambda_b
        self.max_sources = max_sources
        self.n_samples = n_samples
        self.s_stages = s_stages
        self.best_bic = -np.inf
        self.current_estimates = None
        self.last_best_location = None
        self.direction = np.array([1, 1])  # Initially arbitrary
        self.full_path = []
        self.estimate_counter = 0

    def select_next_point(self, current_position):
        """
        Selects the next point based on the current understanding of the environment.
        Initially random exploration, could be improved to use current estimates.
        """
        step_size = self.d_waypoint_distance
        # This example uses a simple strategy. Consider enhancing this with your insights.
        direction = np.random.rand(2) * 2 - 1
        direction /= np.linalg.norm(direction)
        next_point = current_position + direction * step_size
        return np.clip(next_point, [0, 0], self.scenario.workspace_size)

    def add_observation_and_update(self, point):
        measurement = self.scenario.simulate_measurements([point])[0]
        self.observations.append(measurement)
        self.obs_wp.append(point)

        self.estimate_counter += 1
        if self.estimate_counter % 10 == 0 or self.estimate_counter == 1:
            self.perform_estimation()

    def perform_estimation(self):
        estimated_locs, estimated_num_sources, bic = estimate_sources_bayesian(
            self.obs_wp, self.observations, self.lambda_b, self.max_sources, self.n_samples, self.s_stages
        )
        if bic > self.best_bic:
            self.best_bic = bic
            self.current_estimates = (np.array(estimated_locs).reshape(-1, 3), estimated_num_sources)

    def generate_path(self):
        current_position = np.array([0.5, 0.5])
        self.full_path = [current_position]
        with tqdm(total=self.budget, desc="Running Informative BIC Path Planning") as pbar:
            while self.budget > 0:
                next_point = self.select_next_point(current_position)
                self.add_observation_and_update(next_point)
                distance_travelled = np.linalg.norm(next_point - current_position)
                self.budget -= distance_travelled
                pbar.update(distance_travelled)
                current_position = next_point

    def run(self):
        self.generate_path()
        self.obs_wp = np.array(self.obs_wp)
        self.full_path = self.obs_wp.reshape(-1, 2).T
        self.measurements = np.array(self.observations)
        return self.scenario.predict_spatial_field(self.obs_wp, self.measurements)

class InformativeBetaBICPathPlanning(BaseInformative):
    def __init__(self, *args, lambda_b=1, max_sources=1, n_samples=20, s_stages=5, beta_t=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "InformativeBetaBICPathPlanning"
        self.lambda_b = lambda_b
        self.max_sources = max_sources
        self.n_samples = n_samples
        self.s_stages = s_stages
        self.beta_t = beta_t  # Balance between exploration and exploitation
        self.best_bic = -np.inf
        self.current_estimates = None

    def select_next_point(self, current_position):
        """
        Selects the next point based on a blend of GP-based exploration and exploitation
        towards the current best source estimate.
        """
        # Query GP for points within the valid distance range
        indices = self.grid_kdtree.query_ball_point(current_position, self.d_waypoint_distance)
        valid_points = self.grid[indices]

        # Perform GP prediction to guide exploration
        mu, sigma = self.scenario.gp.predict(valid_points, return_std=True)
        acquisition = mu + self.beta_t * sigma  # Exploration factor

        # Bias towards current best estimate (if available)
        if self.current_estimates is not None:
            distances_to_best_estimate = np.linalg.norm(valid_points - self.last_best_location, axis=1)
            # Modify acquisition function to include bias towards best estimate
            acquisition -= distances_to_best_estimate

        # Introduce randomness to the selection process
        top_candidates = np.argsort(acquisition)[-10:]  # Select top 10 candidates
        chosen_index = np.random.choice(top_candidates, 1)[0]  # Randomly pick from the top 10

        # Debugging: Print the chosen point and its acquisition score
        # print("Chosen point:", valid_points[chosen_index], "Score:", acquisition[chosen_index])

        return valid_points[chosen_index]

    def add_observation_and_update(self, point):
        """
        Adds a new waypoint, records its observation, updates the GP model, and performs BIC-based estimation.
        """
        measurement = self.scenario.simulate_measurements([point])[0]
        self.observations.append(measurement)
        self.obs_wp.append(point)
        
        # Update GP model
        if len(self.observations) % 50 == 0:
            waypoints = np.array(self.obs_wp)
            measurements = np.array(self.observations)
            self.scenario.gp.fit(waypoints, np.log10(measurements))

        # Perform BIC-based estimation periodically
        if len(self.observations) % 10 == 0:
            self.perform_bic_based_estimation()

    def perform_bic_based_estimation(self):
        """
        Updates the best source estimates and BIC based on current observations using the BIC estimation function.
        """
        estimated_locs, estimated_num_sources, bic = estimate_sources_bayesian(
            self.obs_wp, self.observations, self.lambda_b, self.max_sources, self.n_samples, self.s_stages
        )

        if bic > self.best_bic:
            self.best_bic = bic
            self.current_estimates = (np.array(estimated_locs).reshape(-1, 3), estimated_num_sources)
            self.last_best_location = self.obs_wp[-1]

    def generate_path(self):
        current_position = np.array([0.5, 0.5])
        with tqdm(total=self.budget, desc="Running Informative Beta BIC Path Planning") as pbar:
            while self.budget > 0:
                next_point = self.select_next_point(current_position)
                self.add_observation_and_update(next_point)
                distance_travelled = np.linalg.norm(next_point - current_position)
                self.budget -= distance_travelled
                pbar.update(distance_travelled)
                current_position = next_point

    def run(self):
        """
        Executes the path planning process aiming to maximize the BIC for source estimation
        and returns the predicted spatial field and standard deviation.
        """
        # Generate path and update observations logic remains the same
        self.generate_path()
        self.obs_wp = np.array(self.obs_wp)
        self.full_path = self.obs_wp.reshape(-1, 2).T  
        self.measurements = np.array(self.observations)
        return self.scenario.predict_spatial_field(self.obs_wp, self.measurements)


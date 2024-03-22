import numpy as np

class InformativePathPlanning:
    def __init__(self, scenario, lambda_param=1.0, beta_t=500.0, n_waypoints=200, d_waypoint_distance=2.5):
        self.scenario = scenario
        self.lambda_param = lambda_param
        self.beta_t = beta_t
        self.n_waypoints = n_waypoints
        self.d_waypoint_distance = d_waypoint_distance
        self.observations = None
        self.obs_wp = []  # Initialize an empty list for observed waypoints
        self.full_path = []  # Initialize an empty list for the full path
        self.name = "InformativePath"
    
    def select_next_point(self, current_position):
        x = np.linspace(0, self.scenario.workspace_size[0], 200)
        y = np.linspace(0, self.scenario.workspace_size[1], 200)
        X, Y = np.meshgrid(x, y)
        grid = np.vstack([X.ravel(), Y.ravel()]).T

        # Calculate the distance of each grid point from the current position
        distances = np.linalg.norm(grid - current_position, axis=1)

        # Filter points that are within the desired step distance
        valid_points = grid[(distances > 2) & (distances <= self.d_waypoint_distance)]

        if len(valid_points) == 0:
            print("No valid points found")
            return None

        # Evaluate the acquisition function for each valid point
        mu, sigma = self.scenario.gp.predict(valid_points, return_std=True)
        # print("Mean: ", mu)
        # print("Sigma: ", sigma)
        acquisition_values = mu + self.beta_t * sigma
        # print("Acquisition values: ", acquisition_values)
        # Select the point with the highest acquisition value
        next_point_index = np.argmax(acquisition_values)
        # print("Next point index: ", next_point_index)
        next_point = valid_points[next_point_index]

        return next_point
    
    def generate_path(self):
        # Start from a random position within the workspace
        current_position = np.array([0, 0])
        self.obs_wp.append(current_position)
        self.observations  = self.scenario.simulate_measurements(np.array([current_position]))
        self.scenario.gp.fit(np.array([current_position]), np.log10(self.observations))  # Assuming log10 scale for compatibility
        for _ in range(1, self.n_waypoints):
            next_point = self.select_next_point(current_position)
            if next_point is None:
                # go up down left or right (chosen randomly) if it is inside the workspace
                # if not, choose a new random point
                # Check up
                if current_position[1] + self.d_waypoint_distance <= self.scenario.workspace_size[1]:
                    next_point = current_position + np.array([0, self.d_waypoint_distance])
                # Check down
                elif current_position[1] - self.d_waypoint_distance >= 0:
                    next_point = current_position - np.array([0, self.d_waypoint_distance])
                # Check left
                elif current_position[0] - self.d_waypoint_distance >= 0:
                    next_point = current_position - np.array([self.d_waypoint_distance, 0])
                # Check right
                elif current_position[0] + self.d_waypoint_distance <= self.scenario.workspace_size[0]:
                    next_point = current_position + np.array([self.d_waypoint_distance, 0])
            self.obs_wp.append(next_point)
            current_position = next_point
            self.observations = self.scenario.simulate_measurements(np.array(self.obs_wp))
            self.scenario.gp.fit(np.array(self.obs_wp), np.log10(self.observations))

        self.obs_wp = np.array(self.obs_wp).reshape(-1, 2)
        self.full_path = np.array(self.obs_wp).reshape(-1, 2).T

    
    def run(self):
        self.generate_path()
        # measurements = self.scenario.simulate_measurements(np.array(self.obs_wp))
        # self.scenario.gp.fit(np.array(self.obs_wp), np.log10(self.observations ))  # Assuming log10 scale for compatibility
        Z_pred, std = self.scenario.predict_spatial_field(np.array(self.obs_wp), self.observations)
        
        return Z_pred, std
    

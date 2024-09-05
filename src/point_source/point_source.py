# src/point_source/point_source.py

"""
Module for modeling a point source field using Gaussian Processes and simulating measurements.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
from sklearn.exceptions import ConvergenceWarning
from typing import Tuple, List, Optional, Dict

# Suppress specific sklearn convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class PointSourceField:
    """
    Models a point source field using Gaussian Processes and simulates measurements with configurable noise levels.

    Attributes:
        sources (List): List of point source sources in the field.
        gp (GaussianProcessRegressor): Gaussian Process Regressor for field simulation.
    """

    def __init__(self, num_sources: int = 1, workspace_size: Tuple[int, int, int, int] = (0, 40, 0, 40), obstacles: List = [],
                 intensity_range: Tuple[int, int] = (100000, 1000000), kernel_params: Optional[Dict[str, float]] = None, seed: Optional[int] = None):
        """
        Initializes the point source field model.

        Args:
            num_sources (int): Number of point source sources.
            workspace_size (Tuple[int, int]): Size of the workspace (width, height).
            intensity_range (Tuple[int, int]): Range of intensities for the sources.
            kernel_params (Optional[Dict[str, float]]): Parameters for the Gaussian process kernel.
            seed (Optional[int]): Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        self.validate_inputs(num_sources, workspace_size, intensity_range)
        self.sources = self.generate_sources(num_sources, workspace_size, intensity_range)
        self.r_s = 0.5  # Source radius
        self.r_d = 0.5  # Detector radius
        self.T = 100  # Transmission factor
        self.workspace_size = workspace_size
        # set of obstacles areas
        self.obstacles = obstacles 
        # [
        #     {
        #         "type": "rectangle",
        #         "x": 10,
        #         "y": 10,
        #         "width": 10,
        #         "height": 5
        #     },
        #     {
        #         "type": "rectangle",
        #         "x": 10,
        #         "y": 5,
        #         "width": 5,
        #         "height": 5
        #     }
        # ]
        self.intensity_range = intensity_range
        self.x = np.linspace(workspace_size[0], workspace_size[1], (workspace_size[1] - workspace_size[0]) * 10)
        self.y = np.linspace(workspace_size[2], workspace_size[3], (workspace_size[3] - workspace_size[2]) * 10)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.g_truth = self.ground_truth()
        kernel = self.construct_kernel(kernel_params)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

    def validate_inputs(self, num_sources: int, workspace_size: Tuple[int, int], intensity_range: Tuple[int, int]) -> None:
        """Validates input parameters for the class constructor."""
        if num_sources < 1:
            raise ValueError("Number of sources must be at least 1.")
        # if workspace_size[0] <= 0 or workspace_size[1] <= 0:
        #     raise ValueError("Workspace dimensions must be positive.")
        if intensity_range[0] <= 0 or intensity_range[1] <= 0 or intensity_range[0] >= intensity_range[1]:
            raise ValueError("Intensity range must be positive and the lower bound must be less than the upper bound.")

    def construct_kernel(self, kernel_params: Optional[Dict[str, float]]) -> RBF:
        """Constructs the Gaussian process kernel."""
        if kernel_params is None:
            kernel_params = {'sigma': 1.0, 'length_scale': 1.0}
        kernel = C(kernel_params['sigma'], (1e-3, 50)) * RBF(kernel_params['length_scale'], (1e-2, 50))
        return kernel

    def generate_sources(self, num_sources: int, workspace_size: Tuple[int, int, int, int], intensity_range: Tuple[int, int]) -> List[List[float]]:
        """Generates random sources within the workspace using vectorized operations."""
        rand_x = np.random.uniform(workspace_size[0], workspace_size[1], num_sources)
        rand_y = np.random.uniform(workspace_size[2], workspace_size[3], num_sources)
        rand_A = np.random.uniform(*intensity_range, num_sources)
        sources = np.column_stack((rand_x, rand_y, rand_A))
        # revert seed   
        np.random.seed(None)
        return sources.tolist()  

    def update_source(self, source_index: int, new_x: float, new_y: float, new_A: float) -> None:
        """Updates a specific source's parameters."""
        if source_index < len(self.sources):
            self.sources[source_index] = [new_x, new_y, new_A]
            self.recalculate_ground_truth()  # Recalculate ground truth after updating the source
        else:
            print("Source index out of range.")

    def recalculate_ground_truth(self) -> None:
        """Recalculates the ground truth based on current source positions."""
        self.g_truth = self.ground_truth()

    def intensity(self, waypoints: List[Tuple[float, float]]) -> np.ndarray:
        """Vectorized computation of intensity for an array of waypoints, including absorption effects."""
        waypoints = np.array(waypoints)  # Ensure waypoints is a numpy array.
        
        if waypoints.ndim == 1:
            waypoints = waypoints[np.newaxis, :]  # Add a new axis if waypoints is a single point.
        
        # Extract coordinates of sources and their amplitudes.
        source_positions = np.array(self.sources)[:, :2]
        source_amplitudes = np.array(self.sources)[:, 2]
        
        # Calculate distances between each waypoint and each source.
        distances = np.linalg.norm(waypoints[:, np.newaxis, :] - source_positions[np.newaxis, :, :], axis=-1)
        distances = np.maximum(distances, 1e-6)

        # Initialize the total absorption array
        total_absorption = np.ones((waypoints.shape[0], len(self.sources)))
        
        # Compute the absorption for each waypoint-source pair
        air_absorption_coefficient = 0.001205
        concrete_absorption_coefficient = 2.35

        for i, waypoint in enumerate(waypoints):
            for j, (source_pos, source_amp) in enumerate(zip(source_positions, source_amplitudes)):
                # Calculate direct distance and air absorption
                dist = distances[i, j]
                total_absorption[i, j] = np.exp(-air_absorption_coefficient * dist)

                # Check for obstacles
                for obstacle in self.obstacles:
                    if obstacle['type'] == 'rectangle':
                        # Compute path length within the obstacle for the line from the source to the waypoint
                        intersect_mask, path_length_in_obstacle = self.compute_path_length_in_obstacle(
                            np.array([complex(waypoint[0], waypoint[1])]),  # Convert waypoint to complex for compute function
                            complex(source_pos[0], source_pos[1]),  # Source as complex
                            obstacle
                        )

                        # If the path intersects the obstacle, apply obstacle absorption
                        if intersect_mask[0]:
                            total_absorption[i, j] *= np.exp(-concrete_absorption_coefficient * path_length_in_obstacle[0])
        
        # Apply the intensity formula including absorption effects.
        intensity = np.where(distances <= self.r_s,
                            source_amplitudes / (4 * np.pi * self.r_s**2),
                            source_amplitudes / (4 * np.pi * distances**2) * total_absorption)

        # Sum the contributions for each waypoint.
        total_intensity = np.sum(intensity, axis=-1)
        
        return total_intensity

    def ground_truth(self):
        """Vectorized computation of ground truth to include absorption effects."""
        # Create a meshgrid of coordinates as complex numbers for vectorized computation.
        R = self.X + 1j * self.Y
        Z_true = np.zeros(R.shape)
        
        air_absorption_coefficient = 0.001205  # Absorption coefficient for air
        concrete_absorption_coefficient = 2.35  # Absorption coefficient for concrete

        for source in self.sources:
            r_n = np.array(source[:2])
            A_n = source[2]
            R_n = r_n[0] + 1j * r_n[1]
            dist = np.abs(R - R_n)
            
            # Calculate theta for response calculation
            theta = np.arcsin(np.minimum(self.r_d / dist, 1))
            
            # Initialize intensity and response arrays
            intensity = np.zeros(R.shape)
            response = np.zeros(R.shape)

            # Vectorized computation of absorption factors
            # Start with air absorption for the direct distance
            total_absorption = np.exp(-air_absorption_coefficient * dist)

            # Check for obstacles
            for obstacle in self.obstacles:
                # Determine the type of obstacle and its properties
                if obstacle['type'] == 'rectangle':
                    # Calculate the path length through the obstacle for each grid point
                    intersect_mask, path_length_in_obstacle = self.compute_path_length_in_obstacle(R, R_n, obstacle)
                    
                    # Update total absorption for points intersecting the obstacle
                    total_absorption[intersect_mask] *= np.exp(-concrete_absorption_coefficient * path_length_in_obstacle[intersect_mask])
            
            # Compute intensity using updated formula
            intensity = np.where(dist <= self.r_s,
                                A_n / (4 * np.pi * self.r_s**2),
                                A_n / (4 * np.pi * dist**2) * total_absorption)
            
            # Compute response using updated formula
            response = np.where(dist <= self.r_d,
                                0.5 * A_n,
                                0.5 * A_n * (1 - np.cos(theta)) * total_absorption)
            
            # Accumulate results
            Z_true += intensity + response
        
        return Z_true

    def compute_path_length_in_obstacle(self, R, R_n, obstacle):
        """
        Computes if a path intersects a rectangular obstacle and calculates the path length within the obstacle.

        Args:
            R (np.ndarray or complex): Meshgrid of complex coordinates or a single complex coordinate.
            R_n (complex): Complex coordinate of the source.
            obstacle (dict): Obstacle description containing type and dimensions.

        Returns:
            intersect_mask (np.ndarray): Boolean mask of points intersecting the obstacle.
            path_length_in_obstacle (np.ndarray): Path length within the obstacle for intersecting points.
        """
        # Obstacle boundaries
        x_min, x_max = obstacle['x'], obstacle['x'] + obstacle['width']
        y_min, y_max = obstacle['y'], obstacle['y'] + obstacle['height']
        
        # Source position
        x_n, y_n = R_n.real, R_n.imag

        # Initialize path length array and intersection mask
        if np.isscalar(R):
            # Handle the case where R is a single complex number
            intersect_mask = np.zeros(1, dtype=bool)
            path_length_in_obstacle = np.zeros(1)
            R_array = [R]  # Make it iterable
        else:
            # R is expected to be a 2D array
            intersect_mask = np.zeros(R.shape, dtype=bool)
            path_length_in_obstacle = np.zeros(R.shape)
            R_array = R.flatten()

        # Vectorized calculation for intersections
        for index, R_point in enumerate(R_array):
            x_r, y_r = R_point.real, R_point.imag

            # Check if the line from source to point intersects with the obstacle
            if self.line_intersects_rectangle(x_n, y_n, x_r, y_r, x_min, x_max, y_min, y_max):
                intersect_mask.flat[index] = True
                
                # Compute entry and exit points within the rectangle
                entry_x, entry_y, exit_x, exit_y = self.calculate_entry_exit_points(x_n, y_n, x_r, y_r, x_min, x_max, y_min, y_max)
                
                # Calculate the path length within the obstacle
                path_length_in_obstacle.flat[index] = np.sqrt((exit_x - entry_x) ** 2 + (exit_y - entry_y) ** 2)

        return intersect_mask, path_length_in_obstacle

    def line_intersects_rectangle(self, x1, y1, x2, y2, x_min, x_max, y_min, y_max):
        """
        Checks if the line from (x1, y1) to (x2, y2) intersects with the rectangle defined by [x_min, x_max, y_min, y_max].
        """
        # Check if either endpoint is inside the rectangle
        if (x_min <= x1 <= x_max and y_min <= y1 <= y_max) or (x_min <= x2 <= x_max and y_min <= y2 <= y_max):
            return True
        
        # Check for intersections with rectangle sides
        if self.line_intersects_line(x1, y1, x2, y2, x_min, y_min, x_min, y_max) or \
        self.line_intersects_line(x1, y1, x2, y2, x_max, y_min, x_max, y_max) or \
        self.line_intersects_line(x1, y1, x2, y2, x_min, y_min, x_max, y_min) or \
        self.line_intersects_line(x1, y1, x2, y2, x_min, y_max, x_max, y_max):
            return True
        
        return False

    def line_intersects_line(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Checks if the line from (x1, y1) to (x2, y2) intersects with the line from (x3, y3) to (x4, y4).
        """
        # Calculate direction of the lines
        d1 = (x2 - x1, y2 - y1)
        d2 = (x4 - x3, y4 - y3)
        
        # Solve for intersection
        denominator = d1[0] * d2[1] - d1[1] * d2[0]
        if denominator == 0:
            return False  # Lines are parallel
        
        # Calculate intersection point
        t = ((x3 - x1) * d2[1] - (y3 - y1) * d2[0]) / denominator
        u = ((x3 - x1) * d1[1] - (y3 - y1) * d1[0]) / denominator
        
        # Check if intersection point is on both segments
        return 0 <= t <= 1 and 0 <= u <= 1

    def calculate_entry_exit_points(self, x1, y1, x2, y2, x_min, x_max, y_min, y_max):
        """
        Calculates the entry and exit points of a line passing through a rectangle.
        """
        entry_x = entry_y = exit_x = exit_y = None
        
        # Calculate potential intersection points
        potential_points = []
        if x1 != x2:
            t_min = (x_min - x1) / (x2 - x1)
            t_max = (x_max - x1) / (x2 - x1)
            y_at_x_min = y1 + t_min * (y2 - y1)
            y_at_x_max = y1 + t_max * (y2 - y1)
            if y_min <= y_at_x_min <= y_max:
                potential_points.append((x_min, y_at_x_min))
            if y_min <= y_at_x_max <= y_max:
                potential_points.append((x_max, y_at_x_max))
        
        if y1 != y2:
            t_min = (y_min - y1) / (y2 - y1)
            t_max = (y_max - y1) / (y2 - y1)
            x_at_y_min = x1 + t_min * (x2 - x1)
            x_at_y_max = x1 + t_max * (x2 - x1)
            if x_min <= x_at_y_min <= x_max:
                potential_points.append((x_at_y_min, y_min))
            if x_min <= x_at_y_max <= x_max:
                potential_points.append((x_at_y_max, y_max))
        
        if len(potential_points) >= 2:
            entry_x, entry_y = potential_points[0]
            exit_x, exit_y = potential_points[1]
        
        return entry_x, entry_y, exit_x, exit_y



    def update(self, obs_wp, log_obs_vals):
        """Update the GP model based on observations."""
        self.gp.fit(obs_wp, log_obs_vals)

    def simulate_measurements(self, waypoints: List[Tuple[float, float]], noise_level: float = 0.001) -> np.ndarray:
        """Simulates measurements at given waypoints with configurable noise, vectorized version."""
        intensities = self.intensity(waypoints)
        noise = np.random.normal(0, noise_level * intensities)
        measurements = np.maximum(intensities + noise, 1e-6)
        return measurements

    def predict_spatial_field(self, waypoints: List[Tuple[float, float]], measurements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts the spatial field based on waypoints and measurements."""
        measurements_log = np.log10(np.maximum(measurements, 1e-6))
        self.gp.fit(waypoints, measurements_log)
        r = np.column_stack((self.X.ravel(), self.Y.ravel()))
        Z_pred_log, std = self.gp.predict(r, return_std=True)
        Z_pred = 10 ** Z_pred_log.reshape(self.X.shape)
        return Z_pred, std
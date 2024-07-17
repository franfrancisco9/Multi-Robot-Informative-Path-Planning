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

    def __init__(self, num_sources: int = 1, workspace_size: Tuple[int, int] = (40, 40), intensity_range: Tuple[int, int] = (10000, 1000000),
                 kernel_params: Optional[Dict[str, float]] = None, seed: Optional[int] = None):
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
        self.intensity_range = intensity_range
        self.x = np.linspace(0, workspace_size[0], 200)
        self.y = np.linspace(0, workspace_size[1], 200)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.g_truth = self.ground_truth()
        kernel = self.construct_kernel(kernel_params)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

    def validate_inputs(self, num_sources: int, workspace_size: Tuple[int, int], intensity_range: Tuple[int, int]) -> None:
        """Validates input parameters for the class constructor."""
        if num_sources < 1:
            raise ValueError("Number of sources must be at least 1.")
        if workspace_size[0] <= 0 or workspace_size[1] <= 0:
            raise ValueError("Workspace dimensions must be positive.")
        if intensity_range[0] <= 0 or intensity_range[1] <= 0 or intensity_range[0] >= intensity_range[1]:
            raise ValueError("Intensity range must be positive and the lower bound must be less than the upper bound.")

    def construct_kernel(self, kernel_params: Optional[Dict[str, float]]) -> RBF:
        """Constructs the Gaussian process kernel."""
        if kernel_params is None:
            kernel_params = {'sigma': 1, 'l': 1}
        kernel = C(kernel_params['sigma'], (1e-5, 5)) ** 2 * RBF(kernel_params['l'], (1e-5, 50))
        return kernel

    def generate_sources(self, num_sources: int, workspace_size: Tuple[int, int], intensity_range: Tuple[int, int]) -> List[List[float]]:
        """Generates random sources within the workspace using vectorized operations."""
        rand_x = np.random.uniform(0, workspace_size[0], num_sources)
        rand_y = np.random.uniform(0, workspace_size[1], num_sources)
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
        """Vectorized computation of intensity for an array of waypoints."""
        waypoints = np.array(waypoints)  # Ensure waypoints is a numpy array.
        
        if waypoints.ndim == 1:
            waypoints = waypoints[np.newaxis, :]  # Add a new axis if waypoints is a single point.
        
        # Extract coordinates of sources and their amplitudes.
        source_positions = np.array(self.sources)[:, :2]
        source_amplitudes = np.array(self.sources)[:, 2]
        
        # Calculate distances between each waypoint and each source.
        distances = np.linalg.norm(waypoints[:, np.newaxis, :] - source_positions[np.newaxis, :, :], axis=-1)
        
        # Apply the intensity formula.
        within_r_s = distances <= self.r_s
        outside_r_s = ~within_r_s
        
        # Intensity contributions from sources within r_s.
        intensity_within = np.where(within_r_s, source_amplitudes / (4 * np.pi * self.r_s**2), 0)
        
        # Intensity contributions from sources outside r_s.
        intensity_outside = np.where(outside_r_s, source_amplitudes * self.T / (4 * np.pi * distances**2), 0)
        
        # Sum the contributions for each waypoint.
        total_intensity = np.sum(intensity_within + intensity_outside, axis=-1)
        # total_intensity = np.sum(source_amplitudes / (distances**2), axis=-1)  # Simplified intensity calculation for testing
        return total_intensity

    def ground_truth(self):
        """Vectorized computation of ground truth to enhance performance."""
        # Create a meshgrid of coordinates as complex numbers for vectorized computation.
        R = self.X + 1j * self.Y
        Z_true = np.zeros(R.shape)

        for source in self.sources:
            r_n = np.array(source[:2])
            A_n = source[2]
            # Convert source position to complex number format.
            R_n = r_n[0] + 1j * r_n[1]
            # Calculate distance from each point in the grid to the source.
            dist = np.abs(R - R_n)
            
            # Vectorized intensity and response calculations
            intensity = np.where(dist <= self.r_s,
                                A_n / (4 * np.pi * self.r_s**2),
                                A_n * self.T / (4 * np.pi * dist**2))

            theta = np.arcsin(np.minimum(self.r_d / dist, 1))
            response = np.where(dist <= self.r_d,
                                0.5 * A_n,
                                0.5 * A_n * (1 - np.cos(theta)))

            # Accumulate results
            Z_true += intensity + 50 * response
            # test a simplified intensity/distance**2
            # r_n = np.array(source[:2])
            # A_n = source[2]
            # # Convert source position to complex number format.
            # R_n = r_n[0] + 1j * r_n[1]
            # # Calculate distance from each point in the grid to the source.
            # dist = np.abs(R - R_n)

            # # Vectorized intensity and response calculations
            # Z_true += A_n / (dist**2)
        return Z_true

    def simulate_measurements(self, waypoints: List[Tuple[float, float]], noise_level: float = 0.0001) -> np.ndarray:
        """Simulates measurements at given waypoints with configurable noise, vectorized version."""
        intensities = self.intensity(waypoints)
        noise = np.random.normal(0, noise_level * intensities)
        measurements = np.maximum(intensities + noise, 1e-6)
        return measurements

    def predict_spatial_field(self, waypoints: List[Tuple[float, float]], measurements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts the spatial field based on waypoints and measurements."""
        measurements_log = np.log10(measurements)
        self.gp.fit(waypoints, measurements_log)
        r = np.column_stack((self.X.ravel(), self.Y.ravel()))
        Z_pred_log, std = self.gp.predict(r, return_std=True)
        Z_pred = 10 ** Z_pred_log.reshape(self.X.shape)
        return Z_pred, std

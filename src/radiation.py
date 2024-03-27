import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress specific sklearn convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class RadiationField:
    """
    Models a radiation field using Gaussian Processes and simulates
    measurements with configurable noise levels.

    Attributes:
        sources (list): List of radiation sources in the field.
        gp (GaussianProcessRegressor): Gaussian Process Regressor for field simulation.
    """

    def __init__(self, num_sources=1, workspace_size=(40, 40), intensity_range=(10000, 100000),
                 kernel_params=None, seed=None):
        """
        Initializes the radiation field model.

        Args:
            num_sources (int): Number of radiation sources.
            workspace_size (tuple): Size of the workspace (width, height).
            intensity_range (tuple): Range of intensities for the sources.
            kernel_params (dict): Parameters for the Gaussian process kernel.
            seed (int): Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        self.validate_inputs(num_sources, workspace_size, intensity_range)
        self.sources = self.generate_sources(num_sources, workspace_size, intensity_range)
        self.r_s = 0.5  # Source radius
        self.r_d = 0.5  # Detector radius
        self.T = 100  # Transmission factor
        self.workspace_size = workspace_size
        self.x = np.linspace(0, workspace_size[0], 200)
        self.y = np.linspace(0, workspace_size[1], 200)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.g_truth = self.ground_truth()
        kernel = self.construct_kernel(kernel_params)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

    def validate_inputs(self, num_sources, workspace_size, intensity_range):
        """Validates input parameters for the class constructor."""
        if num_sources < 1:
            raise ValueError("Number of sources must be at least 1.")
        if workspace_size[0] <= 0 or workspace_size[1] <= 0:
            raise ValueError("Workspace dimensions must be positive.")
        if intensity_range[0] <= 0 or intensity_range[1] <= 0 or intensity_range[0] >= intensity_range[1]:
            raise ValueError("Intensity range must be positive and the lower bound must be less than the upper bound.")

    def construct_kernel(self, kernel_params):
        """Constructs the Gaussian process kernel."""
        if kernel_params is None:
            kernel_params = {'sigma': 1, 'l': 1}
        kernel = C(kernel_params['sigma'], (1e-5, 5)) ** 2 * RBF(kernel_params['l'], (1e-5, 50))
        return kernel

    def generate_sources(self, num_sources, workspace_size, intensity_range):
        """Generates random sources within the workspace."""
        sources = []
        for _ in range(num_sources):
            rand_x = np.random.uniform(0, workspace_size[0])
            rand_y = np.random.uniform(0, workspace_size[1])
            rand_A = np.random.uniform(*intensity_range)
            sources.append([rand_x, rand_y, rand_A])
        return sources

    def update_source(self, source_index, new_x, new_y, new_A):
        """Updates a specific source's parameters."""
        if source_index < len(self.sources):
            self.sources[source_index] = [new_x, new_y, new_A]
            self.recalculate_ground_truth()  # Recalculate ground truth after updating the source
        else:
            print("Source index out of range.")

    def recalculate_ground_truth(self):
        """Recalculates the ground truth based on current source positions."""
        self.g_truth = self.ground_truth()

    def intensity(self, r):
        """Computes intensity at a point r."""
        intensity = 0
        for source in self.sources:
            r_n, A_n = np.array(source[:2]), source[2]
            dist = np.linalg.norm(r - r_n)
            if dist <= self.r_s:
                intensity += A_n / (4 * np.pi * self.r_s**2)
            else:
                intensity += A_n * self.T / (4 * np.pi * dist**2)
        return intensity

    def response(self, r):
        """Computes response at a point r."""
        R = 0
        for source in self.sources:
            r_n, A_n = np.array(source[:2]), source[2]
            dist = np.linalg.norm(r - r_n)
            if dist <= self.r_d:
                R += 0.5 * A_n
            else:
                theta = np.arcsin(self.r_d / dist)
                R += 0.5 * A_n * (1 - np.cos(theta))
        return R

    def ground_truth(self):
        """Vectorized computation of ground truth to enhance performance."""
        Z_true = np.zeros(self.X.shape)
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                r = np.array([self.X[i, j], self.Y[i, j]])
                Z_true[i, j] = self.intensity(r) + 50 * self.response(r)
        return Z_true

    def simulate_measurements(self, waypoints, noise_level=0.0001):
        """Simulates measurements at given waypoints with configurable noise."""
        measurements = []
        for wp in waypoints:
            intensity = self.intensity(wp)
            noise = np.random.normal(0, noise_level * intensity)
            measurement = max(intensity + noise, 1e-6)
            measurements.append(measurement)
        return measurements

    def predict_spatial_field(self, waypoints, measurements):
        """Predicts the spatial field based on waypoints and measurements."""
        measurements_log = np.log10(measurements)
        self.gp.fit(waypoints, measurements_log)
        r = np.column_stack((self.X.ravel(), self.Y.ravel()))
        Z_pred_log, std = self.gp.predict(r, return_std=True)
        Z_pred = 10 ** Z_pred_log.reshape(self.X.shape)
        return Z_pred, std

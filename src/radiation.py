import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# remove convergence warning from sklearn
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class RadiationField:
    def __init__(self, num_sources=1, workspace_size=(40, 40), intensity_range=(10000, 100000), kernel_params=None, seed= None):
        self.sources = self.generate_sources(num_sources, workspace_size, intensity_range)
        self.r_s = 0.5  # Source radius
        self.r_d = 0.5  # Detector radius
        self.T = 100  # Transmission factor
        self.workspace_size = workspace_size
        self.x = np.linspace(0, self.workspace_size[0], 200)
        self.y = np.linspace(0, self.workspace_size[1], 200)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.g_truth = self.ground_truth()
        # kernel should be k(r) = sigma**2 * exp(-r / (2 * l**2))
        if kernel_params is None:
            kernel_params = {'sigma': 1, 'l': 1}
        kernel = C(kernel_params['sigma'], (1e-5, 5))**2 * RBF(kernel_params['l'], (1e-5, 50))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        self.gp.max_iter = 100
        if seed is not None:
            np.random.seed(seed)

    def generate_sources(self, num_sources, workspace_size, intensity_range):
        """Generate random sources within the workspace."""
        sources = []
        for _ in range(num_sources):
            rand_x = np.random.uniform(0, workspace_size[0])
            rand_y = np.random.uniform(0, workspace_size[1])
            rand_A = np.random.uniform(*intensity_range)
            sources.append([rand_x, rand_y, rand_A])
        return sources

    def update_source(self, source_index, new_x, new_y, new_A):
        """Update a specific source's parameters."""
        if source_index < len(self.sources):
            self.sources[source_index] = [new_x, new_y, new_A]
            self.recalculate_ground_truth()  # Recalculate ground truth after updating the source
        else:
            print("Source index out of range.")

    def recalculate_ground_truth(self):
        """Recalculate the ground truth based on current source positions."""
        self.g_truth = self.ground_truth()
        
    def get_sources_info(self):
        """Return information about the current sources."""
        return self.sources

    def intensity(self, r):
        """Compute intensity at a point r."""
        I = 0
        for source in self.sources:
            r_n, A_n = np.array(source[:2]), source[2]
            dist = np.linalg.norm(r - r_n)
            if dist <= self.r_s:
                I += A_n / (4 * np.pi * self.r_s**2)
            else:
                I += A_n * self.T / (4 * np.pi * dist**2)
        return I

    def response(self, r):
        """Compute response at a point r."""
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
        Z_true = np.zeros(self.X.shape)
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                r = np.array([self.X[i, j], self.Y[i, j]])
                Z_true[i, j] = self.intensity(r) + 50 * self.response(r)
        return Z_true
    
    def simulate_measurements(self, waypoints, noise_level=0.0001):
        measurements = []
        for wp in waypoints:
            # noise level should affect the correct magnitude of the intensity
            intensity = self.intensity(wp)
            noise = np.random.normal(0, noise_level*intensity)
            measurement = intensity + noise
            measurements.append(measurement)
        return measurements

    def predict_spatial_field(self, waypoints, measurements):
        # update the current GP model with the new measurements
        # convert measurements base 10 log scale
        measurements = np.log10(measurements)
        self.gp.fit(waypoints, measurements)
        # Return the current entire spatial field prediction
        Z_pred = np.zeros(self.X.shape)
        r = []
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                r.append([self.X[i, j], self.Y[i, j]])
        # print("R:", r)
        # # Make sure Z_pred is a matrix with self.X.shape[0] rows and self.X.shape[1] columns
        Z_pred, std = self.gp.predict(r, return_std=True) 
        Z_pred = Z_pred.reshape(self.X.shape[0], self.X.shape[1]) 
        # print("Z_pred:", Z_pred)
        # convert Z_pred back to base 10
        Z_pred = 10**Z_pred
        return Z_pred, std


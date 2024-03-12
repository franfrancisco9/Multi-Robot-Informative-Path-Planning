import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class RadiationField:
    def __init__(self, num_sources=1, workspace_size=(40, 40), intensity_range=(10000, 100000)):
        self.sources = self.generate_sources(num_sources, workspace_size, intensity_range)
        self.r_s = 0.5  # Source radius
        self.r_d = 0.5  # Detector radius
        self.T = 100  # Transmission factor
        self.workspace_size = workspace_size
        self.x = np.linspace(0, self.workspace_size[0], 200)
        self.y = np.linspace(0, self.workspace_size[1], 200)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.g_truth = self.ground_truth()

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
    
    def simulate_measurements(self, waypoints, noise_level=0.5):
        measurements = []
        for wp in waypoints:
            measurement = self.intensity(wp) + np.random.normal(0, noise_level)
            measurements.append(measurement)
        return measurements

    def predict_spatial_field(self, waypoints, measurements, kernel_params=None):
        # Use GP to predict the spatial field in the workspace
        if kernel_params is None:
            kernel_params = {'length_scale': 1.0, 'length_scale_bounds': (1e-2, 1e2)}
        kernel = C(1.0, (1e-2, 1e2)) * RBF(kernel_params['length_scale'], length_scale_bounds=kernel_params['length_scale_bounds'])
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        # print the number of gaussians used
        # print(gp.n_features_in_)

        gp.fit(waypoints, measurements)
        # print("Waypoints:", waypoints)
        Z_pred = np.zeros(self.X.shape)
        r = []
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                r.append([self.X[i, j], self.Y[i, j]])
        # print("R:", r)
        Z_pred = gp.predict(r).reshape(self.X.shape[0], self.X.shape[1])    # Make sure Z_pred is a matrix with self.X.shape[0] rows and self.X.shape[1] columns
        # print("Z_pred:", Z_pred)
        return Z_pred
    
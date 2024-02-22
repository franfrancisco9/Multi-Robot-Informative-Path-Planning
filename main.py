import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from radiation import RadiationField  # Make sure RadiationField is defined in radiation.py
from cma import CMAEvolutionStrategy as CMAES
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from ipp import generate_nominal_path

# Instantiate RadiationField with desired parameters
radiation_field = RadiationField(num_sources=1, workspace_size=(40, 40), intensity_range=(10000, 100000))

# Define the spatial domain
x = np.linspace(0, 40, 200)
y = np.linspace(0, 40, 200)
X, Y = np.meshgrid(x, y)

# Ground truth radiation field generation
Z_true = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        r = np.array([X[i, j], Y[i, j]])  
        Z_true[i, j] = radiation_field.intensity(r) + 50 * radiation_field.response(r)

# Generate nominal path
waypoints = 29
distance_budget = 250
optimal_waypoints = generate_nominal_path(n_waypoints=waypoints, distance_budget=distance_budget)

# Simulate noisy measurements along the path
noise_level = 0.05  # Adjust as necessary
measurements = [radiation_field.intensity(wp) + np.random.normal(0, noise_level) for wp in optimal_waypoints]

# Use Gaussian Process Regression to predict the spatial field
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gpr.fit(optimal_waypoints, measurements)
Z_pred, _ = gpr.predict(np.hstack((X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis])), return_std=True)
Z_pred = Z_pred.reshape(X.shape)

# Define manual levels for the contour plot to cover each order of magnitude from 10^0
max_log_value = np.ceil(np.log10(Z_true.max()))
levels = np.logspace(0, max_log_value, int(max_log_value) + 1)
# Define a colormap with distinct shades of green for each order of magnitude
cmap = plt.get_cmap('Greens_r', len(levels) - 1)  # Ensuring we get a discrete range of greens

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Predicted field with nominal path
cs_pred = axs[1].contourf(X, Y, Z_pred, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
axs[1].plot(optimal_waypoints[:, 0], optimal_waypoints[:, 1], '-wo', label='Nominal Path')  # Overlay the nominal path
fig.colorbar(cs_pred, ax=axs[1], format=ticker.LogFormatterMathtext())
axs[1].set_title('Predicted Radiation Field with Nominal Path')
axs[1].legend()
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')

# Ground truth
cs_true = axs[0].contourf(X, Y, Z_true, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
fig.colorbar(cs_true, ax=axs[0], format=ticker.LogFormatterMathtext())
axs[0].set_title('Ground Truth Radiation Field')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from radiation import RadiationField  # Ensure this class is correctly defined in your radiation.py
from ipp import InformativePathPlanning  # Ensure this class is correctly defined in your ipp.py
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Instantiate RadiationField with desired parameters
radiation_field = RadiationField(num_sources=1, workspace_size=(40, 40), intensity_range=(10000, 100000))
radiation_field.update_source(0, 20, 20, 100000)  # Update source to be at (20, 20) with intensity 100000

# Define the spatial domain for visualization
Z_true = radiation_field.g_truth

# Initialize IPP with parameters and generate nominal path
ipp = InformativePathPlanning(workspace_size=(40, 40), n_waypoints=29, distance_budget=400)
# optimal_waypoints = ipp.generate_nominal_path()
# To test generate 50 waypoints in a grid pattern in the workspace 
optimal_waypoints = np.array([[i, j] for i in range(0, 40, 2) for j in range(0, 40, 2)])
# test = ipp.GenerateNominalPath()
# Given optimal_waypoints generated from IPP
measurements = radiation_field.simulate_measurements(optimal_waypoints, noise_level=0)

# Use Gaussian Process Regression to predict the spatial field
Z_pred = radiation_field.predict_spatial_field(optimal_waypoints, measurements)

# Define manual levels for the contour plot to cover each order of magnitude from 10^0
max_log_value = np.ceil(np.log10(Z_true.max()))
levels = np.logspace(0, max_log_value, int(max_log_value) + 1)
# Define a colormap with distinct shades of green for each order of magnitude
cmap = plt.get_cmap('Greens_r', len(levels) - 1)  # Ensuring we get a discrete range of greens

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Predicted field with nominal path
cs_pred = axs[1].contourf(radiation_field.X, radiation_field.Y, Z_pred, 
                          levels=levels, cmap=cmap, 
                          norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
axs[1].plot(optimal_waypoints[:, 0], optimal_waypoints[:, 1], '-wo', label='Nominal Path')  # Overlay the nominal path
fig.colorbar(cs_pred, ax=axs[1], format=ticker.LogFormatterMathtext())
axs[1].set_title('Predicted Radiation Field with Nominal Path')
axs[1].legend()
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')

# Ground truth
cs_true = axs[0].contourf(radiation_field.X, radiation_field.Y, Z_true, 
                          levels=levels, cmap=cmap, 
                          norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
fig.colorbar(cs_true, ax=axs[0], format=ticker.LogFormatterMathtext())
axs[0].set_title('Ground Truth Radiation Field')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

plt.tight_layout()
plt.show()



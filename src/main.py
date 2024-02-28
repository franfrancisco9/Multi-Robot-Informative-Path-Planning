import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from scipy.interpolate import make_interp_spline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from ipp import InformativePathPlanning
from radiation import RadiationField

# Initialize the scenarios
scenarios = [
    RadiationField(num_sources=0, workspace_size=(40, 40)),
    RadiationField(num_sources=1, workspace_size=(40, 40)),
    RadiationField(num_sources=2, workspace_size=(40, 40)),
    RadiationField(num_sources=7, workspace_size=(40, 40))
]

def plot_scenario(scenario, scenario_number):
    # Generate boustrophedon path
    ipp = InformativePathPlanning(workspace_size=(40, 40), n_waypoints=200, distance_budget=2000)
    ipp.Boustrophedon()
    nominal_path = ipp.nominal_path
    waypoints = ipp.nominal_spread

    # Simulate measurements along the path
    measurements = scenario.simulate_measurements(waypoints)

    # Predict the spatial field based on measurements
    Z_pred = scenario.predict_spatial_field(waypoints, measurements)

    # Determine log scale levels
    Z_true = scenario.ground_truth()
    if Z_true.max() == 0:
        max_log_value = 1
    else:
        max_log_value = np.ceil(np.log10(Z_true.max()))
    levels = np.logspace(0, max_log_value, int(max_log_value) + 1)
    cmap = plt.get_cmap('Greens_r', len(levels) - 1)

    # Plot ground truth and predicted field
    fig, axs = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)

    # Ground truth
    cs_true = axs[0].contourf(scenario.X, scenario.Y, Z_true, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    fig.colorbar(cs_true, ax=axs[0], format=ticker.LogFormatterMathtext())
    axs[0].set_title(f'Scenario {scenario_number} Ground Truth')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')

    # Predicted field
    cs_pred = axs[1].contourf(scenario.X, scenario.Y, Z_pred, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    fig.colorbar(cs_pred, ax=axs[1], format=ticker.LogFormatterMathtext())
    axs[1].set_title(f'Scenario {scenario_number} Predicted Field')

    # Improved path plot
    x_new, y_new = ipp.nominal_path
    axs[1].plot(x_new, y_new, 'b-', label='Boustrophedon Path')
    axs[1].plot(waypoints[:, 0], waypoints[:, 1], 'ro', markersize=5)  # Waypoints
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')

    plt.savefig(f'../images/scenario_{scenario_number}_comparison.png')
    plt.show()

# Plot each scenario
for i, scenario in enumerate(scenarios, start=1):
    plot_scenario(scenario, i)

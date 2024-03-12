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

def plot_scenario(scenario, scenario_number, ipp_path=None):
    # Initialize Informative Path Planning
    ipp = InformativePathPlanning(workspace_size=(40, 40), n_waypoints=200, distance_budget=2000)
    
    # Generate Boustrophedon path as the nominal path
    ipp.Boustrophedon()
    nominal_path = ipp.nominal_path
    waypoints = ipp.nominal_spread
    
    # If an IPP path is provided, use it; otherwise, use the nominal path for visualization
    if ipp_path is not None:
        path_to_use = ipp.IPP()
    else:
        path_to_use = waypoints

    # Simulate measurements along the path
    measurements = scenario.simulate_measurements(path_to_use)

    # Predict the spatial field based on measurements
    Z_pred = scenario.predict_spatial_field(path_to_use, measurements)

    # Ground truth for comparison
    Z_true = scenario.ground_truth()

    # Plotting configurations
    plot_fields(scenario, Z_true, Z_pred, path_to_use, scenario_number, nominal_path)

def plot_fields(scenario, Z_true, Z_pred, path, scenario_number, nominal_path):
    # Determine log scale levels for contour plots
    if Z_true.max() == 0:
        max_log_value = 1
    else:
        max_log_value = np.ceil(np.log10(Z_true.max()))
    levels = np.logspace(0, max_log_value, int(max_log_value) + 1)
    cmap = plt.get_cmap('Greens_r', len(levels) - 1)

    # Setup figure for ground truth and predicted field visualization
    fig, axs = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)

    # Ground Truth Plot
    cs_true = axs[0].contourf(scenario.X, scenario.Y, Z_true, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    fig.colorbar(cs_true, ax=axs[0], format=ticker.LogFormatterMathtext())
    axs[0].set_title(f'Scenario {scenario_number} Ground Truth')

    # Predicted Field Plot
    cs_pred = axs[1].contourf(scenario.X, scenario.Y, Z_pred, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    fig.colorbar(cs_pred, ax=axs[1], format=ticker.LogFormatterMathtext())
    axs[1].set_title(f'Scenario {scenario_number} Predicted Field')

    x_new, y_new = nominal_path
    axs[1].plot(x_new, y_new, 'b-', label='Boustrophedon Path')
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # add the source locations with red X's
    sources = scenario.get_sources_info()
    for source in sources:
        axs[1].plot(source[0], source[1], 'rX', markersize=10, label='Source')
    # Common configurations for both subplots
    for ax in axs:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.plot(path[:, 0], path[:, 1], 'ro', markersize=5, label='Waypoints')  # Waypoints visualization
        ax.set_facecolor(cmap(0))

    plt.savefig(f'../images/scenario_ipp_{scenario_number}_comparison.png')
    plt.show()

# Main execution loop for plotting scenarios
if __name__ == "__main__":
    for i, scenario in enumerate(scenarios, start=1):
        # Here you can call IPP to generate a path, for now, we use None to use the nominal path
        ipp_path = None  # Placeholder for IPP-generated path
        plot_scenario(scenario, i, ipp_path=ipp_path)

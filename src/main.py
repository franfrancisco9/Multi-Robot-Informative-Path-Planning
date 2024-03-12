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
    # RadiationField(num_sources=0, workspace_size=(40, 40)),
    RadiationField(num_sources=1, workspace_size=(40, 40)),
    RadiationField(num_sources=2, workspace_size=(40, 40)),
    RadiationField(num_sources=7, workspace_size=(40, 40))
]

# set the source to 20 20 and intensity to 100000
scenarios[0].update_source(0, 20, 20, 100000)

def helper_plot(scenario, scenario_number, Z_true, Z_pred, std, current_waypoints, nominal_path):
    # Determine log scale levels
    if Z_true.max() == 0:
        max_log_value = 1
    else:
        max_log_value = np.ceil(np.log10(Z_true.max()))
    levels = np.logspace(0, max_log_value, int(max_log_value) + 1)
    cmap = plt.get_cmap('Greens_r', len(levels) - 1)

    # Plot ground truth and predicted field
    fig, axs = plt.subplots(2, 2, figsize=(20, 8), constrained_layout=True)
    
    # Ground truth
    cs_true = axs[0][0].contourf(scenario.X, scenario.Y, Z_true, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    fig.colorbar(cs_true, ax=axs[0][0], format=ticker.LogFormatterMathtext())
    axs[0][0].set_title(f'Scenario {scenario_number} Ground Truth')
    axs[0][0].set_xlabel('x')
    axs[0][0].set_ylabel('y')
    # make the background the colour of the lowest contour level
    axs[0][0].set_facecolor(cmap(0))

    # Predicted field
    cs_pred = axs[0][1].contourf(scenario.X, scenario.Y, Z_pred, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    fig.colorbar(cs_pred, ax=axs[0][1], format=ticker.LogFormatterMathtext())
    axs[0][1].set_title(f'Scenario {scenario_number} Predicted Field')
    x_new, y_new = nominal_path
    axs[0][1].plot(x_new, y_new, 'b-', label='Boustrophedon Path')
    # add the waypoints with red circles
    axs[0][1].plot(current_waypoints[:, 0], current_waypoints[:, 1], 'ro', markersize=5)  # Waypoints
    # make the background the colour of the lowest contour level
    axs[0][1].set_facecolor(cmap(0))
    sources = scenario.get_sources_info()
    for source in sources:
        axs[0][1].plot(source[0], source[1], 'rX', markersize=10, label='Source')

    # Uncertainty field
    std = std.reshape(scenario.X.shape[0], scenario.X.shape[1])
    cs_uncertainty = axs[1][0].contourf(scenario.X, scenario.Y, std, cmap='Reds')
    fig.colorbar(cs_uncertainty, ax=axs[1][0])
    axs[1][0].set_title(f'Scenario {scenario_number} Uncertainty Field')
    # make the background the colour of the lowest contour level
    axs[1][0].set_facecolor('pink')

    plt.show()
    # save into ../gp_test/ folder
    # plt.savefig(f'../gp_test/scenario_{scenario_number}_waypoints_{len(current_waypoints)}.png')
    # plt.close()
    print("Tested waypoints: ", len(current_waypoints), " for scenario ", scenario_number)


def plot_scenario(scenario, scenario_number):
    # Generate boustrophedon path
    ipp = InformativePathPlanning(workspace_size=(40, 40), n_waypoints=200, distance_budget=2000)
    ipp.Boustrophedon()
    nominal_path = ipp.nominal_path
    waypoints = ipp.nominal_spread

    # Starting from the first two waypoints
    # Get the measurements and call the predict_spatial_field method and plot the ground truth and predicted field
    # and additionally plot the uncertainty field using the std returned by the predict_spatial_field method
    # to plot call the helper_plot method
    # current_waypoints = waypoints[:2]
    # measurements = scenario.simulate_measurements(current_waypoints)
    # Z_pred, std = scenario.predict_spatial_field(current_waypoints, measurements)
    # Z_true = scenario.ground_truth()
    # helper_plot(scenario, scenario_number, Z_true, Z_pred, std, current_waypoints)
    # for i in range(2, len(waypoints)):
    #     # Add the next waypoint to the current waypoints
    #     current_waypoints = waypoints[:i]
    #     # Get the measurements and call the predict_spatial_field method and plot the ground truth and predicted field
    #     # and additionally plot the uncertainty field using the std returned by the predict_spatial_field method
    #     # to plot call the helper_plot method
    #     measurements = scenario.simulate_measurements(current_waypoints)
    #     Z_pred, std = scenario.predict_spatial_field(current_waypoints, measurements)
    #     Z_true = scenario.ground_truth()
    #     helper_plot(scenario, scenario_number, Z_true, Z_pred, std, current_waypoints)
    # Simulate measurements along the path
    measurements = scenario.simulate_measurements(waypoints)
    # Predict the spatial field based on measurements
    Z_pred, std = scenario.predict_spatial_field(waypoints, measurements)

    # Determine log scale levels
    Z_true = scenario.ground_truth()
    helper_plot(scenario, scenario_number, Z_true, Z_pred, std, waypoints, nominal_path)
    # if Z_true.max() == 0:
    #     max_log_value = 1
    # else:
    #     max_log_value = np.ceil(np.log10(Z_true.max()))
    # levels = np.logspace(0, max_log_value, int(max_log_value) + 1)
    # cmap = plt.get_cmap('Greens_r', len(levels) - 1)

    # # Plot ground truth and predicted field
    # fig, axs = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
    
    # # Ground truth
    # cs_true = axs[0].contourf(scenario.X, scenario.Y, Z_true, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    # fig.colorbar(cs_true, ax=axs[0], format=ticker.LogFormatterMathtext())
    # axs[0].set_title(f'Scenario {scenario_number} Ground Truth')
    # axs[0].set_xlabel('x')
    # axs[0].set_ylabel('y')
    # # make the background the colour of the lowest contour level
    # axs[0].set_facecolor(cmap(0))
    
    # if Z_pred.max() == 0:
    #     max_log_value = 1
    # else:
    #     max_log_value = np.ceil(np.log10(Z_pred.max()))
    # levels = np.logspace(0, max_log_value, int(max_log_value) + 1)
    # cmap = plt.get_cmap('Greens_r', len(levels) - 1)

    # # Predicted field
    # cs_pred = axs[1].contourf(scenario.X, scenario.Y, Z_pred, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    # fig.colorbar(cs_pred, ax=axs[1], format=ticker.LogFormatterMathtext())
    # axs[1].set_title(f'Scenario {scenario_number} Predicted Field')
    # # make the background the colour of the lowest contour level
    # # Improved path plot
    # x_new, y_new = ipp.nominal_path
    # axs[1].plot(x_new, y_new, 'b-', label='Boustrophedon Path')
    # axs[1].plot(waypoints[:, 0], waypoints[:, 1], 'ro', markersize=5)  # Waypoints
    # # add the source locations with red X's
    # sources = scenario.get_sources_info()
    # for source in sources:
    #     axs[1].plot(source[0], source[1], 'rX', markersize=10, label='Source')
    # axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # axs[1].set_xlabel('x')
    # axs[1].set_ylabel('y')

    # axs[1].set_facecolor(cmap(0))
    # plt.savefig(f'../images/scenario_{scenario_number}_comparison.png')
    # plt.show()

# Plot each scenario
for i, scenario in enumerate(scenarios, start=1):
    plot_scenario(scenario, i)

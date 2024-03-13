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

# Iniatilize a RMSE list of lists
RMSE_list =  [[] for _ in range(len(scenarios))]

def helper_plot(scenario, scenario_number, Z_true, Z_pred, std, current_waypoints, nominal_path, RMSE_list):
    # Determine log scale levels,
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
    # fig.colorbar(cs_uncertainty, ax=axs[1][0])
    axs[1][0].set_title(f'Scenario {scenario_number} Uncertainty Field')
    # make the background the colour of the lowest contour level
    axs[1][0].set_facecolor('pink')

    # Plot RMSE as vertical y with median as a dot and std as error bars
    # x label is just the scenario number and title is 10 run average RMSE
    # Only one x label for each scenario
    axs[1][1].errorbar(scenario_number, np.mean(RMSE_list), yerr=np.std(RMSE_list), fmt='o', linewidth=2, capsize=6)
    axs[1][1].set_title('5 run average RMSE')
    axs[1][1].set_xlabel('Scenario' + str(scenario_number))
    # only show one tick for the x axis
    axs[1][1].set_xticks([scenario_number])
    axs[1][1].set_ylabel('RMSE')

    plt.savefig(f'../images/scenario_full_RMSE_{scenario_number}_comparison_high_error.png')
    plt.show()
    plt.close()
    print("Tested waypoints: ", len(current_waypoints), " for scenario ", scenario_number)


def run_scenario(scenario, scenario_number, final = False):
    ipp = InformativePathPlanning(workspace_size=(40, 40), n_waypoints=200, distance_budget=2000)
    ipp.Boustrophedon()
    nominal_path = ipp.nominal_path
    waypoints = ipp.nominal_spread
    measurements = scenario.simulate_measurements(waypoints)
    Z_pred, std = scenario.predict_spatial_field(waypoints, measurements)
    Z_true = scenario.ground_truth()
    # add RMSE 
    RMSE = np.sqrt(1 / Z_true.size * np.sum((Z_true - Z_pred)**2))
    # Normalize the RMSE between 0 and 1
    RMSE = RMSE / (Z_true.max() - Z_true.min())
    print("RMSE: ", RMSE)
    RMSE_list[scenario_number - 1].append(RMSE)
    if final:
        helper_plot(scenario, scenario_number, Z_true, Z_pred, 
                    std, waypoints, nominal_path, RMSE_list[scenario_number - 1])



if __name__ == '__main__':
    # Run each scenario 10 times in the final pass final = True
    ROUNDS = 5
    for i in range(ROUNDS):
        print("Run ", i)
        for j, scenario in enumerate(scenarios, start=1):
            if i == ROUNDS - 1:
                run_scenario(scenario, j, True)
            else:
                run_scenario(scenario, j, False)

    

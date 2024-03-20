import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors

from boustrophedon import Boustrophedon
from radiation import RadiationField
from randomwalker import RandomWalker 
from informative import InformativePathPlanning
from RRT import InformativeRRTPathPlanning, BetaInformativeRRTPathPlanning

# Initialize the scenarios
scenarios = [
    # RadiationField(num_sources=0, workspace_size=(40, 40)),
    RadiationField(num_sources=1, workspace_size=(40, 40), seed=42),
    RadiationField(num_sources=2, workspace_size=(40, 40), seed=42),
    RadiationField(num_sources=7, workspace_size=(40, 40), seed=42),
]

# set the source to 20 20 and intensity to 100000
scenarios[0].update_source(0, 20, 20, 100000)


# Iniatilize a RMSE list of lists
RMSE_list_boust =  [[] for _ in range(len(scenarios))]
RMSE_list_random = [[] for _ in range(len(scenarios))]
RMSE_list_informative = [[] for _ in range(len(scenarios))]
RMSE_list_RRT = [[] for _ in range(len(scenarios))]

ROUNDS = 1

def helper_plot(scenario, scenario_number, Z_true, Z_pred, std, path, RMSE_list):
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
    x_new, y_new = path.full_path
    axs[0][1].plot(x_new, y_new, 'b-', label=path.name + ' Path')
    # add the waypoints with red circles
    # print(path.obs_wp)
    axs[0][1].plot(path.obs_wp[:, 0], path.obs_wp[:, 1], 'ro', markersize=5)  # Waypoints
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
    axs[1][1].set_title(f'{ROUNDS} Run Average RMSE')
    axs[1][1].set_xlabel('Scenario' + str(scenario_number))
    # only show one tick for the x axis
    axs[1][1].set_xticks([scenario_number])
    axs[1][1].set_ylabel('RMSE')

    plt.savefig(f'../images/scenario_{scenario_number}_run_{ROUNDS}_path_{path.name}_beta_50.png')
    plt.show()
    plt.close()
    print("Tested waypoints: ", len(path.obs_wp), " for scenario ", scenario_number)

def run_Boustrophedon_scenario(scenario, scenario_number, final = False):
    boust = Boustrophedon(d_waypoint_distance=2.5)

    measurements = scenario.simulate_measurements(boust.obs_wp)
    Z_pred, std = scenario.predict_spatial_field(boust.obs_wp, measurements)
    Z_true = scenario.ground_truth()

    # add RMSE 
    RMSE = np.sqrt(1 / Z_true.size * np.sum((np.log10(Z_true + 1) - np.log10(Z_pred + 1))**2))
    print("RMSE: ", RMSE)
    RMSE_list_boust[scenario_number - 1].append(RMSE)
    if final:
        helper_plot(scenario, scenario_number, Z_true, Z_pred, 
                    std, boust, RMSE_list_boust[scenario_number - 1])

def run_Random_Scenario(scenario, scenario_number, final=False):
    random_walker = RandomWalker(d_waypoint_distance=2.5)

    measurements = scenario.simulate_measurements(random_walker.obs_wp)
    Z_pred, std = scenario.predict_spatial_field(random_walker.obs_wp, measurements)
    Z_true = scenario.ground_truth()

    # Calculate RMSE
    RMSE = np.sqrt(np.mean((np.log10(Z_true + 1) - np.log10(Z_pred + 1))**2))
    print("RMSE: ", RMSE)
    RMSE_list_random[scenario_number - 1].append(RMSE)
    if final:
        helper_plot(scenario, scenario_number, Z_true, Z_pred, std, random_walker, RMSE_list_random[scenario_number - 1])

def run_Informative_Scenario(scenario, scenario_number, final=False):
    informative_path = InformativePathPlanning(scenario, n_waypoints=200, d_waypoint_distance=2.5)
    Z_pred, std = informative_path.run()
    
    # Assuming you want to visualize the results as in other scenarios
    # Z_pred, std = scenario.predict_spatial_field(np.array(informative_path.obs_wp), scenario.simulate_measurements(informative_path.obs_wp))
    Z_true = scenario.ground_truth()
    
    RMSE = np.sqrt(np.mean((np.log10(Z_true + 1) - np.log10(Z_pred + 1))**2))
    print("RMSE: ", RMSE)
    RMSE_list_informative[scenario_number - 1].append(RMSE)
    if final:
        helper_plot(scenario, scenario_number, Z_true, Z_pred, std, informative_path, RMSE_list_informative[scenario_number - 1])

def run_InformativeRRT_Scenario(scenario, scenario_number, final=False):
    rrt_path = InformativeRRTPathPlanning(scenario, n_waypoints=20, d_waypoint_distance=2.5)
    Z_pred, std = rrt_path.run()
    
    Z_true = scenario.ground_truth()
    RMSE = np.sqrt(np.mean((np.log10(Z_true + 1) - np.log10(Z_pred + 1))**2))
    print("RRT RMSE: ", RMSE)
    RMSE_list_RRT[scenario_number - 1].append(RMSE)  # Make sure this list is defined
    
    if final:
        helper_plot(scenario, scenario_number, Z_true, Z_pred, std, rrt_path, RMSE_list_RRT[scenario_number - 1])

def run_BetaInformativeRRT_Scenario(scenario, scenario_number, final=False):
    rrt_path = BetaInformativeRRTPathPlanning(scenario, n_waypoints=40, d_waypoint_distance=2.5, beta_t = 50)
    Z_pred, std = rrt_path.run()
    
    Z_true = scenario.ground_truth()
    RMSE = np.sqrt(np.mean((np.log10(Z_true + 1) - np.log10(Z_pred + 1))**2))
    print("RRT RMSE: ", RMSE)
    RMSE_list_RRT[scenario_number - 1].append(RMSE)  # Make sure this list is defined
    
    if final:
        helper_plot(scenario, scenario_number, Z_true, Z_pred, std, rrt_path, RMSE_list_RRT[scenario_number - 1])	

if __name__ == '__main__':
    for i in range(ROUNDS):
        print("Run ", i)    
        for j, scenario in enumerate(scenarios, start=1):
            if i == ROUNDS - 1:
                # run_Boustrophedon_scenario(scenario, j, True)
                # run_Random_Scenario(scenario, j, True)
                # run_Informative_Scenario(scenario, j, True)
                # run_InformativeRRT_Scenario(scenario, j, True)
                run_BetaInformativeRRT_Scenario(scenario, j, True)
            else:
                # run_Boustrophedon_scenario(scenario, j, False)
                # run_Random_Scenario(scenario, j, False)
                # run_Informative_Scenario(scenario, j, False)
                # run_InformativeRRT_Scenario(scenario, j, False)
                run_BetaInformativeRRT_Scenario(scenario, j, False)

    

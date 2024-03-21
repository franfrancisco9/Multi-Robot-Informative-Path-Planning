import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors

from boustrophedon import Boustrophedon
from radiation import RadiationField
from randomwalker import RandomWalker 
from informative import InformativePathPlanning
from RRT import InformativeRRTPathPlanning, BetaInformativeRRTPathPlanning, BiasInformativeRRTPathPlanning

from path_planning_utils import helper_plot

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
RMSE_list_RRT_BIAS = [[] for _ in range(len(scenarios))]
RMSE_list_RRT_BETA = [[] for _ in range(len(scenarios))]
# Number of rounds to run
ROUNDS = 3

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
                    std, boust, RMSE_list_boust[scenario_number - 1], ROUNDS)

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
        helper_plot(scenario, scenario_number, Z_true, Z_pred, std, random_walker, RMSE_list_random[scenario_number - 1], ROUNDS)

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
        helper_plot(scenario, scenario_number, Z_true, Z_pred, std, informative_path, RMSE_list_informative[scenario_number - 1], ROUNDS)

def run_InformativeRRT_Scenario(scenario, scenario_number, final=False):
    rrt_path = InformativeRRTPathPlanning(scenario, n_waypoints=20, d_waypoint_distance=2.5, beta_t = 50)
    Z_pred, std = rrt_path.run()
    
    Z_true = scenario.ground_truth()
    RMSE = np.sqrt(np.mean((np.log10(Z_true + 1) - np.log10(Z_pred + 1))**2))
    print("RRT RMSE: ", RMSE)
    RMSE_list_RRT[scenario_number - 1].append(RMSE)  # Make sure this list is defined
    
    if final:
        helper_plot(scenario, scenario_number, Z_true, Z_pred, std, rrt_path, RMSE_list_RRT[scenario_number - 1], ROUNDS)

def run_BiasInformativeRRT_Scenario(scenario, scenario_number, final=False):
    rrt_path = BiasInformativeRRTPathPlanning(scenario, n_waypoints=20, d_waypoint_distance=2.5, beta_t = 50)
    Z_pred, std = rrt_path.run()
    
    Z_true = scenario.ground_truth()
    RMSE = np.sqrt(np.mean((np.log10(Z_true + 1) - np.log10(Z_pred + 1))**2))
    print("RRT RMSE: ", RMSE)
    RMSE_list_RRT_BIAS[scenario_number - 1].append(RMSE)  # Make sure this list is defined
    
    if final:
        helper_plot(scenario, scenario_number, Z_true, Z_pred, std, rrt_path, RMSE_list_RRT_BIAS[scenario_number - 1], ROUNDS)

def run_BetaInformativeRRT_Scenario(scenario, scenario_number, final=False):
    rrt_path = BetaInformativeRRTPathPlanning(scenario, n_waypoints=40, d_waypoint_distance=2.5, beta_t = 50)
    Z_pred, std = rrt_path.run()
    
    Z_true = scenario.ground_truth()
    RMSE = np.sqrt(np.mean((np.log10(Z_true + 1) - np.log10(Z_pred + 1))**2))
    print("RRT RMSE: ", RMSE)
    RMSE_list_RRT_BETA[scenario_number - 1].append(RMSE)  # Make sure this list is defined
    
    if final:
        helper_plot(scenario, scenario_number, Z_true, Z_pred, std, rrt_path, RMSE_list_RRT_BETA[scenario_number - 1], ROUNDS)

if __name__ == '__main__':
    for i in range(ROUNDS):
        print("Run ", i)    
        for j, scenario in enumerate(scenarios, start=1):
            if i == ROUNDS - 1:
                run_Boustrophedon_scenario(scenario, j, True)
                run_Random_Scenario(scenario, j, True)
                run_Informative_Scenario(scenario, j, True)
                run_InformativeRRT_Scenario(scenario, j, True)
                run_BiasInformativeRRT_Scenario(scenario, j, True)
                run_BetaInformativeRRT_Scenario(scenario, j, True)
            else:
                run_Boustrophedon_scenario(scenario, j, False)
                run_Random_Scenario(scenario, j, False)
                run_Informative_Scenario(scenario, j, False)
                run_InformativeRRT_Scenario(scenario, j, False)
                run_BiasInformativeRRT_Scenario(scenario, j, False)
                run_BetaInformativeRRT_Scenario(scenario, j, False)

    

import numpy as np
import argparse
import json
from tqdm import tqdm

from boustrophedon import Boustrophedon
from radiation import RadiationField
from informative import InformativePathPlanning
from RRT import StrategicRRTPathPlanning, NaiveRRTPathPlanning, BiasRRTPathPlanning, BiasBetaRRTPathPlanning, AdaptiveRRTPathPlanning, InformativeRRTPathPlanning
from path_planning_utils import helper_plot

# Argument parsing
parser = argparse.ArgumentParser(description="Run path planning scenarios.")
parser.add_argument('-config', '--config', required=True, help="Path to the configuration JSON file.")
args = parser.parse_args()

# Load configuration from JSON
with open(args.config, 'r') as file:
    config = json.load(file)

# Extract general args from config if needed
args = config["args"]

# Initialize scenarios based on config
scenarios = []
for scenario_config in config["scenarios"]:
    scenario_class = RadiationField
    # add seed to scenario config
    scenario_config["params"]["seed"] = args["seed"]
    scenario = scenario_class(**scenario_config["params"])
    if "specific_params" in scenario_config:
        for key, value in scenario_config["specific_params"].items():
            scenario.update_source(int(key)-1, value[0], value[1], value[2])
    scenarios.append(scenario)

# Initialize strategies based on config
strategy_constructors = {
    strategy_name: globals()[strategy_name] for strategy_name in config["strategies"]
}

# for each constructor check the possible arguments and those found in args and pass them to the constructor
for strategy_name, constructor in strategy_constructors.items():
    # get from args the arguments that are in the constructor
    constructor_args = {k: v for k, v in args.items() if k in constructor.__init__.__code__.co_varnames}
    # get from config the arguments that are in the constructor
    constructor_args.update({k: v for k, v in args.items() if k in constructor.__init__.__code__.co_varnames})
    # create the strategy with the arguments
    strategy_constructors[strategy_name] = lambda scenario, constructor=constructor, constructor_args=constructor_args: constructor(scenario, **constructor_args)


# Initialize RMSE lists
RMSE_lists = {strategy_name: [] for strategy_name in config["strategies"]}

# Run simulations
with tqdm(total=args["rounds"] * len(scenarios) * len(strategy_constructors), desc="Overall Progress") as pbar:
    for scenario_idx, scenario in enumerate(scenarios, start=1):
        RMSE_lists = {strategy_name: [] for strategy_name in strategy_constructors}
        for round_number in range(1, args["rounds"] + 1):
            for strategy_name, constructor in strategy_constructors.items():
                strategy = constructor(scenario)
                tqdm.write(f"Round {round_number}/{args["rounds"]}, Scenario {scenario_idx}, Strategy: {strategy_name}")
                Z_pred, std = strategy.run()
                Z_true = scenario.ground_truth()
                RMSE = np.sqrt(np.mean((np.log10(Z_true + 1) - np.log10(Z_pred + 1))**2))
                tqdm.write(f"{strategy_name} RMSE: {RMSE}")
                RMSE_lists[strategy_name].append(RMSE)
                # Call helper_plot on the last round if save flag is set
                if round_number == args["rounds"]:
                    helper_plot(scenario, scenario_idx, Z_true, Z_pred, std, strategy, RMSE_lists[strategy_name], args["rounds"], save=args["save"], show=args["show"])
                
                pbar.update(1)


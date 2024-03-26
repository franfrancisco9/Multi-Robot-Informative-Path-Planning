import numpy as np
import argparse
import json
from tqdm import tqdm

# Assuming the modules are correctly implemented
from boustrophedon import Boustrophedon
from radiation import RadiationField
from informative import InformativePathPlanning
from RRT import (StrategicRRTPathPlanning, NaiveRRTPathPlanning, BiasRRTPathPlanning, 
                 BiasBetaRRTPathPlanning, AdaptiveRRTPathPlanning, InformativeRRTPathPlanning)
from path_planning_utils import helper_plot

def load_configuration(config_path):
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found.")
        exit(1)
    except json.JSONDecodeError:
        print("Error decoding the configuration file. Please check its format.")
        exit(1)

def initialize_scenarios(config):
    """Initialize scenarios based on the configuration."""
    scenarios = []
    for scenario_config in config["scenarios"]:
        scenario_class = RadiationField
        scenario = scenario_class(**scenario_config["params"])
        if "specific_params" in scenario_config:
            for key, value in scenario_config["specific_params"].items():
                scenario.update_source(int(key)-1, *value)
        scenarios.append(scenario)
    return scenarios

def initialize_strategies(config, args):
    """Initialize strategies based on the configuration."""
    strategy_constructors = {
        strategy_name: globals()[strategy_name] for strategy_name in config["strategies"]
    }
    strategy_instances = {}
    for strategy_name, constructor in strategy_constructors.items():
        # Filter constructor arguments from both args and the config
        constructor_args = {k: v for k, v in args.items() if k in constructor.__init__.__code__.co_varnames}
        strategy_instances[strategy_name] = lambda scenario, constructor=constructor, constructor_args=constructor_args: constructor(scenario, **constructor_args)
    return strategy_instances

def run_simulations(scenarios, strategy_constructors, args):
    """Run simulations for all scenarios and strategies."""
    RMSE_lists = {strategy_name: [] for strategy_name in strategy_constructors}
    with tqdm(total=args["rounds"] * len(scenarios) * len(strategy_constructors), desc="Overall Progress") as pbar:
        for scenario_idx, scenario in enumerate(scenarios, start=1):
            for round_number in range(1, args["rounds"] + 1):
                for strategy_name, constructor in strategy_constructors.items():
                    strategy = constructor(scenario)
                    tqdm.write(f"Round {round_number}/{args['rounds']}, Scenario {scenario_idx}, Strategy: {strategy_name}")
                    Z_pred, std = strategy.run()
                    Z_true = scenario.ground_truth()
                    RMSE = np.sqrt(np.mean((np.log10(Z_true + 1) - np.log10(Z_pred + 1))**2))
                    tqdm.write(f"{strategy_name} RMSE: {RMSE}")
                    RMSE_lists[strategy_name].append(RMSE)
                    if round_number == args["rounds"] and args.get("save", False):
                        helper_plot(scenario, scenario_idx, Z_true, Z_pred, std, strategy, RMSE_lists[strategy_name], args["rounds"], save=args["save"], show=args["show"])
                    pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Run path planning scenarios.")
    parser.add_argument('-config', '--config', required=True, help="Path to the configuration JSON file.")
    args = parser.parse_args()

    config = load_configuration(args.config)
    scenarios = initialize_scenarios(config)
    strategy_instances = initialize_strategies(config, config["args"])
    run_simulations(scenarios, strategy_instances, config["args"])

if __name__ == "__main__":
    main()

import numpy as np
import argparse
import json
from tqdm import tqdm

# Assuming the modules are correctly implemented
from boustrophedon import Boustrophedon
from radiation import RadiationField
from informative import InformativePathPlanning
from RRT import (StrategicRRTPathPlanning, BaseRRTPathPlanning, BiasRRTPathPlanning, 
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
        # Add seed to the scenario configuration if present in the main configuration
        if "seed" in config["args"]:
            scenario_config["params"]["seed"] = config["args"]["seed"]
        scenario = RadiationField(**scenario_config["params"])
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
        # Retrieve constructor arguments for the base class and subclass
        base_args = {}
        if hasattr(constructor, '__bases__'):
            for base in constructor.__bases__:
                if hasattr(base, '__init__') and hasattr(base.__init__, '__code__'):
                    base_args.update({k: v for k, v in args.items() if k in base.__init__.__code__.co_varnames})
                    for base_base in base.__bases__:
                        if hasattr(base_base, '__init__') and hasattr(base_base.__init__, '__code__'):
                            base_args.update({k: v for k, v in args.items() if k in base_base.__init__.__code__.co_varnames})

        # Merge args for the subclass
        constructor_args = {**base_args, **{k: v for k, v in args.items() if k in constructor.__init__.__code__.co_varnames}}
        print(f"Strategy: {strategy_name}, Args: {constructor_args}")
        def make_strategy(scenario, constructor=constructor, constructor_args=constructor_args):
            return constructor(scenario, **constructor_args)

        strategy_instances[strategy_name] = make_strategy

    return strategy_instances


def run_simulations(scenarios, strategy_constructors, args):
    """Run simulations for all scenarios and strategies."""
    RMSE_lists = {strategy_name: [] for strategy_name in strategy_constructors}
    with tqdm(total=args["rounds"] * len(scenarios) * len(strategy_constructors), desc="Overall Progress") as pbar:
        for scenario_idx, scenario in enumerate(scenarios, start=1):
            print("#" * 80)
            RMSE_lists = {strategy_name: [] for strategy_name in strategy_constructors}
            for round_number in range(1, args["rounds"] + 1):
                for strategy_name, constructor in strategy_constructors.items():
                    strategy = constructor(scenario)
                    tqdm.write(f"Round {round_number}/{args['rounds']}, Scenario {scenario_idx}, Strategy: {strategy_name}")
                    Z_pred, std = strategy.run()
                    Z_true = scenario.ground_truth()
                    RMSE = np.sqrt(np.mean((np.log10(Z_true + 1) - np.log10(Z_pred + 1))**2))
                    tqdm.write(f"{strategy_name} RMSE: {RMSE}")
                    RMSE_lists[strategy_name].append(RMSE)
                    if round_number == args["rounds"]:
                        helper_plot(scenario, scenario_idx, Z_true, Z_pred, std, strategy, RMSE_lists[strategy_name], args["rounds"], save=args["save"], show=args["show"])
                    pbar.update(1)
            # print current RMSE for each strategy organized from lowest to highest regarding the average RMSE
            print(f"Scenario {scenario_idx} RMSE:")
            # sort the RMSE lists based on the average RMSE
            sorted_RMSE = sorted(RMSE_lists.items(), key=lambda x: np.mean(x[1]))
            for strategy_name, RMSE_list in sorted_RMSE:
                print(f"{strategy_name}: {np.mean(RMSE_list)}")

def main():
    parser = argparse.ArgumentParser(description="Run path planning scenarios.")
    parser.add_argument('-config', '--config', required=True, help="Path to the configuration JSON file.")
    args = parser.parse_args()
    print("#" * 80)
    print(f"Loading configuration from {args.config}")
    config = load_configuration(args.config)
    print("Configuration loaded successfully.")
    print("#" * 80)
    print("Arguments:")
    for key, value in config["args"].items():
        print(f"{key}: {value}")
    print("#" * 80)
    print("Loading scenarios and strategies...")
    scenarios = initialize_scenarios(config)
    strategy_instances = initialize_strategies(config, config["args"])
    print("Scenarios and strategies loaded successfully.")
    print("#" * 80)
    print("Running simulations...")
    run_simulations(scenarios, strategy_instances, config["args"])
    print("#" * 80)
    print("Simulations completed successfully.")
    print("#" * 80)
if __name__ == "__main__":
    main()

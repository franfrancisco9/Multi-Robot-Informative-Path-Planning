import numpy as np
import argparse
import json
from tqdm import tqdm

# Assuming the modules are correctly implemented
from boustrophedon import *
from radiation import RadiationField
from informative import *
from RRT import *
from path_planning_utils import helper_plot, calculate_differential_entropy, save_run_info, run_number_from_folder, calculate_source_errors

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
            if config["args"]["seed"] == -1:
                scenario_config["params"]["seed"] = None
            else:
                scenario_config["params"]["seed"] = config["args"]["seed"]
                
        scenario = RadiationField(**scenario_config["params"])
        if "specific_params" in scenario_config:
            for key, value in scenario_config["specific_params"].items():
                scenario.update_source(int(key)-1, *value)
        scenarios.append(scenario)
    return scenarios

def get_constructor_args(cls, args):
    """
    Recursively collects constructor arguments from the given class and its base classes.
    
    Parameters:
    - cls: The class to collect arguments for.
    - args: Dictionary of available arguments.
    
    Returns:
    - A dictionary of arguments that can be used to instantiate the class.
    """
    constructor_args = {}
    if hasattr(cls, '__init__') and hasattr(cls.__init__, '__code__'):
        # Collect arguments for the current class's __init__ method
        constructor_args.update({k: v for k, v in args.items() if k in cls.__init__.__code__.co_varnames})

    # Recursively collect arguments from base classes
    for base in getattr(cls, '__bases__', []):
        constructor_args.update(get_constructor_args(base, args))
    
    return constructor_args

def initialize_strategies(config, args):
    """
    Initialize strategies based on the configuration, using recursion to collect constructor arguments.
    
    Parameters:
    - config: Configuration dictionary containing strategy classes.
    - args: Dictionary of available arguments.
    
    Returns:
    - Dictionary of strategy instances keyed by their names.
    """
    strategy_constructors = {strategy_name: globals()[strategy_name] for strategy_name in config["strategies"]}
    strategy_instances = {}

    for strategy_name, constructor in strategy_constructors.items():
        constructor_args = get_constructor_args(constructor, args)
        print(f"Strategy: {strategy_name}, Args: {constructor_args}")
        
        def make_strategy(scenario, constructor=constructor, constructor_args=constructor_args):
            return constructor(scenario, **constructor_args)

        strategy_instances[strategy_name] = make_strategy

    return strategy_instances


def run_simulations(scenarios, strategy_instances, args):
    """Run simulations for all scenarios and strategies."""
    if args["run_number"] == -1:
        run_number = run_number_from_folder()
    else:
        run_number = args["run_number"]
    print(f"Run number: {run_number}")
    RMSE_per_scenario = {}
    WRMSE_per_scenario = {}
    Diff_Entropy_per_scenario = {}
    TIME_per_scenario = {}
    Source_per_scenario = {}

    with tqdm(total=args["rounds"] * len(scenarios) * len(strategy_instances), desc="Overall Progress") as pbar:
        for scenario_idx, scenario in enumerate(scenarios, start=1):
            print("#" * 80)
            # Temporary storage for the current scenario's RMSE and differential entropy
            RMSE_lists = {strategy_name: [] for strategy_name in strategy_instances}
            WRMSE_lists = {strategy_name: [] for strategy_name in strategy_instances}
            Diff_Entropy_lists = {strategy_name: [] for strategy_name in strategy_instances}
            TIME_lists = {strategy_name: [] for strategy_name in strategy_instances}
            Source_lists = {strategy_name: {'source':[], 'n_sources': []} for strategy_name in strategy_instances}
            for round_number in range(1, args["rounds"] + 1):
                for strategy_name, constructor in strategy_instances.items():
                    strategy = constructor(scenario)
                    tqdm.write(f"Round {round_number}/{args['rounds']}, Scenario {scenario_idx}, Strategy: {strategy_name}")
                    Z_pred, std = strategy.run()
                    Z_true = scenario.ground_truth()
                    RMSE = np.sqrt(np.mean((np.log10(Z_true + 1) - np.log10(Z_pred + 1))**2))
                    WEIGHTED_RMSE = np.sqrt(np.sum((Z_true * (np.log10(Z_true + 1) - np.log10(Z_pred + 1))**2)) / np.sum(Z_true))

                    Diff_Entropy = calculate_differential_entropy(std)
                    TIME = strategy.time_taken if hasattr(strategy, 'time_taken') else None
                    # Obtain the estimated locations, number of sources, and theta samples
                    estimated_locs = strategy.best_estimates if hasattr(strategy, 'best_estimates') else []
                    estimated_locs = np.array(estimated_locs).reshape(-1, 3)
                    Source_lists[strategy_name]['source'].append(estimated_locs)
                    Source_lists[strategy_name]['n_sources'].append(len(estimated_locs))
                    
                    tqdm.write(f"{strategy_name} RMSE: {RMSE}, WMSE: {WEIGHTED_RMSE}, Time: {TIME}")
                    RMSE_lists[strategy_name].append(RMSE)
                    WRMSE_lists[strategy_name].append(WEIGHTED_RMSE)
                    Diff_Entropy_lists[strategy_name].append(Diff_Entropy)
                    TIME_lists[strategy_name].append(TIME)
                    if round_number == args["rounds"]:
                        helper_plot(scenario, scenario_idx, Z_true, Z_pred, std, strategy, RMSE_lists[strategy_name], WRMSE_lists[strategy_name],
                                    Source_lists[strategy_name], args["rounds"], run_number, save=args["save"], show=args["show"])
                    pbar.update(1)
            RMSE_per_scenario[f"Scenario_{scenario_idx}"] = RMSE_lists
            WRMSE_per_scenario[f"Scenario_{scenario_idx}"] = WRMSE_lists
            Diff_Entropy_per_scenario[f"Scenario_{scenario_idx}"] = Diff_Entropy_lists
            Source_per_scenario[f"Scenario_{scenario_idx}"] = Source_lists
            TIME_per_scenario[f"Scenario_{scenario_idx}"] = TIME_lists

    # Save run information after processing all scenarios
    save_run_info(run_number, RMSE_per_scenario, WRMSE_per_scenario, Diff_Entropy_per_scenario, Source_per_scenario, TIME_per_scenario, args, scenarios)

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

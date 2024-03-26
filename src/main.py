import numpy as np
import argparse
from tqdm import tqdm

from boustrophedon import Boustrophedon
from radiation import RadiationField
from informative import InformativePathPlanning
from RRT import StrategicRRTPathPlanning, NaiveRRTPathPlanning, BiasRRTPathPlanning, BiasBetaRRTPathPlanning, AdaptiveRRTPathPlanning, InformativeRRTPathPlanning
from path_planning_utils import helper_plot

# Argument parsing
parser = argparse.ArgumentParser(description="Run path planning scenarios.")
parser.add_argument('-r', '--rounds', type=int, default=1, help="Number of rounds to run (default: 1).")
parser.add_argument('-beta', '--beta_t', type=float, default=1.0, help="Beta parameter for exploration-exploitation trade-off.")
parser.add_argument('-save', '--save', action='store_true', help="Save the results if this flag is set.")
parser.add_argument('-show', '--show', action='store_true', help="Show the results if this flag is set.")
parser.add_argument('-seed', '--seed', type=int, default=95789, help="Seed for random number generation.")
parser.add_argument('-budget', '--budget', type=int, default=375, help="Budget for path planning.")
parser.add_argument('-d', '--d_waypoint_distance', type=float, default=2.5, help="Waypoint distance.")
parser.add_argument('-budget-iter', '--budget_iter', type=int, default=10, help="Budget for path planning.")
args = parser.parse_args()

# Scenarios initialization
scenarios = [
    RadiationField(num_sources=0, workspace_size=(40, 40), seed=args.seed),
    RadiationField(num_sources=1, workspace_size=(40, 40), seed=args.seed),
    RadiationField(num_sources=2, workspace_size=(40, 40), seed=args.seed),
    RadiationField(num_sources=5, workspace_size=(40, 40), seed=args.seed),
    RadiationField(num_sources=10, workspace_size=(100, 100), seed=args.seed),
    ]

# set first source to be at (20, 20) for scenario 1
scenarios[1].update_source(0, 20, 20, 100000)

# Strategies setup
strategy_constructors = {
    "Boustrophedon": lambda scenario: Boustrophedon(scenario, d_waypoint_distance=2.5, budget=375),
    "Informative": lambda scenario: InformativePathPlanning(scenario, beta_t=args.beta_t, budget=375, d_waypoint_distance=2.5),
    "NaiveRRT": lambda scenario: NaiveRRTPathPlanning(scenario, beta_t=args.beta_t, budget=args.budget, d_waypoint_distance=args.d_waypoint_distance, budget_iter=args.budget_iter),
    "StrategicRRT": lambda scenario: StrategicRRTPathPlanning(scenario, beta_t=args.beta_t, budget=args.budget, d_waypoint_distance=args.d_waypoint_distance, budget_iter=args.budget_iter),
    "BiasRRT": lambda scenario: BiasRRTPathPlanning(scenario, beta_t=args.beta_t, budget=args.budget, d_waypoint_distance=args.d_waypoint_distance, budget_iter=args.budget_iter),
    "BiasBetaRRT": lambda scenario: BiasBetaRRTPathPlanning(scenario, beta_t=args.beta_t, budget=args.budget, d_waypoint_distance=args.d_waypoint_distance, budget_iter=args.budget_iter),
    "AdaRRT": lambda scenario: AdaptiveRRTPathPlanning(scenario, beta_t=args.beta_t, budget=args.budget, d_waypoint_distance=args.d_waypoint_distance, budget_iter=args.budget_iter),
    "InfoRRT": lambda scenario: InformativeRRTPathPlanning(scenario, beta_t=args.beta_t, budget=args.budget, d_waypoint_distance=args.d_waypoint_distance, budget_iter=args.budget_iter),

}

# RMSE lists setup
RMSE_lists = {strategy_name: [] for strategy_name in strategy_constructors}

# Run simulations
with tqdm(total=args.rounds * len(scenarios) * len(strategy_constructors), desc="Overall Progress") as pbar:
    for scenario_idx, scenario in enumerate(scenarios, start=1):
        RMSE_lists = {strategy_name: [] for strategy_name in strategy_constructors}
        for round_number in range(1, args.rounds + 1):
            for strategy_name, constructor in strategy_constructors.items():
                strategy = constructor(scenario)
                tqdm.write(f"Round {round_number}/{args.rounds}, Scenario {scenario_idx}, Strategy: {strategy_name}")
                Z_pred, std = strategy.run()
                Z_true = scenario.ground_truth()
                RMSE = np.sqrt(np.mean((np.log10(Z_true + 1) - np.log10(Z_pred + 1))**2))
                tqdm.write(f"{strategy_name} RMSE: {RMSE}")
                RMSE_lists[strategy_name].append(RMSE)
                # Call helper_plot on the last round if save flag is set
                if round_number == args.rounds:
                    helper_plot(scenario, scenario_idx, Z_true, Z_pred, std, strategy, RMSE_lists[strategy_name], args.rounds, save=args.save, show=args.show)
                
                pbar.update(1)

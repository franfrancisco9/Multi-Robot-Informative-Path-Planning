# src/utils/path_planning_utils.py

"""
Utilities for path planning in radiation field scenarios.
"""

import numpy as np
import os
from typing import Tuple, List, Dict
from src.point_source.point_source import PointSourceField

def run_number_from_folder(folder_path: str = "../images") -> int:
    """
    Checks in the specified folder for the latest run number and returns the next run number.

    Parameters:
    - folder_path: Path to the folder where images are stored. Default is "../images".

    Returns:
    - The next run number as an integer.
    """
    if not os.path.exists(folder_path):
        return 1
    existing_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    run_numbers = [int(f) for f in existing_folders if f.isdigit()]
    next_run_number = max(run_numbers) + 1 if run_numbers else 1
    return next_run_number

def save_run_info(run_number: int, rmse_per_scenario: Dict, wrmse_per_scenario: Dict, entropy_per_scenario: Dict, source_per_scenario: Dict, time_per_scenario: Dict, args: Dict, scenario_classes: List, folder_path: str = "../runs_review") -> None:
    """
    Saves run information to a file.

    Parameters:
    - run_number: The run number for this set of simulations.
    - rmse_per_scenario: Dictionary of RMSE values per scenario.
    - wrmse_per_scenario: Dictionary of weighted RMSE values per scenario.
    - entropy_per_scenario: Dictionary of differential entropy values per scenario.
    - source_per_scenario: Dictionary of source information per scenario.
    - time_per_scenario: Dictionary of time taken per scenario.
    - args: Arguments used for the simulations.
    - scenario_classes: List of scenario classes.
    - folder_path: Path to the folder where run information is saved. Default is "../runs_review".
    """
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.join(folder_path, f"run_{run_number}.txt")

    def calculate_rmse(predicted_sources: np.ndarray, actual_sources: np.ndarray) -> List:
        """Calculate RMSE for x, y, and intensity between predicted and actual sources."""
        if not predicted_sources.size or not actual_sources.size:
            return [np.nan] * len(actual_sources)

        rmse_results = []
        for actual in actual_sources:
            distances = np.linalg.norm(predicted_sources[:, :2] - actual[:2], axis=1)
            closest_idx = np.argmin(distances)
            closest_predicted = predicted_sources[closest_idx]

            if len(predicted_sources) < len(actual_sources):
                rmse_x = np.sqrt(np.mean((actual[0] - closest_predicted[0])**2)) if closest_idx < len(predicted_sources) else np.nan
                rmse_y = np.sqrt(np.mean((actual[1] - closest_predicted[1])**2)) if closest_idx < len(predicted_sources) else np.nan
                rmse_intensity = np.sqrt(np.mean((actual[2] - closest_predicted[2])**2)) if closest_idx < len(predicted_sources) else np.nan
            else:
                rmse_x = np.sqrt(np.mean((actual[0] - closest_predicted[0])**2))
                rmse_y = np.sqrt(np.mean((actual[1] - closest_predicted[1])**2))
                rmse_intensity = np.sqrt(np.mean((actual[2] - closest_predicted[2])**2))

            rmse_results.append((rmse_x, rmse_y, rmse_intensity))

        return rmse_results

    with open(filename, 'w') as f:
        f.write("Run Summary\n")
        f.write("=" * 40 + "\n\nArguments:\n")
        for key, value in args.items():
            f.write(f"\t{key}: {value}\n")

        for scenario_idx, scenario_class in enumerate(scenario_classes, start=1):
            scenario_key = f"Scenario_{scenario_idx}"
            actual_sources = np.array([[s[0], s[1], s[2]] for s in scenario_class.sources])

            f.write(f"\n{scenario_key} Metrics:\n")
            f.write("\tRMSE:\n")
            if scenario_key in rmse_per_scenario:
                for strategy, rmses in rmse_per_scenario[scenario_key].items():
                    avg_rmse = np.mean(rmses)
                    f.write(f"\t\t{strategy}: Avg RMSE = {avg_rmse:.4f}, Rounds = {len(rmses)}\n")
            
            f.write("\tWeighted RMSE:\n")
            if scenario_key in wrmse_per_scenario:
                for strategy, wrmses in wrmse_per_scenario[scenario_key].items():
                    avg_wrmse = np.mean(wrmses)
                    f.write(f"\t\t{strategy}: Avg WRMSE = {avg_wrmse:.4f}, Rounds = {len(wrmses)}\n")
                    
            f.write("\tDifferential Entropy:\n")
            if scenario_key in entropy_per_scenario:
                for strategy, entropies in entropy_per_scenario[scenario_key].items():
                    avg_entropy = np.mean(entropies)
                    f.write(f"\t\t{strategy}: Avg Entropy = {avg_entropy:.4f}, Rounds = {len(entropies)}\n")

            f.write("\tTime Taken:\n")
            if scenario_key in time_per_scenario:
                for strategy, times in time_per_scenario[scenario_key].items():
                    times = [t for t in times if t is not None]
                    avg_time = np.mean(times)
                    f.write(f"\t\t{strategy}: Avg Time = {avg_time:.4f}, Rounds = {len(times)}\n")
            
            f.write("\tSource Information:\n")
            if scenario_key in source_per_scenario:
                for strategy, info in source_per_scenario[scenario_key].items():
                    f.write(f"\tStrategy: {strategy}\n")
                    for round_index, predicted_sources in enumerate(info['source'], start=1):
                        predicted_sources = np.array(predicted_sources).reshape(-1, 3)
                        rmse = calculate_rmse(predicted_sources, actual_sources) if predicted_sources.size else "N/A"
                        f.write(f"\t\tRound {round_index} - Predicted sources (x, y, intensity):\n")
                        f.write(f"\t\t\t{predicted_sources.tolist()}\n")
                        f.write(f"\t\t\tRMSE (Location & Intensity): {rmse}\n")
                        f.write(f"\t\tNumber of predicted sources: {info['n_sources'][round_index - 1]}\n")
                        correct_number = "Yes" if info['n_sources'][-1] == len(actual_sources) else "No"
                        f.write(f"\t\tCorrect number of sources predicted in last round: {correct_number}\n")
            f.write(f"\tActual sources (x, y, intensity):\n")
            f.write(f"\t\t{actual_sources.tolist()}\n")

    print(f"Run information saved to {filename}")

def calculate_source_errors(actual_sources: np.ndarray, estimated_locs: np.ndarray) -> Dict:
    """
    Calculate the errors between actual sources and estimated locations.

    Parameters:
    - actual_sources: Actual source locations and intensities.
    - estimated_locs: Estimated source locations and intensities.

    Returns:
    - Dictionary of errors for x, y, and intensity.
    """
    errors = {'x_error': [], 'y_error': [], 'intensity_error': []}
    for actual, estimated in zip(actual_sources, estimated_locs):
        errors['x_error'].append(abs(actual[0] - estimated[0]))
        errors['y_error'].append(abs(actual[1] - estimated[1]))
        errors['intensity_error'].append(abs(actual[2] - estimated[2]))
    return errors

# src/visualization/plot_helper.py
"""
Helper functions for plotting in path planning.
- Created by: Francisco Fonseca on July 2024
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
from matplotlib import ticker, colors
from typing import List, Dict, Tuple
from src.point_source.point_source import PointSourceField
import warnings

# Ignore for nanmean.
warnings.simplefilter("ignore", category=RuntimeWarning)

def plot_tree_node(node, ax, color='blue') -> None:
    """
    Recursively plots each node in the tree.

    Parameters:
    - node: The TreeNode to plot.
    - ax: The matplotlib axis to plot on.
    - color: The color of the lines. Default is 'blue'.
    """
    if node.parent:
        ax.plot([node.point[0], node.parent.point[0]], [node.point[1], node.parent.point[1]], color=color)
    for child in node.children:
        plot_tree_node(child, ax, color=color)

def plot_path(path: Tuple[np.ndarray, np.ndarray], ax, color='red', linewidth=2) -> None:
    """
    Plots a path as a series of line segments.

    Parameters:
    - path: The path to plot as a tuple of (x, y) coordinates.
    - ax: The matplotlib axis to plot on.
    - color: The color of the path. Default is 'red'.
    - linewidth: The width of the path line. Default is 2.
    """
    x, y = path
    ax.plot(x, y, color=color, linewidth=linewidth)

def helper_plot(scenario, scenario_number: int, z_true: np.ndarray, z_pred: np.ndarray, std: np.ndarray,
                path, rmse_list: List[float], wrmse_list: List[float], source_list: Dict, path_list: Dict,
                rounds: int, run_number: int, save: bool = False, show: bool = False) -> None:
    """
    Generates and optionally saves or shows various plots related to a scenario.

    Parameters:
    - scenario: The scenario object containing X, Y, sources, and workspace_size.
    - scenario_number: The number identifying the scenario.
    - z_true: The ground truth data.
    - z_pred: The predicted data.
    - std: The standard deviation (uncertainty) data.
    - path: The path object containing name, beta_t, full_path, obs_wp, and trees.
    - rmse_list: A list of RMSE values.
    - rounds: The number of simulation rounds.
    - save: If True, saves the generated plots. Default is False.
    - show: If True, displays the generated plots. Default is False.
    """
    strategy_title = f'{path.name} Strategy - Scenario {scenario_number}'
    # print current folder
    print(os.getcwd())
    if not os.path.exists('./images'):
        os.makedirs('./images')
    folder = f'./images/{run_number}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_fig_title = f'{folder}/run_{rounds}_scenario_{scenario_number}_path_{path.name}.png'
    if hasattr(path, 'beta_t') and "SourceMetric" not in path.name:
        strategy_title += f' - Beta_t: {path.beta_t}'
        save_fig_title = save_fig_title.replace('.png', f'_beta_{path.beta_t}.png')
    if hasattr(path, 'num_agents'):
        strategy_title += f' - Agents: {path.num_agents}'
        save_fig_title = save_fig_title.replace('.png', f'_agents_{path.num_agents}.png')
    if hasattr(path, 'stage_lambda'):
        strategy_title += f' - Stage Lambda: {path.stage_lambda}'
        save_fig_title = save_fig_title.replace('.png', f'_stage_lambda_{path.stage_lambda}.png')
    
    max_log_value = np.ceil(np.log10(z_true.max())) if z_true.max() != 0 else 1
    levels = np.logspace(0, max_log_value, int(max_log_value) + 1)
    cmap = plt.get_cmap('Greens_r', len(levels) - 1)
    fig, axs = plt.subplots(2, 2, figsize=(20, 8), constrained_layout=True)
    fig.suptitle(strategy_title, fontsize=16)

    cs_true = axs[0, 0].contourf(scenario.X, scenario.Y, z_true, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    colorbar = fig.colorbar(cs_true, ax=axs[0, 0], format=ticker.LogFormatterMathtext())
    axs[0, 0].set_title('Ground Truth')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].set_facecolor(cmap(0))

    for obstacle in scenario.obstacles:
        if obstacle['type'] == 'rectangle':
            rect = plt.Rectangle((obstacle['x'], obstacle['y']), obstacle['width'], obstacle['height'], color='black')
            axs[0, 0].add_patch(rect)
    # Similar to how we save the top right corner plot, We want to save into a scenario folders the top left corner plot with the
    # scenario number 
    if not os.path.exists(f'{folder}/top_left_corner'):
        os.makedirs(f'{folder}/top_left_corner')
    # only save the axs[0,0] with different title (scenario number) and dont save if it already exists
    # scenario_fig, axs_scenario = plt.subplots(1, 1, figsize=(20, 8), constrained_layout=True)
    # scenario_fig.suptitle("Scenario " + str(scenario_number), fontsize=16)
    # cs_true = axs_scenario.contourf(scenario.X, scenario.Y, z_true, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    # scenario_fig.colorbar(cs_true, ax=axs_scenario, format=ticker.LogFormatterMathtext())
    # axs_scenario.set_xlabel('x (m)')
    # axs_scenario.set_ylabel('y (m)')
    # axs_scenario.set_facecolor(cmap(0))
    # # make sure text size is big enough to be readable
    # for item in ([axs_scenario.title, axs_scenario.xaxis.label, axs_scenario.yaxis.label] +
    #             axs_scenario.get_xticklabels() + axs_scenario.get_yticklabels() + colorbar.ax.get_yticklabels()):
    #     item.set_fontsize(16)
    # for obstacle in scenario.obstacles:
    #     if obstacle['type'] == 'rectangle':
    #         rect = plt.Rectangle((obstacle['x'], obstacle['y']), obstacle['width'], obstacle['height'], color='black')
    #         axs_scenario.add_patch(rect)
    # # mark with red cross the sources only label once
    # for source in scenario.sources:
    #     axs_scenario.plot(source[0], source[1], 'rX', markersize=10, label='Sources') if scenario.sources.index(source) == 0 else axs_scenario.plot(source[0], source[1], 'rX', markersize=10)

    # scenario_fig.savefig(f'{folder}/top_left_corner/{os.path.basename(save_fig_title).replace(".png", "_scenario_" + str(scenario_number) + ".png")}')
    # plt.close(scenario_fig)

    cs_pred = axs[0, 1].contourf(scenario.X, scenario.Y, z_pred, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    #fig.colorbar(cs_pred, ax=axs[0, 1], format=ticker.LogFormatterMathtext())
    axs[0, 1].set_title('Predicted Field')
    x_new, y_new = path_list['full_path'][-1]
    if hasattr(path, 'num_agents'):
        colors_path = ['b', 'c', 'm', 'y', 'k', 'w']
        for i in range(path.num_agents):
            current_x, current_y = np.array(path_list['agents_full_path'][-1][i]).reshape(-1, 2).T
            axs[0, 1].plot(current_x, current_y, label=f'Agent {i+1} Path', linewidth=1, color=colors_path[i])
    else:
        axs[0, 1].plot(x_new, y_new, 'b-', label=path.name + ' Path', linewidth=1)
    
    axs[0, 1].plot(path.obs_wp[:, 0], path.obs_wp[:, 1], 'ro', markersize=1)
    for source in scenario.sources:
        axs[0, 1].plot(source[0], source[1], 'rX', markersize=10, label='Source')
    # source_estimated = source_list['source'][-1]
    # for source in source_estimated:
    #     axs[0, 1].plot(source[0], source[1], 'yX', markersize=10, label='Estimated Source')

    axs[0, 1].set_facecolor(cmap(0))
    for obstacle in scenario.obstacles:
        if obstacle['type'] == 'rectangle':
            rect = plt.Rectangle((obstacle['x'], obstacle['y']), obstacle['width'], obstacle['height'], color='black')
            axs[0, 1].add_patch(rect)
    for i in range(1, rounds + 1):
        top_corner_fig, axs_top = plt.subplots(1, 1, figsize=(20, 8), constrained_layout=True)
        top_corner_fig.suptitle(strategy_title, fontsize=16)
        x_new, y_new = path_list['full_path'][i-1]
        axs_top.contourf(scenario.X, scenario.Y, path_list['z_pred'][i-1], levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
        if hasattr(path, 'num_agents'):
            for j in range(path.num_agents):
                current_x, current_y = np.array(path_list['agents_full_path'][i-1][j]).reshape(-1, 2).T
                axs_top.plot(current_x, current_y, label=f'Agent {i+1} Path', linewidth=1, color=colors_path[j])
        else:
            axs_top.plot(x_new, y_new, 'b-', label=path.name + ' Path', linewidth=1)
        axs_top.plot(path_list['obs_wp'][i-1][:, 0], path_list['obs_wp'][i-1][:, 1], 'ro', markersize=1)
        for source in scenario.sources:
            axs_top.plot(source[0], source[1], 'rX', markersize=10, label='Source')
        # for source in source_list['source'][i-1]:
        #     axs_top.plot(source[0], source[1], 'yX', markersize=10, label='Estimated Source')
        axs_top.set_facecolor(cmap(0))

        axs_top.set_title('Predicted Field')
        axs_top.set_xlabel('x (m)')
        axs_top.set_ylabel('y (m)')
        if not os.path.exists(f'{folder}/top_corner'):
            os.makedirs(f'{folder}/top_corner')
        # colorber
        fig.colorbar(cs_pred, ax=axs_top, format=ticker.LogFormatterMathtext())
        top_corner_fig.savefig(f'{folder}/top_corner/{os.path.basename(save_fig_title).replace(".png", "_predicted_field_run_" + str(i) + ".png")}')
        plt.close(top_corner_fig)

    std_reshaped = std.reshape(scenario.X.shape)
    cs_uncertainty = axs[1, 0].contourf(scenario.X, scenario.Y, std_reshaped, cmap='Reds')
    axs[1, 0].set_title('Uncertainty Field')
    axs[1, 0].set_facecolor('pink')
    # add colorbar for uncertainty
    fig.colorbar(cs_uncertainty, ax=axs[1, 0])
    
    axs[1, 1].boxplot([rmse_list, wrmse_list], labels=['RMSE', 'WRMSE'], whis=50)
    axs[1, 1].set_title('RMSE and WRMSE Evolution')
    axs[1, 1].set_xlabel('Metric')
    axs[1, 1].set_ylabel('Value')
    # make y start in 0 always
    axs[1, 1].set_ylim(0.0, max(max(rmse_list), max(wrmse_list)) + 0.1)
    axs[1, 1].grid(True)

    if not os.path.exists(f'{folder}/metrics'):
        os.makedirs(f'{folder}/metrics')
    metrics_fig, axs_metrics = plt.subplots(1, 1, figsize=(20, 8), constrained_layout=True)
    metrics_fig.suptitle(strategy_title, fontsize=16)
    axs_metrics.boxplot([rmse_list, wrmse_list], labels=['RMSE', 'WRMSE'], whis=50)
    axs_metrics.set_title('RMSE and WRMSE Evolution')
    axs_metrics.set_xlabel('Metric')
    axs_metrics.set_ylabel('Value')
    # make y start in 0 always
    axs_metrics.set_ylim(0.0, max(max(rmse_list), max(wrmse_list)) + 0.1)
    axs_metrics.grid(True)
    metrics_fig.savefig(f'{folder}/metrics/{os.path.basename(save_fig_title).replace(".png", "_metrics.png")}')
    plt.close(metrics_fig)

    if save:
        plt.savefig(save_fig_title)
    if show:
        plt.show()
    distance_histogram(scenario, path.obs_wp, save_fig_title.replace('.png', '_histogram.png'), show=show)

    source_errors_plot_title = save_fig_title.replace('.png', '_source_errors.png')
    if hasattr(path, 'best_estimates'):
        source_distance(scenario, source_list, save_fig_title=source_errors_plot_title, show=show)
    plt.close()

    if "RRT" in path.name:
        rrt_helper_plot(save_fig_title, strategy_title, scenario, path, colors_path, save=save, show=show)

def rrt_helper_plot(save_fig_title: str, strategy_title: str, scenario: PointSourceField, path, colors_path: List[str], save: bool = False, show: bool = False) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle(f'Additional Insights for {strategy_title}', fontsize=16)

    colors_tree = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    for tree_root in path.trees.trees:
        plot_tree_node(tree_root, axs[0], color=colors_tree[path.trees.trees.index(tree_root) % len(colors_tree)])
    if hasattr(path, 'num_agents'):
        for i in range(path.num_agents):
            current_x, current_y = np.array(path.agents_full_path[i]).reshape(-1, 2).T
            axs[0].plot(current_x, current_y, label=f'Agent {i+1} Path', linewidth=1, color=colors_path[i])
    else:
        plot_path(path.full_path, axs[0])        

    axs[0].set_title('Final Path with All Trees')
    axs[0].set_xlim(0, scenario.workspace_size[0])
    axs[0].set_ylim(0, scenario.workspace_size[1])

    axs[1].plot(path.uncertainty_reduction, marker='o')
    axs[1].set_title('Reduction in Uncertainty Over Iterations')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Average Uncertainty (std)')
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig(save_fig_title.replace('.png', '_additional.png'))
    if show:
        plt.show()
    plt.close()

def distance_histogram(scenario: PointSourceField, obs_wp: np.ndarray, save_fig_title: str = None, show: bool = False) -> None:
    """
    Plots and shows the Histogram of the distances between each source and the observations in O.

    Parameters:
    - scenario: The scenario object containing sources and workspace_size.
    - obs_wp: The observed waypoints.
    - save_fig_title: The file path to save the histogram image.
    - show: If True, displays the histogram plot. Default is False.
    """
    distances = []
    for source in scenario.sources:
        for obs in obs_wp:
            distances.append(np.linalg.norm(source[:2] - obs))
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of Distances between Sources and Observations')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    if save_fig_title:
        plt.savefig(save_fig_title)
    if show:
        plt.show()

def create_gif(save_dir: str, output_filename: str = "sources_estimation_test_0.gif") -> None:
    images = []
    test_str = "test_" + output_filename.split("_")[-1].split(".")[0]
    image_names = [f for f in os.listdir(save_dir) if f.endswith(f"{test_str}.png")]
    image_names.sort(key=lambda x: int(x.split("_")[1]))
    for image_name in image_names:
        images.append(imageio.imread(os.path.join(save_dir, image_name)))
    imageio.mimsave(os.path.join(save_dir, output_filename), images, fps=1)

def save_plot_iteration(i: int, scenario: PointSourceField, estimated_locs: np.ndarray, obs_wp_send: np.ndarray, save_dir: str) -> None:
    print(f"Plotting iteration {i}")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Estimated Sources Iteration: {i}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    for source in scenario.sources:
        ax.plot(source[0], source[1], 'ro', markersize=10, label='Actual Source')
    
    for est_source in estimated_locs:
        ax.plot(est_source[0], est_source[1], 'bx', markersize=10, label='Estimated Source')
    
    for obs in obs_wp_send:
        ax.plot(*obs, 'go', markersize=5)
    
    ax.set_xlim([0, 40])
    ax.set_ylim([0, 40])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.savefig(os.path.join(save_dir, f"iteration_{i}_test_" + str(len(scenario.sources)) + ".png"))
    plt.close(fig)

def log_estimation_details(iteration: int, estimated_locs: np.ndarray, estimated_num_sources: int, actual_sources: np.ndarray) -> None:
    print(f"Iteration {iteration}:")
    print("Estimated Locations:", estimated_locs)
    print("Estimated Number of Sources:", estimated_num_sources)
    print("Actual Sources:", actual_sources)
    
    for i, source in enumerate(actual_sources):
        if i < estimated_num_sources:
            x_error = abs(source[0] - estimated_locs[i][0])
            y_error = abs(source[1] - estimated_locs[i][1])
            intensity_error = abs(source[2] - estimated_locs[i][2])
            norm_error = np.linalg.norm([x_error, y_error])
            
            print(f"Source {i+1} Errors:")
            print(f"    X Error: {x_error} (Error %: {x_error / source[0] * 100}%)")
            print(f"    Y Error: {y_error} (Error %: {y_error / source[1] * 100}%)")
            print(f"    Norm error: {norm_error}")
            print(f"    Intensity Error: {intensity_error} (Error %: {intensity_error / source[2] * 100}%)")
        else:
            print(f"Source {i+1} Error: Not estimated")
    
    num_sources_error = abs(len(actual_sources) - estimated_num_sources)
    print(f"Number of Sources Error: {num_sources_error}")

def source_distance(scenario: PointSourceField, source_list: Dict, save_fig_title: str = None, show: bool = False) -> None:
    """
    Plots errors in estimating the position and intensity of sources as scatter plots with error bars.
    """
    x_errors = []
    y_errors = []
    intensity_errors = []

    for i, actual_source in enumerate(scenario.sources):
        these_x_errors = []
        these_y_errors = []
        these_intensity_errors = []
        for sources in source_list['source']:
            if len(sources) == 0:
                these_x_errors.append(np.nan)
                these_y_errors.append(np.nan)
                these_intensity_errors.append(np.nan)
            elif i < len(sources):
                these_x_errors.append(abs(actual_source[0] - sources[i][0]))
                these_y_errors.append(abs(actual_source[1] - sources[i][1]))
                these_intensity_errors.append(abs(actual_source[2] - sources[i][2]))
        
        x_errors.append(these_x_errors)
        y_errors.append(these_y_errors)
        intensity_errors.append(these_intensity_errors)
    fig, ax = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    source_ids = [f"Source {i+1}" for i in range(len(scenario.sources))]

    for idx, (errors, label) in enumerate(zip([x_errors, y_errors, intensity_errors], ['X Error', 'Y Error', 'Intensity Error'])):
        averages = [np.nanmean(e) for e in errors]
        std_devs = [np.nanstd(e) for e in errors]
        ax[idx].bar(source_ids, averages, yerr=std_devs, alpha=0.6, color='b', label=label)
        ax[idx].set_ylabel('Error')
        ax[idx].set_title(label)
        ax[idx].legend()

        for i, e in enumerate(errors):
            if all(np.isnan(e)):
                ax[idx].axvline(i, color='r', linestyle='--', label='Missing estimate')

    plt.xlabel('Sources')
    plt.suptitle('Errors in Source Estimation')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    if save_fig_title:
        plt.savefig(save_fig_title)
    if show:
        plt.show()
    plt.close()

def calculate_differential_entropy(std_devs: np.ndarray) -> float:
    """
    Calculate the differential entropy given a list of standard deviations.

    Parameters:
    - std_devs: An array of standard deviation values for each prediction on the grid.

    Returns:
    - Differential entropy in bits.
    """
    pi_e = np.pi * np.e
    if np.any(std_devs == 0):
        std_devs[std_devs == 0] = 1e-6
    entropy_sum = np.sum(np.log(std_devs * np.sqrt(2 * pi_e)))
    differential_entropy = entropy_sum / (np.log(2))
    return differential_entropy

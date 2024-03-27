import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors

def helper_plot(scenario, scenario_number, Z_true, Z_pred, std, path, RMSE_list, ROUNDS, save=False, show=False):
    # Titles for clarity
    strategy_title = f'{path.name} Strategy - Scenario {scenario_number}'
    # save fig title
    save_fig_title = f'../images/18/scenario_{scenario_number}_run_{ROUNDS}_path_{path.name}.png'
    # if there is a beta_t then add it to the title
    if hasattr(path, 'beta_t'):
        strategy_title += f' - Beta_t: {path.beta_t}'
        save_fig_title = f'../images/18/scenario_{scenario_number}_run_{ROUNDS}_path_{path.name}_beta_{path.beta_t}.png'
    ground_truth_title = 'Ground Truth'
    predicted_field_title = 'Predicted Field'
    uncertainty_field_title = 'Uncertainty Field'
    rmse_title = f'{ROUNDS} Run Average RMSE'

    # Determine log scale levels,
    if Z_true.max() == 0:
        max_log_value = 1
    else:
        max_log_value = np.ceil(np.log10(Z_true.max()))
    levels = np.logspace(0, max_log_value, int(max_log_value) + 1)
    cmap = plt.get_cmap('Greens_r', len(levels) - 1)

    # Plot ground truth and predicted field
    fig, axs = plt.subplots(2, 2, figsize=(20, 8), constrained_layout=True)
    fig.suptitle(strategy_title, fontsize=16)
    # Ground truth
    cs_true = axs[0][0].contourf(scenario.X, scenario.Y, Z_true, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    fig.colorbar(cs_true, ax=axs[0][0], format=ticker.LogFormatterMathtext())
    axs[0][0].set_title(ground_truth_title)
    axs[0][0].set_xlabel('x')
    axs[0][0].set_ylabel('y')
    # make the background the colour of the lowest contour level
    axs[0][0].set_facecolor(cmap(0))

    # Predicted field
    cs_pred = axs[0][1].contourf(scenario.X, scenario.Y, Z_pred, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    fig.colorbar(cs_pred, ax=axs[0][1], format=ticker.LogFormatterMathtext())
    axs[0][1].set_title(predicted_field_title)
    x_new, y_new = path.full_path
    axs[0][1].plot(x_new, y_new, 'b-', label=path.name + ' Path')
    # add the waypoints with red circles
    # print(path.obs_wp)
    axs[0][1].plot(path.obs_wp[:, 0], path.obs_wp[:, 1], 'ro', markersize=5)  # Waypoints
    # make the background the colour of the lowest contour level
    axs[0][1].set_facecolor(cmap(0))
    sources = scenario.sources
    for source in sources:
        axs[0][1].plot(source[0], source[1], 'rX', markersize=10, label='Source')

    # Uncertainty field
    std = std.reshape(scenario.X.shape[0], scenario.X.shape[1])
    cs_uncertainty = axs[1][0].contourf(scenario.X, scenario.Y, std, cmap='Reds')
    # fig.colorbar(cs_uncertainty, ax=axs[1][0])
    axs[1][0].set_title(uncertainty_field_title)
    # make the background the colour of the lowest contour level
    axs[1][0].set_facecolor('pink')

    # Plot RMSE as vertical y with median as a dot and std as error bars
    # x label is just the scenario number and title is 10 run average RMSE
    # Only one x label for each scenario
    axs[1][1].errorbar(scenario_number, np.mean(RMSE_list), yerr=np.std(RMSE_list), fmt='o', linewidth=2, capsize=6)
    axs[1][1].set_title(rmse_title)
    axs[1][1].set_xlabel('Scenario' + str(scenario_number))
    # only show one tick for the x axis
    axs[1][1].set_xticks([scenario_number])
    axs[1][1].set_ylabel('RMSE')

    if show:
        plt.show()
    if save:
        plt.savefig(save_fig_title)

    plt.close()
    # Check if path's name contains "RRT"
    if "RRT" in path.name:
        # Combine additional RRT plots into the existing plotting process
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))
        fig.suptitle(f'Additional Insights for {strategy_title}', fontsize=16)
        # Plot all trees and the final chosen path
        for tree_root in path.trees:
            plot_tree_node(tree_root, axs[0], color='lightgray')  # Plot all trees in light gray
        plot_path(path.full_path, axs[0], color='red', linewidth=3)  # Highlight the final path
        axs[0].set_title('Final Path with All Trees')
        axs[0].set_xlim(0, scenario.workspace_size[0])
        axs[0].set_ylim(0, scenario.workspace_size[1])

        # Plot the reduction in uncertainty over iterations
        axs[1].plot(path.uncertainty_reduction, marker='o')
        axs[1].set_title('Reduction in Uncertainty Over Iterations')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Average Uncertainty (std)')
        axs[1].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if show:
            plt.show()
        if save:
            plt.savefig(save_fig_title.replace('.png', '_additional.png'))
        plt.close()

def plot_tree_node(node, ax, color='blue'):
    """Recursively plot each node in the tree."""
    if node.parent:
        ax.plot([node.point[0], node.parent.point[0]], [node.point[1], node.parent.point[1]], color=color)
    for child in node.children:
        plot_tree_node(child, ax, color=color)

def plot_path(path, ax, color='red', linewidth=2):
    """Plot a path as a series of line segments."""
    x, y = path
    ax.plot(x, y, color=color, linewidth=linewidth)

class TreeNode:
    def __init__(self, point, parent=None):
        self.point = point
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

class TreeCollection(TreeNode):
    def __init__(self):
        self.trees = []

    def add(self, tree):
        self.trees.append(tree)

    def __iter__(self):
        return iter(self.trees)

    def __getitem__(self, idx):
        return self.trees[idx]

    def __len__(self):
        return len(self.trees)
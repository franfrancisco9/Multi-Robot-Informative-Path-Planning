import matplotlib.pyplot as plt
from matplotlib import ticker, colors
import numpy as np
import os 

class TreeNode:
    """Represents a node in a tree structure."""
    def __init__(self, point, parent=None, cost=0):
        """
        Initializes a new instance of the TreeNode class.
        
        Parameters:
        - point: The (x, y) coordinates of the node.
        - parent: The parent TreeNode. Default is None.
        """
        self.point = point
        self.parent = parent
        self.children = []
        self.cost = cost

    def add_child(self, child):
        """Adds a child node to this node."""
        self.children.append(child)
        child.parent = self

class TreeCollection:
    """Represents a collection of trees."""
    def __init__(self):
        """Initializes a new instance of the TreeCollection class."""
        self.trees = []

    def add(self, tree):
        """Adds a tree to the collection."""
        self.trees.append(tree)

    def __iter__(self):
        return iter(self.trees)

    def __getitem__(self, idx):
        return self.trees[idx]

    def __len__(self):
        return len(self.trees)

def plot_tree_node(node, ax, color='blue'):
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

def plot_path(path, ax, color='red', linewidth=2):
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

def run_number_from_folder():
    """
    Checks in ../images/ for the latest run number and returns the next run number.
    """
    folder = '../images'
    if not os.path.exists(folder):
        return 1
    # check the current folders names 
    existing_folders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    run_numbers = [int(f) for f in existing_folders if f.isdigit()]
    next_run_number = max(run_numbers) + 1 if run_numbers else 1
    return next_run_number

def helper_plot(scenario, scenario_number, z_true, z_pred, std, path, rmse_list, rounds, run_number, save=False, show=False):
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
    # Setup plot titles and save paths
    strategy_title = f'{path.name} Strategy - Scenario {scenario_number}'
    # if not present create images folder
    if not os.path.exists('../images'):
        os.makedirs('../images')
    folder = f'images/{run_number}'
    if not os.path.exists(f'../{folder}'):
        os.makedirs(f'../{folder}')
    save_fig_title = f'../{folder}/run_{rounds}_scenario_{scenario_number}_path_{path.name}.png'
    if hasattr(path, 'beta_t'):
        strategy_title += f' - Beta_t: {path.beta_t}'
        save_fig_title = f'../{folder}/run_{rounds}_scenario_{scenario_number}_path_{path.name}_beta_{path.beta_t}.png'
    
    # Determine the levels for log scale based on z_true
    max_log_value = np.ceil(np.log10(z_true.max())) if z_true.max() != 0 else 1
    levels = np.logspace(0, max_log_value, int(max_log_value) + 1)
    cmap = plt.get_cmap('Greens_r', len(levels) - 1)
    # Initialize figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(20, 8), constrained_layout=True)
    fig.suptitle(strategy_title, fontsize=16)

    # Plot Ground Truth
    cs_true = axs[0, 0].contourf(scenario.X, scenario.Y, z_true, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    fig.colorbar(cs_true, ax=axs[0, 0], format=ticker.LogFormatterMathtext())
    axs[0, 0].set_title('Ground Truth')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].set_facecolor(cmap(0))  # Background color of the lowest contour level

    # Plot Predicted Field
    cs_pred = axs[0, 1].contourf(scenario.X, scenario.Y, z_pred, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    fig.colorbar(cs_pred, ax=axs[0, 1], format=ticker.LogFormatterMathtext())
    axs[0, 1].set_title('Predicted Field')
    x_new, y_new = path.full_path
    axs[0, 1].plot(x_new, y_new, 'b-', label=path.name + ' Path')
    axs[0, 1].plot(path.obs_wp[:, 0], path.obs_wp[:, 1], 'ro', markersize=5)  # Waypoints
    for source in scenario.sources:
        axs[0, 1].plot(source[0], source[1], 'rX', markersize=10, label='Source')
    axs[0, 1].set_facecolor(cmap(0))

    # Plot Uncertainty Field
    std_reshaped = std.reshape(scenario.X.shape)
    cs_uncertainty = axs[1, 0].contourf(scenario.X, scenario.Y, std_reshaped, cmap='Reds')
    axs[1, 0].set_title('Uncertainty Field')
    axs[1, 0].set_facecolor('pink')

    # Plot RMSE
    axs[1, 1].errorbar(scenario_number, np.mean(rmse_list), yerr=np.std(rmse_list), fmt='o', linewidth=2, capsize=6)
    axs[1, 1].set_title(f'{rounds} Run Average RMSE')
    axs[1, 1].set_xlabel(f'Scenario {scenario_number}')
    axs[1, 1].set_xticks([scenario_number])
    axs[1, 1].set_ylabel('RMSE')

    # Show or save the plot as needed
    if save:
        plt.savefig(save_fig_title)
    if show:
        plt.show()
    plt.close()

    # Additional RRT-specific plots
    if "RRT" in path.name:
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))
        fig.suptitle(f'Additional Insights for {strategy_title}', fontsize=16)

        # Plot all trees and the final chosen path
        for tree_root in path.trees:
            plot_tree_node(tree_root, axs[0], color='lightgray')
        plot_path(path.full_path, axs[0], color='red', linewidth=3)
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
        if save:
            plt.savefig(save_fig_title.replace('.png', '_additional.png'))
        if show:
            plt.show()
        plt.close()

def calculate_differential_entropy(std_devs):
    """
    Calculate the differential entropy given a list of standard deviations.
    
    Parameters:
    - std_devs: An array of standard deviation values for each prediction on the grid.
    
    Returns:
    - Differential entropy in bits.
    """
    pi_e = np.pi * np.e  # Product of pi and Euler's number
    # Compute the sum of log(sigma * sqrt(2 * pi * e)) for each standard deviation
    entropy_sum = np.sum(np.log(std_devs * np.sqrt(2 * pi_e)))
    # Convert the sum from nats to bits and normalize by the number of predictions
    differential_entropy = entropy_sum / (np.log(2))
    return differential_entropy

def save_run_info(run_number, rmse_list, entropy_list, args, folder_path="../runs_review"):
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.join(folder_path, f"run_{run_number}.txt")

    with open(filename, 'w') as f:
        f.write("Run Summary\n")
        f.write("=" * 40 + "\n\nArguments:\n")
        for key, value in args.items():
            f.write(f"{key}: {value}\n")
        
        for scenario, scenario_rmse in rmse_list.items():
            f.write(f"\n{scenario} RMSE:\n")
            for strategy, values in scenario_rmse.items():
                f.write(f"{strategy}: Avg RMSE = {np.mean(values):.4f}\n")
        
        for scenario, scenario_entropy in entropy_list.items():
            f.write(f"\n{scenario} Differential Entropy:\n")
            for strategy, values in scenario_entropy.items():
                f.write(f"{strategy}: Avg Entropy = {np.mean(values):.4f}\n")
    
    print(f"Run information saved to {filename}")

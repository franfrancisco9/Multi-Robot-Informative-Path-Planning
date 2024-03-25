import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors


def helper_plot(scenario, scenario_number, Z_true, Z_pred, std, path, RMSE_list, ROUNDS):
    # Determine log scale levels,
    if Z_true.max() == 0:
        max_log_value = 1
    else:
        max_log_value = np.ceil(np.log10(Z_true.max()))
    levels = np.logspace(0, max_log_value, int(max_log_value) + 1)
    cmap = plt.get_cmap('Greens_r', len(levels) - 1)

    # Plot ground truth and predicted field
    fig, axs = plt.subplots(2, 2, figsize=(20, 8), constrained_layout=True)
    
    # Ground truth
    cs_true = axs[0][0].contourf(scenario.X, scenario.Y, Z_true, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    fig.colorbar(cs_true, ax=axs[0][0], format=ticker.LogFormatterMathtext())
    axs[0][0].set_title(f'Scenario {scenario_number} Ground Truth')
    axs[0][0].set_xlabel('x')
    axs[0][0].set_ylabel('y')
    # make the background the colour of the lowest contour level
    axs[0][0].set_facecolor(cmap(0))

    # Predicted field
    cs_pred = axs[0][1].contourf(scenario.X, scenario.Y, Z_pred, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    fig.colorbar(cs_pred, ax=axs[0][1], format=ticker.LogFormatterMathtext())
    axs[0][1].set_title(f'Scenario {scenario_number} Predicted Field')
    x_new, y_new = path.full_path
    axs[0][1].plot(x_new, y_new, 'b-', label=path.name + ' Path')
    # add the waypoints with red circles
    # print(path.obs_wp)
    axs[0][1].plot(path.obs_wp[:, 0], path.obs_wp[:, 1], 'ro', markersize=5)  # Waypoints
    # make the background the colour of the lowest contour level
    axs[0][1].set_facecolor(cmap(0))
    sources = scenario.get_sources_info()
    for source in sources:
        axs[0][1].plot(source[0], source[1], 'rX', markersize=10, label='Source')

    # Uncertainty field
    std = std.reshape(scenario.X.shape[0], scenario.X.shape[1])
    cs_uncertainty = axs[1][0].contourf(scenario.X, scenario.Y, std, cmap='Reds')
    # fig.colorbar(cs_uncertainty, ax=axs[1][0])
    axs[1][0].set_title(f'Scenario {scenario_number} Uncertainty Field')
    # make the background the colour of the lowest contour level
    axs[1][0].set_facecolor('pink')

    # Plot RMSE as vertical y with median as a dot and std as error bars
    # x label is just the scenario number and title is 10 run average RMSE
    # Only one x label for each scenario
    axs[1][1].errorbar(scenario_number, np.mean(RMSE_list), yerr=np.std(RMSE_list), fmt='o', linewidth=2, capsize=6)
    axs[1][1].set_title(f'{ROUNDS} Run Average RMSE')
    axs[1][1].set_xlabel('Scenario' + str(scenario_number))
    # only show one tick for the x axis
    axs[1][1].set_xticks([scenario_number])
    axs[1][1].set_ylabel('RMSE')

    # plt.savefig(f'../images/scenario_{scenario_number}_run_{ROUNDS}_path_{path.name}.png')
    plt.show()
    plt.close()


def save_iteration(data, iteration, directory="path_planning_iterations"):
    """Save the state of the path planning process at a given iteration."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"iteration_{iteration}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def load_iteration(iteration, directory="path_planning_iterations"):
    """Load a saved iteration."""
    filepath = os.path.join(directory, f"iteration_{iteration}.pkl")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data

def plot_from_iteration(data, scenario_number, save_path=None):
    """Generate and save or display a plot from saved iteration data."""
    Z_true = data['Z_true']
    Z_pred = data['Z_pred']
    std = data['std']
    obs_wp = data['obs_wp']
    full_path = data['full_path']
    RMSE_list = data['RMSE_list']

    # Plotting logic (adapted from the provided helper_plot)
    # [Insert the adapted plotting logic here]

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_gif_from_iterations(directory="path_planning_iterations", output_filename="path_planning.gif"):
    """Create a GIF from saved iterations."""
    # [Add logic to create a GIF from saved iteration plots]

def plot_tree(scenario, tree):
    """
    Plots the current state of the RRT tree including nodes and edges.
    """

    plt.figure(figsize=(10, 10))
    # Plot workspace boundary
    plt.plot([0, scenario.workspace_size[0], scenario.workspace_size[0], 0, 0],
                [0, 0, scenario.workspace_size[1], scenario.workspace_size[1], 0], 'k-')

    # Plot all points in the tree
    plt.plot(tree.data[:, 0], tree.data[:, 1], 'bo', label='Tree Nodes')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('RRT Tree Expansion')
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def evaluate_information_gain(current_std, new_prediction_std):
    """
    Evaluates the effectiveness of a new observation position in reducing the overall uncertainty.

    Parameters:
    - current_std: The current standard deviation across the environment before adding the new observation.
    - new_prediction_std: The predicted standard deviation across the environment after adding the new observation.

    Returns:
    - information_gain: A measure of the reduction in uncertainty, where higher values indicate greater reductions.
    """
    # Calculate the average standard deviation before and after the new observation
    avg_current_std = np.mean(current_std)
    avg_new_prediction_std = np.mean(new_prediction_std)

    # The information gain is the reduction in the average standard deviation
    information_gain = avg_current_std - avg_new_prediction_std

    return information_gain

# Example usage within a path planning class method:
# self.save_iteration({
#     'Z_true': Z_true,
#     'Z_pred': Z_pred,
#     'std': std,
#     'obs_wp': self.obs_wp,
#     'full_path': self.full_path,
#     'RMSE_list': RMSE_list,
# }, iteration)

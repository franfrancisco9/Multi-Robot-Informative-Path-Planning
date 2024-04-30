import matplotlib.pyplot as plt, imageio
from matplotlib import ticker, colors
import numpy as np
import os 
from scipy.stats import poisson, uniform, norm
import warnings
# Ignore for nanmean.
warnings.simplefilter("ignore", category=RuntimeWarning)

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
    ax.plot(x, y, color='blue', linewidth=linewidth)

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

def helper_plot(scenario, scenario_number, z_true, z_pred, std, path, rmse_list, source_list, rounds, run_number, save=False, show=False):
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
        save_fig_title = save_fig_title.replace('.png', f'_beta_{path.beta_t}.png')
    if hasattr(path, 'num_agents'):
        strategy_title += f' - Agents: {path.num_agents}'
        save_fig_title = save_fig_title.replace('.png', f'_agents_{path.num_agents}.png')
    
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
    if hasattr(path, 'num_agents'):
        # plot each with a different color except for green
        # blue, cyan, magenta, yellow, black, white
        colors_path = ['b', 'c', 'm', 'y', 'k', 'w']
        for i in range(path.num_agents):
            current_x, current_y = np.array(path.agents_full_path[i]).reshape(-1, 2).T
            axs[0, 1].plot(current_x, current_y, label=f'Agent {i+1} Path', linewidth=1, color=colors_path[i])
    else:
        axs[0, 1].plot(x_new, y_new, 'b-', label=path.name + ' Path', linewidth=1)
    
    axs[0, 1].plot(path.obs_wp[:, 0], path.obs_wp[:, 1], 'ro', markersize=1)  # Waypoints
    for source in scenario.sources:
        axs[0, 1].plot(source[0], source[1], 'rX', markersize=10, label='Source')
    # plot the sources estimated of the last run
    source_estimated = source_list['source'][-1]
    for source in source_estimated:
        axs[0, 1].plot(source[0], source[1], 'yX', markersize=10, label='Estimated Source')

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
    distance_histogram(scenario, path.obs_wp, save_fig_title.replace('.png', '_histogram.png'), show=show)

    source_errors_plot_title = save_fig_title.replace('.png', '_source_errors.png')
    if hasattr(path, 'best_estimates'):
        source_distance(scenario, source_list, save_fig_title=source_errors_plot_title, show=show)
    plt.close()

    # Additional RRT-specific plots
    if "RRT" in path.name:
        rrt_helper_plot(save_fig_title, strategy_title, scenario, path, colors_path, save=save, show=show)

def rrt_helper_plot(save_fig_title, strategy_title, scenario, path, colors_path, save=False, show=False):
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))
        fig.suptitle(f'Additional Insights for {strategy_title}', fontsize=16)

        # Plot all trees and the final chosen path
        for tree_root in path.trees:
            plot_tree_node(tree_root, axs[0], color='lightgray')
        # is has attribute num_agents, plot each agent path with a different color
        if hasattr(path, 'num_agents'):
            for i in range(path.num_agents):
                current_x, current_y = np.array(path.agents_full_path[i]).reshape(-1, 2).T
                axs[0].plot(current_x, current_y, label=f'Agent {i+1} Path', linewidth=1, color=colors_path[i])
        else:
            plot_path(path.full_path, axs[0])        

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

def distance_histogram(scenario, obs_wp, save_fig_title=None, show=False):
    """
    Plots and shows the Histogram of the distances between each source and the observations in O.
    
    Parameters:
    - scenario: The scenario object containing sources and workspace_size.
    - obs_wp: The observed waypoints.
    """
    # Compute the distances between each source and the observations
    distances = []
    for source in scenario.sources:
        for obs in obs_wp:
            distances.append(np.linalg.norm(source[:2] - obs))
    # new figure
    plt.figure(figsize=(10, 6))
    # Plot the histogram of distances
    plt.hist(distances, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of Distances between Sources and Observations')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    if save_fig_title:
        plt.savefig(save_fig_title)
    if show:
        plt.show()

def save_run_info(run_number, rmse_per_scenario, entropy_per_scenario, source_per_scenario, time_per_scenario, args, scenario_classes, folder_path="../runs_review"):
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.join(folder_path, f"run_{run_number}.txt")

    def calculate_rmse(predicted_sources, actual_sources):
        """Calculate RMSE for x, y, and intensity between predicted and actual sources."""
        if not predicted_sources.size or not actual_sources.size:
            return [np.nan] * len(actual_sources)  # Return "N/A" for each actual source if no predictions or no actual sources

        # Initialize an array to store RMSE values for each actual source
        rmse_results = []

        # Calculate RMSE for each actual source by finding the closest predicted source
        for actual in actual_sources:
            distances = np.linalg.norm(predicted_sources[:, :2] - actual[:2], axis=1)
            closest_idx = np.argmin(distances)
            closest_predicted = predicted_sources[closest_idx]

            if len(predicted_sources) < len(actual_sources):
                # Check if there are fewer predicted than actual sources and handle accordingly
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
            
            f.write("\tDifferential Entropy:\n")
            if scenario_key in entropy_per_scenario:
                for strategy, entropies in entropy_per_scenario[scenario_key].items():
                    avg_entropy = np.mean(entropies)
                    f.write(f"\t\t{strategy}: Avg Entropy = {avg_entropy:.4f}, Rounds = {len(entropies)}\n")

            f.write("\tTime Taken:\n")
            if scenario_key in time_per_scenario:
                for strategy, times in time_per_scenario[scenario_key].items():
                    # if there are None values in the times list, ignore them
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

def calculate_source_errors(actual_sources, estimated_locs):
    errors = {'x_error': [], 'y_error': [], 'intensity_error': []}
    for actual, estimated in zip(actual_sources, estimated_locs):
        errors['x_error'].append(abs(actual[0] - estimated[0]))
        errors['y_error'].append(abs(actual[1] - estimated[1]))
        errors['intensity_error'].append(abs(actual[2] - estimated[2]))
    return errors

def source_distance(scenario, source_list, save_fig_title=None, show=False):
    """
    Plots errors in estimating the position and intensity of sources as scatter plots with error bars.
    """
    x_errors = []
    y_errors = []
    intensity_errors = []
    n_sources_estimated = []

    # Loop over each actual source and compare it to corresponding predicted source
    for i, actual_source in enumerate(scenario.sources):
        these_x_errors = []
        these_y_errors = []
        these_intensity_errors = []

        # Compare with each round of predictions
        for sources in source_list['source']:
            if sources == []:
                # If no sources were estimated, append NaN values
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
        n_sources_estimated.append(len(these_x_errors))
    # Plotting
    fig, ax = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    source_ids = [f"Source {i+1}" for i in range(len(scenario.sources))]

    for idx, (errors, label) in enumerate(zip([x_errors, y_errors, intensity_errors], ['X Error', 'Y Error', 'Intensity Error'])):
        averages = [np.nanmean(e) for e in errors]
        std_devs = [np.nanstd(e) for e in errors]
        ax[idx].bar(source_ids, averages, yerr=std_devs, alpha=0.6, color='b', label=label)
        ax[idx].set_ylabel('Error')
        ax[idx].set_title(label)
        ax[idx].legend()

        # Add vertical lines for missing data
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

"""
Estimate Sources 
Citation:
M. Morelande, B. Ristic and A. Gunatilaka, "Detection and parameter estimation of multiple radioactive sources," 
2007 10th International Conference on Information Fusion, Quebec, QC, Canada, 2007, pp. 1-7, doi: 10.1109/ICIF.2007.4408094. 
"""

def poisson_log_likelihood(theta, obs_wp, obs_vals, lambda_b, M):
    converted_obs_vals = np.round(obs_vals).astype(int) 

    log_likelihood = 0.0
    sources = theta.reshape((M, 3)) if len(theta) == 3 * M else np.array([theta])
    for obs_index, (x_obs, y_obs) in enumerate(obs_wp):
        lambda_j = lambda_b
        for source in sources:
            x_source, y_source, source_intensity = source
            d_ji = np.sqrt((x_obs - x_source)**2 + (y_obs - y_source)**2)
            # Convert source intensity to expected counts at this point by applying alpha
            alpha_i = source_intensity 
            lambda_j += alpha_i / max(d_ji**2, 1e-6)

        # Use converted_obs_vals which are now in the appropriate count format
        log_pmf = poisson.logpmf(converted_obs_vals[obs_index], lambda_j)
        log_likelihood += log_pmf

    return -log_likelihood  # Minimization in optimization routines

def importance_sampling_with_progressive_correction(obs_wp, obs_vals, lambda_b, M, n_samples, s_stages, prior_dist, alpha=0.5):
    # Step 1: Select γ1, ..., γs (these are parameters that control the tightness of the approximation)
    gammas = np.zeros(s_stages)
    gammas[0] = 0.1
    for k in range(1, s_stages):
        gammas[k] = gammas[k-1] + (1 - gammas[0]) / (s_stages - 1)
        
    # Initialize theta samples either from previous samples or from the prior
    theta_samples = np.column_stack([dist.rvs(n_samples) for dist in prior_dist])
    # print("Initial theta samples:", theta_samples)
    theta_samples_prev = theta_samples.copy()  # Store the initial sample for the first iteration
    # print("Initial theta samples:", theta_samples)

    def calc_weights(theta, gamma, obs_wp, obs_vals, lambda_b, M):
        n_samples = len(theta)
        log_weights = np.zeros(n_samples)
        for i in range(n_samples):
            sample_log_likelihood = -poisson_log_likelihood(theta[i], obs_wp, obs_vals, lambda_b, M)
            log_weights[i] = gamma * sample_log_likelihood
            
        weights = np.exp(log_weights - np.max(log_weights))  # Normalize the weights
        weights /= np.sum(weights)  # Ensure the weights sum to 1
        return weights

    # for k in tqdm(range(s_stages), desc="Progressive Correction"):
    for k in range(s_stages):
        gamma = gammas[k]
        weights = calc_weights(theta_samples_prev, gamma, obs_wp, obs_vals, lambda_b, M)
        indices = np.random.choice(n_samples, size=n_samples, p=weights, replace=True)
        resampled_samples = theta_samples[indices, :]

        # Compute KDE parameters once for each source parameter across all samples
        x_means = resampled_samples[:, ::3].mean(axis=0)
        y_means = resampled_samples[:, 1::3].mean(axis=0)
        intensity_means = resampled_samples[:, 2::3].mean(axis=0)

        x_stds = resampled_samples[:, ::3].std(axis=0, ddof=1)
        y_stds = resampled_samples[:, 1::3].std(axis=0, ddof=1)
        intensity_stds = resampled_samples[:, 2::3].std(axis=0, ddof=1)

        # Sample perturbations in a vectorized way
        x_perturbations = norm.rvs(loc=x_means, scale=x_stds*alpha, size=(n_samples, M))
        y_perturbations = norm.rvs(loc=y_means, scale=y_stds*alpha, size=(n_samples, M))
        intensity_perturbations = norm.rvs(loc=intensity_means, scale=intensity_stds*alpha, size=(n_samples, M))

        # Directly assign perturbations without reshaping
        resampled_samples[:, ::3] = x_perturbations
        resampled_samples[:, 1::3] = y_perturbations
        resampled_samples[:, 2::3] = intensity_perturbations

        # Update the theta samples for the next iteration
        theta_samples = resampled_samples
        theta_samples_prev = theta_samples.copy()


    # Compute the final parameter estimate
    theta_estimate = np.mean(theta_samples, axis=0)
    return theta_estimate, theta_samples
    
def calculate_bic(log_likelihood, num_params, num_data_points):
    """Calculate the Bayesian Information Criterion."""
    bic = - 2 * log_likelihood + num_params * np.log(num_data_points)
    return bic

def estimate_sources_bayesian(obs_wp, obs_vals, lambda_b, max_sources, n_samples, s_stages):
    best_bic = -np.inf
    best_estimate = None
    best_M = 0
    # for M in tqdm(range(1, max_sources + 1), desc="Estimating Sources"):
    for M in range(1, max_sources + 1):
        #print(f"Estimating sources for M = {M}/{max_sources}")
        # Define the prior distribution for source parameters (uniform within workspace)
        prior_x = uniform(loc=0, scale=40)  # Uniform distribution for x
        prior_y = uniform(loc=0, scale=40)  # Uniform distribution for y
        # from 1e3 to 1e5
        prior_intensity = uniform(loc=1e4, scale=1e5)
        # Prior distribution for all parameters of all sources
        prior_dist = [prior_x, prior_y, prior_intensity] * M

        theta_estimate, theta_samples = importance_sampling_with_progressive_correction(
            obs_wp,
            obs_vals,
            lambda_b,
            M,
            n_samples,
            s_stages,
            prior_dist
        )
        # Compute the posterior expectation of the theta estimate
        log_likelihood = -poisson_log_likelihood(theta_estimate, obs_wp, obs_vals, lambda_b, M)

        # Number of parameters is 3 times the number of sources
        num_params = 3 * M
        
        # Calculate BIC
        bic = calculate_bic(log_likelihood, num_params, len(obs_vals))
        #print("BIC:", bic)
        if bic > best_bic:
            best_bic = bic
            best_estimate = theta_estimate
            best_M = M
    #print("prev_theta_samples:", len(prev_theta_samples))
    return best_estimate, best_M, best_bic

def create_gif(save_dir, output_filename="sources_estimation_test_0.gif"):
    images = []
    test_str = "test_" + output_filename.split("_")[-1].split(".")[0] 
    image_names = [f for f in os.listdir(save_dir) if f.endswith(f"{test_str}.png")]
    image_names.sort(key=lambda x: int(x.split("_")[1]))
    for image_name in image_names:
        images.append(imageio.imread(os.path.join(save_dir, image_name)))
    imageio.mimsave(os.path.join(save_dir, output_filename), images, fps=1)

def save_plot_iteration(i, scenario, estimated_locs, obs_wp_send):
        print(f"Plotting iteration {i}")
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Estimated Sources Iteration: {}".format(i))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        # Plot actual sources
        for source in scenario.sources:
            ax.plot(source[0], source[1], 'ro', markersize=10, label='Actual Source')
        
        # Plot estimated sources
        for est_source in estimated_locs:
            ax.plot(est_source[0], est_source[1], 'bx', markersize=10, label='Estimated Source')
        
        # Plot observation waypoints
        for obs in obs_wp_send:
            ax.plot(*obs, 'go', markersize=5)
        
        ax.set_xlim([0, 40])
        ax.set_ylim([0, 40])
        # legend outside the plot to the right
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Save the plot
        # plt.savefig(os.path.join(save_dir, f"iteration_{i}_test_" + str(len(scenario.sources)) + ".png"))
        plt.show()
        plt.close(fig)

def log_estimation_details(iteration, estimated_locs, estimated_num_sources, actual_sources):
    print(f"Iteration {iteration}:")
    print("Estimated Locations:", estimated_locs)
    print("Estimated Number of Sources:", estimated_num_sources)
    print("Actual Sources:", actual_sources)
    
    # Calculate and print errors
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
    
    # Number of sources error
    num_sources_error = abs(len(actual_sources) - estimated_num_sources)
    print(f"Number of Sources Error: {num_sources_error}")
  
if __name__ == "__main__":
    # Example on how to estimate sources using the both methods
    from radiation import RadiationField
    from boustrophedon import Boustrophedon
    # Directory setup for saving plots
    save_dir = "../runs_review/sources_test"
    # Simulation and Bayesian estimation parameters
    workspace_size = (40, 40)
    budget = 375
    n_samples = 200  # Number of samples for importance sampling
    s_stages = 25   # Number of stages for progressive correction
    max_sources = 3 # Max number of sources to test
    lambda_b = 1    # Background radiation level
    iteration_step = 10  # Iteration step for progressive estimation
    save_dir = save_dir + str(s_stages) + "_stages_" + str(n_samples) + "_samples_" + str(max_sources) + "_sources"
    os.makedirs(save_dir, exist_ok=True)

    for num_sources in range(1, 4, 1):
        # Setup scenario and boustrophedon
        scenario = RadiationField(workspace_size=workspace_size, num_sources=num_sources, seed=42)
        boust = Boustrophedon(scenario, budget=budget)
        
        Z_pred, std = boust.run()
        Z_true = scenario.ground_truth()    
        RMSE = np.sqrt(np.mean((np.log10(Z_true + 1) - np.log10(Z_pred + 1))**2))
        
        current_obs_wp = boust.obs_wp.copy()
        current_obs_vals = boust.measurements.copy()
        theta_samples = []
        iteration_counter = 1
        
        bic_vals = []
        best_theta_samples = []
        # Simulation loop
        for iter_val in range(iteration_step, len(boust.obs_wp) + iteration_step, iteration_step):
            iter_val = min(iter_val, len(boust.obs_wp))  # Ensure we do not go beyond the total number of waypoints
            obs_vals_send, obs_wp_send = current_obs_vals[:iter_val], current_obs_wp[:iter_val]
            
            estimated_locs, estimated_num_sources, bic = estimate_sources_bayesian(
                obs_wp_send, obs_vals_send, lambda_b, max_sources, n_samples, s_stages
                  )
            
            if iteration_counter == 1 or bic > max(bic_vals):
                bic_vals.append(bic)
                best_estimated_locs = estimated_locs
                best_estimated_num_sources = estimated_num_sources
            else:
                bic_vals.append(max(bic_vals))
            best_estimated_locs = best_estimated_locs.reshape((-1, 3))

            # Log details of the current estimation
            log_estimation_details(iteration_counter, best_estimated_locs, best_estimated_num_sources, scenario.sources)
            
            save_plot_iteration(iteration_counter, scenario, best_estimated_locs, obs_wp_send)
            iteration_counter += 1
        # plot the evolution of the bic
        plt.figure()
        plt.plot(bic_vals)
        plt.title("BIC Evolution")
        plt.xlabel("Iteration")
        plt.ylabel("BIC")
        plt.grid()
        plt.savefig(os.path.join(save_dir, f"bic_evolution_" + str(num_sources) + ".png"))
        plt.close()
        create_gif(save_dir, "sources_estimation_test_" + str(num_sources) + ".gif")

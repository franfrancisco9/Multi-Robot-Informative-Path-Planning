import matplotlib.pyplot as plt
from matplotlib import ticker, colors
import numpy as np
import os 
from tqdm import tqdm

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
    distance_histogram(scenario, path.obs_wp, save_fig_title.replace('.png', '_histogram.png'), show=show)
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
            # order by the average of the values
            for strategy, values in sorted(scenario_rmse.items(), key=lambda x: np.mean(x[1])):
                f.write(f"{strategy}: Avg RMSE = {np.mean(values):.4f}\n")
        
        for scenario, scenario_entropy in entropy_list.items():
            f.write(f"\n{scenario} Differential Entropy:\n")
            # order by the average of the values
            for strategy, values in sorted(scenario_entropy.items(), key=lambda x: np.mean(x[1])):
                f.write(f"{strategy}: Avg Entropy = {np.mean(values):.4f}\n")
    
    print(f"Run information saved to {filename}")

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from radiation import RadiationField

intensity = RadiationField(workspace_size=(40, 40), num_sources=1).intensity

# The estimate sources function

# l(z| theta ) = prod (i=1 to m) P(z_j;lambda_j(theta))
# z are the m measurements (obs_vals)
# P(z;func) = exp(-func)func^z/z! (Poission distribution) evaluated at
# each z with func = lambda_j(theta) = lambda_b + sum (i=1 to r) alpha_i/d_ji
# where r is the number of sources, d_ji is the distance between the jth observation and the ith source
# lambda_b is the average count due to background radiation and we assume it is known

# the goals is to use MLE to maximize the likelihood function such that we find the vector of~
# theta sources such that theta_hat_ML = arg max _theta l(z|theta)
# and to estimate the number of sources we consider M = {0, ..., M_max} and the best number of sources is
# the best beta_r = log p(z|theta_hat_ml,r) - 1/2 log |J(theta_hat_ml,r)| and J(theta) is the Fisher information matrix

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
import cma

def poisson_log_likelihood(theta, obs_wp, obs_vals, lambda_b, M, alpha=0.001):
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

import numpy as np
from scipy.stats import poisson

def estimate_parameters(obs_wp, obs_vals, lambda_b, M):
    # Initial guess and bounds adjusted for using the log of intensity values
    sigma = 0.5  # Step size
    lower_bounds = [0, 0, 1e3] * M  # Log of intensity lower bound
    upper_bounds = [40, 40, 1e5] * M  # Log of intensity upper bound
    initial_guess = np.random.uniform(lower_bounds, upper_bounds)
    es = cma.CMAEvolutionStrategy(initial_guess, sigma, {'bounds': [lower_bounds, upper_bounds]})
    es.optimize(lambda x: poisson_log_likelihood(x, obs_wp, obs_vals, lambda_b, M))

    # Best solution with exponentiated intensities
    xbest_transformed = es.result.xbest
    return xbest_transformed

def estimate_parameters_nelder_mead(obs_wp, obs_vals, lambda_b, M):
    initial_guess = np.concatenate([np.random.uniform(0, 40, 2*M), np.random.uniform(1e3, 1e5, M)])
    result = minimize(lambda theta: -poisson_log_likelihood(theta, obs_wp, obs_vals, lambda_b, M),
                      initial_guess, method='l-bfgs-b', bounds=[(0, 40)] * 2*M + [(1e3, 1e5)] * M)
    if result.success:
        estimated_theta = result.x
        return estimated_theta
    else:
        return None
    
def compute_FIM(obs_wp, estimated_theta, lambda_b, M):
    FIM = np.zeros((3 * M, 3 * M))
    epsilon = 1e-6
    # Extract source positions and intensities from estimated_theta
    source_positions = estimated_theta[:2*M].reshape((M, 2))
    source_intensities = estimated_theta[2*M:]

    for j, obs_point in enumerate(obs_wp):
        for i in range(M):
            x_i, y_i = source_positions[i]
            alpha_i = source_intensities[i]
            d_ji = max(epsilon, np.sqrt((obs_point[0] - x_i)**2 + (obs_point[1] - y_i)**2))
            d_ji_squared = d_ji**2

            # Compute partial derivatives as per the paper
            d_lambda_j_d_xi = (2 * alpha_i * (obs_point[0] - x_i)) / d_ji_squared
            d_lambda_j_d_yi = (2 * alpha_i * (obs_point[1] - y_i)) / d_ji_squared
            d_lambda_j_d_alpha_i = 1 / d_ji

            # Stack the derivatives for all sources to form the gradient
            gradient = np.zeros(3 * M)
            gradient[3*i] = d_lambda_j_d_xi
            gradient[3*i + 1] = d_lambda_j_d_yi
            gradient[3*i + 2] = d_lambda_j_d_alpha_i

            # Calculate lambda_j for the current observation and source parameters
            lambda_j = lambda_b
            for k, (x_k, y_k) in enumerate(source_positions):
                alpha_k = source_intensities[k]
                d_jk = max(epsilon, np.sqrt((obs_point[0] - x_k)**2 + (obs_point[1] - y_k)**2))
                lambda_j += alpha_k / d_jk**2

            # Update the FIM with the outer product of the gradient, scaled by 1/lambda_j
            FIM += np.outer(gradient, gradient) / lambda_j

    return FIM

def estimate_sources(obs_wp, obs_vals, lambda_b, M_max):
    best_score = -np.inf  # Initialize to negative infinity for maximization
    best_model = None
    best_M = None
    epsilon = 1e-6
    for M in range(1, M_max + 1):
        bounds = [(0, np.max(obs_wp)), (0, np.max(obs_wp))] * M + [(1e3, 1e5)] * M
        estimated_theta = estimate_parameters_nelder_mead(obs_wp, obs_vals, lambda_b, M)
        if estimated_theta is None:
            estimated_theta = np.array([0, 0, 1e3] * M)
        print(f"Estimated theta: {estimated_theta}")
        if estimated_theta is not None:
            # Compute the likelihood at the estimated parameters, ensure it's the likelihood, not negative log-likelihood
            # Since the poisson_log_likelihood function returns the negative log-likelihood for minimization, negate its result.
            log_likelihood = -(poisson_log_likelihood(estimated_theta, obs_wp, obs_vals, lambda_b, M))
            
            # Calculate the Fisher Information Matrix
            FIM = compute_FIM(obs_wp, estimated_theta, lambda_b, M)

            # Ensure the determinant of FIM is positive to avoid taking log of a non-positive number
            det_FIM = np.linalg.det(FIM)
            if det_FIM <= 0:
                print("Warning: Determinant of FIM is non-positive, adjusting to epsilon.")
                det_FIM = -det_FIM + epsilon

            # Compute the penalty term using the determinant of the Fisher Information Matrix
            penalty = -0.5 * np.log(det_FIM)

            # Calculate beta_r according to the formula
            beta_r = log_likelihood + penalty  # Note: log_likelihood is already the log of the probability, not negative

            print(f"Number of sources: {M}, Beta_r: {beta_r}")
            if beta_r > best_score:
                best_score = beta_r
                best_model = estimated_theta
                best_M = M
        else:
            estimated_theta = np.array([0, 0, 1e3] * M)

    return best_model, best_M

def importance_sampling_with_progressive_correction(obs_wp, obs_vals, lambda_b, M, n_samples, s_stages, prior_dist, alpha=0.001):
    # Step 1: Select γ1, ..., γs (these are parameters that control the tightness of the approximation)
    gammas = np.linspace(0.1, 1, s_stages)
    # print("Gammas:", gammas)
    obs_vals = np.round(obs_vals).astype(int)
    # Step 2: Draw initial samples from the prior distribution
    theta_samples = np.column_stack([dist.rvs(n_samples) for dist in prior_dist])
    # print("Initial theta samples:", theta_samples)

    def calc_weights(theta, gamma, obs_wp, obs_vals, lambda_b, alpha):
        log_weights = np.zeros(n_samples)
        M = theta.shape[1] // 3
        # print("M:", M)
        # print("Gamma:", gamma)
        max_log_weight = -np.inf  # Initialize with the lowest possible value
        converted_obs_vals = np.round(obs_vals).astype(int)
        for i in range(n_samples):
            # print("Sample:", i)
            sample_log_weight = 0.0
            for obs_index, (x_obs, y_obs) in enumerate(obs_wp):
                lambda_j = lambda_b
                for source_index in range(M):
                    x_source, y_source, source_intensity = theta[i, source_index*3:(source_index+1)*3]
                    d_ji = max(np.sqrt((x_obs - x_source)**2 + (y_obs - y_source)**2), 1e-6)
                    lambda_j += source_intensity / d_ji**2
                
                log_pmf = poisson.logpmf(converted_obs_vals[obs_index], lambda_j)
                # print("Log PMF:", log_pmf)
                sample_log_weight += log_pmf 
                # print("Sample log weight:", sample_log_weight)
                
            
            sample_log_weight = gamma * sample_log_weight
            # print("Sample log weight:", sample_log_weight)
            log_weights[i] = sample_log_weight
            # print("Sample log weight:", sample_log_weight)
            max_log_weight = max(max_log_weight, sample_log_weight)

        # Apply log-sum-exp trick to prevent underflow when exponentiating
        weights = np.exp(log_weights - max_log_weight)
        # print("Weights:", weights)
        # Normalize weights
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights /= weight_sum
        else:
            weights = np.ones(n_samples) / n_samples
        # print("Normalized weights:", weights)
        return weights

    # Step 3: Iterate through the stages
    with tqdm(total=s_stages) as pbar:
        for k, gamma in enumerate(gammas):
            pbar.set_description(f"Stage {k+1}/{s_stages}")
            pbar.update(1)
            # a) Compute the weights
            weights = calc_weights(theta_samples, gamma, obs_wp, obs_vals, lambda_b, alpha)
            # replace NaNs with 0
            weights = np.nan_to_num(weights)
            # b) Resample according to the weights
            indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True, p=weights)
            theta_samples = theta_samples[indices]
            
            # c) Perturb the resampled particles
            perturbations = np.random.normal(0, alpha, size=theta_samples.shape)
            theta_samples += perturbations
            print("Theta Estimate mean:", np.mean(theta_samples, axis=0))

        # Step 4: Compute the parameter estimate as the mean of the samples
        theta_estimate = np.mean(theta_samples, axis=0)

        return theta_estimate

def calculate_bic(log_likelihood, num_params, num_data_points):
    """Calculate the Bayesian Information Criterion."""
    bic = 2 * log_likelihood + num_params * np.log(num_data_points)
    return bic

def estimate_sources_bayesian(obs_wp, obs_vals, lambda_b, max_sources, n_samples, s_stages):
    best_bic = -np.inf
    best_estimate = None
    best_M = 0
    
    for M in range(1, max_sources + 1):
        print(f"Estimating sources for M = {M}/{max_sources}")
        # Define the prior distribution for source parameters (uniform within workspace)
        prior_x = uniform(loc=0, scale=40)  # Uniform distribution for x
        prior_y = uniform(loc=0, scale=40)  # Uniform distribution for y
        # from 1e3 to 1e5
        prior_intensity = uniform(loc=1e3, scale=1e5)  # Uniform distribution for intensity

        # Prior distribution for all parameters of all sources
        prior_dist = [prior_x, prior_y, prior_intensity] * M

        theta_estimate = importance_sampling_with_progressive_correction(
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
        print("BIC:", bic)
        if bic > best_bic:
            best_bic = bic
            best_estimate = theta_estimate
            best_M = M
            
    return best_estimate, best_M

if __name__ == "__main__":
    from radiation import RadiationField
    from boustrophedon import Boustrophedon
    from scipy.stats import uniform  # Assuming uniform priors

    # Define the prior distribution for source parameters (uniform within workspace)

    # Example usage
    scenario = RadiationField(workspace_size=(40, 40), num_sources=1)
    boust = Boustrophedon(scenario, budget=375)

    Z_pred, std = boust.run()
    Z_true = scenario.ground_truth()

    RMSE = np.sqrt(np.mean((np.log10(Z_true + 1) - np.log10(Z_pred + 1))**2))

    helper_plot(scenario, 1, Z_true, Z_pred, std, boust, RMSE, 1, 1, save=False, show=False)
    # Example usage (placeholders for obs_wp and obs_vals)
    print("Boust observation waypoints:", boust.obs_wp)
    print("Boust observation values:", boust.measurements)


    # Call the Bayesian estimation function
    n_samples = 1000  # Number of samples for importance sampling
    s_stages = 25  # Number of stages for progressive correction
    max_sources = 4  # Max number of sources to test
    lambda_b = 1  # Background radiation level
    # Estimate sources and number of sources using Bayesian approach
    estimated_locs, estimated_num_sources = estimate_sources_bayesian(
        boust.obs_wp, boust.measurements, lambda_b, max_sources,
        n_samples, s_stages
    )
    # rearrange estimated_locs such that every 3 are an array
    estimated_locs = estimated_locs.reshape((-1, 3))
    print("Estimated Locations:", estimated_locs)
    print("Estimated Number of Sources:", estimated_num_sources)
    print("Actual Sources:", scenario.sources)
    # Errors 
    for i, source in enumerate(scenario.sources):
        if i >= estimated_num_sources:
            print(f"Source {i+1} Error: Not estimated")
        else:
            print(f"X Error: {abs(source[0] - estimated_locs[i][0])}" + f" Error (%): {abs(source[0] - estimated_locs[i][0]) / source[0] * 100}%" )
            print(f"Y Error: {abs(source[1] - estimated_locs[i][1])}" + f" Error (%): {abs(source[1] - estimated_locs[i][1]) / source[1] * 100}%" )
            print(f"Norm error: {np.linalg.norm([source[0] - estimated_locs[i][0], source[1] - estimated_locs[i][1]])}")
            print(f"Intensity Error (%): {abs(source[2] - estimated_locs[i][2]) / source[2] * 100}%")

    # Number of sources error
    print(f"Number of Sources Error: {abs(len(scenario.sources) - estimated_num_sources)}")
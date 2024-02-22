import numpy as np
import matplotlib.pyplot as plt
import cma
from scipy.interpolate import make_interp_spline

# Function to generate B-spline path
def generate_b_spline_path(waypoints, resolution=100):
    x = waypoints[:, 0]
    y = waypoints[:, 1] 
    t = np.linspace(0, 1, len(waypoints))
    t_new = np.linspace(0, 1, resolution)
    spline_x = make_interp_spline(t, x, k=3)
    spline_y = make_interp_spline(t, y, k=3)
    x_new = spline_x(t_new)
    y_new = spline_y(t_new)
    return x_new, y_new

def calculate_entropy(waypoints, workspace_size=(40, 40), bins=(8, 8)):
    hist, _ = np.histogramdd(waypoints, bins=bins, range=[[0, workspace_size[0]], [0, workspace_size[1]]], density=True)
    hist = hist.flatten()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist))
    return entropy

def objective_function(waypoints, workspace_size, budget, nominal_path):
    waypoints = np.array(waypoints).reshape(-1, 2)
    # Ensure waypoints start and end at (0, 0)
    waypoints = np.vstack(([0, 0], waypoints, [0, 0]))
    length = calculate_path_length(waypoints)
    budget_penalty = max(0, length - budget) 
    workspace_penalty_value = workspace_penalty(waypoints, workspace_size) * 50
    # deviation_penalty_value = deviation_penalty(waypoints, nominal_path)
    return -calculate_entropy(waypoints, workspace_size) + budget_penalty + workspace_penalty_value# + deviation_penalty_value

def calculate_path_length(waypoints):
    return np.sum(np.sqrt(np.sum(np.diff(waypoints, axis=0) ** 2, axis=1)))

def workspace_penalty(waypoints, workspace_size):
    penalty = 0
    for point in waypoints:
        if not (0 <= point[0] <= workspace_size[0]) or not (0 <= point[1] <= workspace_size[1]):
            penalty += np.inf
    return penalty

def deviation_penalty(waypoints, nominal_path):
    """
    Calculate the penalty for deviating from the nominal path.
    """
    penalty = 0
    # Iterate through each waypoint and find its minimum distance to the nominal path
    for wp in waypoints:
        dists = np.sqrt(np.sum((nominal_path - wp) ** 2, axis=1))  # Calculate distances to all points in the nominal path
        penalty += np.min(dists)  # Add the minimum distance to the penalty
    return penalty
def generate_nominal_path(workspace_size=(40, 40), n_waypoints=20, distance_budget=100, nominal_path=np.array([[0, 0], [40, 40]])):
    # Adjust the number of waypoints for optimization by subtracting the fixed start and end points
    adjustable_waypoints = n_waypoints - 2
    
    # Define lower and upper bounds for the adjustable waypoints
    lower_bounds = [0] * (adjustable_waypoints * 2)
    upper_bounds = [workspace_size[0]] * adjustable_waypoints + [workspace_size[1]] * adjustable_waypoints
    bounds = [lower_bounds, upper_bounds]
    
    # Create an initial guess for the adjustable waypoints, ensuring it doesn't include the start and end points
    x0 = np.random.rand(adjustable_waypoints * 2) * workspace_size[0]
    sigma = workspace_size[0] / 4  # Initial step size for CMA-ES
    
    # Optimize the adjustable waypoints
    es = cma.CMAEvolutionStrategy(x0, sigma, {'bounds': bounds})
    es.optimize(lambda x: objective_function(x.reshape((-1, 2)), workspace_size, distance_budget, nominal_path))
    
    # Reshape the result from CMA-ES and prepend and append the start and end points at (0, 0)
    optimized_waypoints = np.vstack(([0, 0], es.result.xbest.reshape((-1, 2)), [0, 0]))
    
    return optimized_waypoints


if __name__ == "__main__":
    n_waypoints = 29
    distance_budget = 250
    optimal_waypoints = generate_nominal_path(n_waypoints=n_waypoints, distance_budget=distance_budget)

    x_new, y_new = generate_b_spline_path(optimal_waypoints)

    plt.figure(figsize=(8, 8))
    plt.plot(optimal_waypoints[:, 0], optimal_waypoints[:, 1], 'o', label='Waypoints')
    plt.plot(x_new, y_new, '-', label='B-spline Path')
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Nominal Path with B-splines and Distance Budget')
    plt.legend()
    plt.grid(True)
    plt.show()
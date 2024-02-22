import numpy as np
import matplotlib.pyplot as plt
import cma
from scipy.interpolate import make_interp_spline

class InformativePathPlanning:
    def __init__(self, workspace_size=(40, 40), n_waypoints=20, distance_budget=100):
        self.workspace_size = workspace_size
        self.n_waypoints = n_waypoints
        self.distance_budget = distance_budget
        self.nominal_path = None

    def generate_b_spline_path(self, waypoints, resolution=100):
        x, y = waypoints[:, 0], waypoints[:, 1]
        t = np.linspace(0, 1, len(waypoints))
        spline_x = make_interp_spline(t, x, k=3)(np.linspace(0, 1, resolution))
        spline_y = make_interp_spline(t, y, k=3)(np.linspace(0, 1, resolution))
        return spline_x, spline_y

    def calculate_entropy(self, waypoints):
        hist, _ = np.histogramdd(waypoints, bins=8, range=[[0, self.workspace_size[0]], [0, self.workspace_size[1]]], density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))

    def objective_function(self, waypoints):
            # This now only takes waypoints as an input.
            waypoints = np.vstack(([0, 0], waypoints.reshape(-1, 2), [0, 0]))
            length = self.calculate_path_length(waypoints)
            budget_penalty = max(0, length - self.distance_budget) * 50
            workspace_penalty_value = self.workspace_penalty(waypoints) * 50  
            entropy_penalty = -self.calculate_entropy(waypoints)  # Directly use instance method
            return entropy_penalty + budget_penalty + workspace_penalty_value

    def workspace_penalty(self, waypoints):
        # Adjusted to use instance attribute for workspace_size
        penalty = 0
        for point in waypoints:
            if not (0 <= point[0] <= self.workspace_size[0]) or not (0 <= point[1] <= self.workspace_size[1]):
                penalty += np.inf
        return penalty
    
    def calculate_path_length(self, waypoints):
        return np.sum(np.sqrt(np.sum(np.diff(waypoints, axis=0) ** 2, axis=1)))

    def generate_nominal_path(self):
        adjustable_waypoints = self.n_waypoints - 2
        x0 = np.random.rand(adjustable_waypoints * 2) * self.workspace_size[0]
        bounds = [0, self.workspace_size[0]]  # Simplified bounds for optimization
        es = cma.CMAEvolutionStrategy(x0, self.workspace_size[0] / 4, {'bounds': bounds})
        es.optimize(lambda x: self.objective_function(x))  # Lambda function to pass only waypoints
        self.nominal_path = np.vstack(([0, 0], es.result.xbest.reshape((-1, 2)), [0, 0]))
        return self.nominal_path

    def plot_path(self):
        if self.nominal_path is None:
            print("Nominal path not generated yet.")
            return
        x_new, y_new = self.generate_b_spline_path(self.nominal_path)
        plt.figure(figsize=(8, 8))
        plt.plot(self.nominal_path[:, 0], self.nominal_path[:, 1], 'o', label='Waypoints')
        plt.plot(x_new, y_new, '-', label='B-spline Path')
        plt.xlim(0, self.workspace_size[0])
        plt.ylim(0, self.workspace_size[1])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Nominal Path with B-splines and Distance Budget')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    ipp = InformativePathPlanning(n_waypoints=29, distance_budget=290)
    ipp.generate_nominal_path()
    ipp.plot_path()

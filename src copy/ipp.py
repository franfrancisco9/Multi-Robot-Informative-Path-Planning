import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from numpy.linalg import det
from bspline import bspline
from radiation import RadiationField


class InformativePathPlanning:
    def __init__(self, workspace_size=(40, 40), n_waypoints=200, distance_budget=2000):
        self.workspace_size = workspace_size
        self.n_waypoints = n_waypoints
        self.distance_budget = distance_budget
        self.nominal_path = None
        self.nominal_spread = None
        self.gp = GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)), n_restarts_optimizer=10)

    def Boustrophedon(self, d_waypoint_distance=2.5):
        cv = np.array([
                [ 0.5,  0.5],
                [ 0.5, 39.5],
                [ 5.,  39.5],
                [ 5.,  0.5],
                [ 10.,  0.5],
                [ 10., 39.5],
                [ 15., 39.5],
                [ 15.,  0.5],
                [ 20.,  0.5],
                [ 20., 39.5],
                [ 25., 39.5],
                [ 25.,  0.5],
                [ 30.,  0.5],
                [ 30., 39.5],
                [ 35., 39.5],
                [ 35.,  0.5],
                [ 39.5,  0.5],
                [ 39.5, 39.5]
                ])
        d = 2
        p = bspline(cv,n=1000,degree=d)
        p_way = [p[0]]
        for i in range(1, len(p)):
            if np.linalg.norm(p[i] - p_way[-1]) > d_waypoint_distance:
                p_way.append(p[i])
        p_way = np.array(p_way)
        self.nominal_path = p.T
        self.nominal_spread = p_way

    def generate_spline_path(self):
        if self.nominal_path is None:
            print("No path planned.")
            return

        x, y = self.nominal_path[:, 0], self.nominal_path[:, 1]
        spline_x = make_interp_spline(range(len(x)), x, bc_type='natural')
        spline_y = make_interp_spline(range(len(y)), y, bc_type='natural')
        t_new = np.linspace(0, len(x) - 1, 1000)
        x_new, y_new = spline_x(t_new), spline_y(t_new)

        return x_new, y_new

    def plot_path(self):
        x_new, y_new = self.nominal_path
        plt.figure(figsize=(10, 10))
        plt.plot(x_new, y_new, 'g-', label='Nominal Path')
        plt.plot(self.nominal_spread[:, 0], self.nominal_spread[:, 1], 'ro', label='Waypoints')
        plt.xlim(0, self.workspace_size[0])
        plt.ylim(0, self.workspace_size[1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Boustrophedon Path')
        # LOCATE LEGEND OUTSIDE OF PLOT
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()


    def path_length(self, path):
        lenght = 0
        for i in range(len(path)-1):
            lenght += np.linalg.norm(path[i+1]-path[i])
        return lenght
        
    def observe_path(self, path, O_p):
        scenario = RadiationField(num_sources=1, workspace_size=(40, 40))
        measurements = scenario.simulate_measurements(path)
        Z_pred = scenario.predict_spatial_field(path, measurements)
        return Z_pred
    
    def Optimize_Path(self, p_agent, p_waypoint, p_next_waypoint, budget_i, travel_distance, inter_observation_distance, horizon_distance, O_p):
        # Define the objective function for optimization
        def objective_function(X):
            # Reshape X to match sklearn's expected input format
            X = X.reshape(1, -1)
            
          # Use the GP model to predict the mean and standard deviation for point X
            pred, std = self.gp.predict(X, return_std=True)
            print("pred: ", pred, "std: ", std)
            # Instead of using determinant, use the uncertainty (std) directly
            # Higher uncertainty indicates higher information gain when exploring this point
            # Negate because we want to maximize this uncertainty (minimize negative uncertainty)
            utility = 0.5 * np.log(np.linalg.det(np.identity(len(pred)) * std))
            return -utility
        # Example of optimizing with bounds - this needs to be adjusted to your scenario
        bounds = [(0, self.workspace_size[0]), (0, self.workspace_size[1])]
        initial_guess = (p_waypoint + p_next_waypoint) / 2
        
        result = minimize(objective_function, initial_guess, bounds=bounds)
        
        # Extract the optimized waypoint and construct the path
        optimized_waypoint = result.x
        optimized_path = np.vstack([p_agent, optimized_waypoint, p_next_waypoint])
        
        return optimized_path

    def IPP(self, budget_fraction=1, horizon_distance=2.5, travel_distance=2.5, inter_observation_distance=2.5, O_p=[]):
        budget_spent = 0
        budget_nominal = budget_fraction * self.distance_budget
        Q = []

        self.Boustrophedon()
        N_waypoints = self.nominal_spread
        p_agent = N_waypoints[0]

        while budget_spent < budget_nominal and len(N_waypoints) > 1:
            # Select next waypoint based on the budget and remaining waypoints
            p_next_waypoint = N_waypoints[1]
            budget_i = min(budget_nominal - budget_spent, travel_distance)
            
            # Optimize the path from the current position to the next waypoint
            P = self.Optimize_Path(p_agent, p_agent, p_next_waypoint, budget_i, travel_distance, inter_observation_distance, horizon_distance, O_p)
            
            # Simulate measurements along the optimized path
            Z_pred = self.observe_path(P, O_p)
            O_p.extend(list(zip(P, Z_pred)))  # Update observations
            
            # Update the GP model with new observations
            X_obs, Y_obs = zip(*O_p)
            self.gp.fit(np.array(X_obs), np.array(Y_obs))
            
            # Update variables for the next iteration
            p_agent = P[-1]  # Last point of the optimized path
            budget_spent += self.path_length(P)
            Q.append(P)
            N_waypoints = N_waypoints[1:]  # Move to the next segment

        return np.concatenate(Q), O_p
# Example usage
if __name__ == "__main__":
    ipp = InformativePathPlanning(workspace_size=(40, 40), n_waypoints=200, distance_budget=2000)
    ipp.Boustrophedon()
    ipp.plot_path()
    Q, O_p = ipp.IPP()


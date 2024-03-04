import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from bspline import bspline
from radiation import RadiationField

class InformativePathPlanning:
    def __init__(self, workspace_size=(40, 40), n_waypoints=200, distance_budget=2000):
        self.workspace_size = workspace_size
        self.n_waypoints = n_waypoints
        self.distance_budget = distance_budget
        self.nominal_path = None
        self.nominal_spread = None

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


    def path_lenght(self, path):
        lenght = 0
        for i in range(len(path)-1):
            lenght += np.linalg.norm(path[i+1]-path[i])
        return lenght
        
    def observe_path(self, path, O_p):
        scenario = RadiationField(num_sources=1, workspace_size=(40, 40))
        measurements = scenario.simulate_measurements(path)
        Z_pred = scenario.predict_spatial_field(path, measurements)
        return Z_pred
    
    def Optmize_Path(self, p_agent, p_waypoint, p_next_waypoint, budget_i, travel_distance, inter_observation_distance, horizon_distance, O_p):
        # for now lets ignore budgets
        # we want to maximize the following u
        # u(X,O) = 1/2 log(det(I + K(X,O)))
        # X future observations and O past observations
        # K is the kernel matrix
        # I is the identity matrix
        # We want to maximize the information gain over the C control points for a B-spline curve
        # K is extracted from GP trained on data from the past observations
        # GP uses SE kernel
        def u(X, O):
            return 1/2 * np.log(np.linalg.det(np.identity(len(X)) + K(X, O)))
        # K is the kernel matrix
        def K(X, O):
            if np.array(O).shape[0] == 0:  # Check if O is empty
                return np.zeros((len(X), len(X)))  
            # do the norm between lists of points
            def norm(X, O):
                K = np.zeros((len(X), len(O)))
                for i in range(len(X)):
                    for j in range(len(O)):
                        
                        K[i, j] = np.linalg.norm(X[i] - O[j])
                return K
            # set the kernel matrix
            K = np.exp(-norm(X, O)**2)
            return K
        import scipy.optimize as opt
        # Set the initial path
        P = [p_agent, p_waypoint, p_next_waypoint]
        # Define X
        X = []
        # Set the initial information gain
        u_i = u(X, O_p)
        # Set the initial path lenght
        p_lenght = self.path_lenght(P)
        # Set the initial path
        P_i = P
        # call the optimizer
        for i in range(100):
            # Set the new path
            P_new = P_i + np.random.normal(0, 0.1, size=(3, 2))

            X = O_p
            # Set the new information gain
            u_new = u(X, O_p)
            # Set the new path lenght
            p_new_lenght = self.path_lenght(P_new)
            # If the new path is better, update the path
            if u_new > u_i and p_new_lenght < budget_i:
                P_i = P_new
                u_i = u_new
                p_lenght = p_new_lenght
        return P_i



    # Informative Path Planner.
    def IPP(self, budget_fraction=1, horizon_distance=2.5, travel_distance=2.5, inter_observation_distance=2.5, O_p = []):
        i = 0 # Initial Iteration
        budget_spent = 0 # Initial Budget Spent
        budget_nominal = budget_fraction * self.distance_budget # Budget Nominal
        waypoint_distance = budget_fraction * horizon_distance # Inter-waypoint distance
        budget_i = (2-budget_fraction) * horizon_distance # Budget per iteration
        Q = [] # Initial Path
        # From the boustrophedon path, set the nominal path as a Bspline curve 
        self.Boustrophedon() # Nominal Path
        N = self.nominal_path
        N_waypoints = self.nominal_spread
        N_lenght = self.path_lenght(N)
        p_agent = N_waypoints[0] # Initial Agent Position
        while budget_spent < self.distance_budget:
            # Set the waypoints for this iteration
            waypoint_distance = min(horizon_distance + i * waypoint_distance, N_lenght)
            n_waypoint_distance = min(horizon_distance + (i + 1) * waypoint_distance, N_lenght)
            # p_waypoints 
            p_waypoint = N_waypoints[i+1]
            p_next_waypoint = N_waypoints[i+2]
            # Set the budget for this iteration
            budget_i = min(budget_i, self.distance_budget - budget_spent)
            # set the travel distance for this iteration
            travel_distance = min(travel_distance, self.distance_budget - budget_spent)
            # import local optimizer and solve the optimization problem to find the best path to 
            # maximize the information gain
            P = self.Optmize_Path(p_agent, p_waypoint, p_next_waypoint, budget_i, travel_distance, inter_observation_distance, horizon_distance, O_p)
            print(P)
            # plot the current path
            # transform each array item into a list
            for i in range(len(P)):
                P[i] = P[i].tolist()
            P = np.array(P)
            print(P)
            
            x_new, y_new = self.nominal_path
            plt.figure(figsize=(10, 10))
            plt.plot(x_new, y_new, 'g-', label='Nominal Path')
            plt.plot(P[:,0], P[:,1], 'b-', label='Optimized Path')
            plt.plot(self.nominal_spread[:, 0], self.nominal_spread[:, 1], 'ro', label='Waypoints')
            plt.xlim(0, self.workspace_size[0])
            plt.ylim(0, self.workspace_size[1])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Boustrophedon Path')
            # LOCATE LEGEND OUTSIDE OF PLOT
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()
            plt.show()

            # Travel the path with observations up to the travel distance
            if Q == []:
                Q = P
            else:
                Q = Q + P
            if np.array(O_p).shape[0] == 0:
                O_p = self.observe_path(P, [])
            else:
                O_p = O_p + self.observe_path(P, O_p)
            # Update the agent position
            p_agent = Q[-1]
            # Update the budget spent
            budget_spent = budget_spent + travel_distance
            # Update the iteration
            i = i + 1
        return Q, O_p
# Example usage
if __name__ == "__main__":
    ipp = InformativePathPlanning(workspace_size=(40, 40), n_waypoints=200, distance_budget=2000)
    ipp.Boustrophedon()
    ipp.plot_path()
    Q, O_p = ipp.IPP()


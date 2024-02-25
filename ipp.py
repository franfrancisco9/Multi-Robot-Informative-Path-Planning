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
        print(waypoints)
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
            budget_penalty = max(0, length - self.distance_budget) * 10
            workspace_penalty_value = self.workspace_penalty(waypoints) * 10
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
    
    def ComputeWaypoints(self, N, d_waypoint):
        """
        Input: B-spline path N, inter-waypoint distance d_waypoint
        Output: Waypoints W
        """
        # Initialize the waypoints
        W = []
        # Compute the number of segments
        n_segments = len(N) - 1
        # for i in range(n_segments)
        for i in range(n_segments):
            # Compute the segment length
            L = self.calculate_path_length(N[i:i+2])
            # Compute the number of waypoints
            n_waypoints = np.floor(L / d_waypoint)
            # Compute the segment direction
            u = (N[i+1] - N[i]) / np.linalg.norm(N[i+1] - N[i])
            # for j in range(n_waypoints)
            for j in range(n_waypoints):
                # Compute the waypoint position
                w = N[i] + j * d_waypoint * u
                # Append the waypoint to the set
                W.append(w)
        return W
    
    def Calculate2DHistogram(self, W):
        """
        Input: Waypoints W
        Output: 2D histogram H
        """
        # Compute the 2D histogram of the waypoints
        H, _, _ = np.histogram2d(W[:, 0], W[:, 1], bins=8, range=[[0, self.workspace_size[0]], [0, self.workspace_size[1]]], density=True)
        return H
    
    def GenerateNominalPath(self):
        """
        Input: Nominal path budget B_nominal, workspace W, inter-waypoint distance d_waypoint
        Output: Optimized Path N
        """
        # Set the home position
        p_home = [0, 0]
        # Execute the Initialization of the CMAES
        x0 = np.random.rand(self.n_waypoints * 2) * self.workspace_size[0]
        sigma0 = self.workspace_size[0] / 4
        es = cma.CMAEvolutionStrategy(x0, sigma0)
        # while true
        while not es.stop():
            # Sample population from CMAES saved into S
            fit, S = [], []
            while len(S) < es.popsize:
                curr_fit = None
                while curr_fit in (None, np.NaN):
                    x = es.ask(1)[0]
                    curr_fit = cma.ff.somenan(x, cma.ff.elli)
                S.append(x)
                fit.append(curr_fit)
            for c in S: 
                # Prepend and append the home position to complete the set, forcing the path to start and end at the same position
                c = [p_home, c, p_home]
                # Convert the control vertices into a B-spline path
                N = self.generate_b_spline_path(c)
                # Compute the waypoints
                W = self.calculate_waypoints(N)
                # Compute the 2D histogram of the waypoints
                H = self.calculate_2D_histogram(W)
                # Compute the entropy of the 2D histogram
                E = self.calculate_entropy(H)
                # Evaluate the fitness of the solution and add it to the solutions' fitness set
                E_budget_penalty = max(0, self.calculate_path_length(N) - self.distance_budget) * 10
                E_workspace_penalty = self.workspace_penalty(N) * 10
                u = E - E_budget_penalty - E_workspace_penalty
                fit.append(u)
            # update parameters of CMAES using S and U
            es.tell(S, fit)
            es.logger.add()  # write data to disc to be plotted
            es.disp()
        # Return the best solution
        C = es.result.xbest.reshape((-1, 2))
        # transform the middle array into normal 
        print(C)
        N = self.generate_b_spline_path(C)
        return N

        
    def generate_nominal_path(self):
        adjustable_waypoints = self.n_waypoints - 2
        x0 = np.random.rand(adjustable_waypoints * 2) * self.workspace_size[0]
        es = cma.CMAEvolutionStrategy(x0, 0.5)
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
    ipp.GenerateNominalPath()
    ipp.plot_path()

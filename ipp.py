import numpy as np
import matplotlib.pyplot as plt
import cma
from scipy.interpolate import BSpline

class InformativePathPlanning:
    def __init__(self, workspace_size=(40, 40), n_waypoints=20, distance_budget=100):
        self.workspace_size = workspace_size
        self.n_waypoints = n_waypoints
        self.distance_budget = distance_budget
        self.nominal_path = None

    def generate_b_spline_path(self, waypoints, resolution=100):
        # print(waypoints)
        x, y = waypoints[:, 0], waypoints[:, 1]
        t = np.linspace(0, 1, len(waypoints))
        k = 2
        spl = BSpline(t, np.column_stack([x, y]), k)
        return spl
    def calculate_entropy(self, H):
        """
        // Calculate the entropy of the waypoints.
        Given H, the 2D histogram of the waypoints, calculate the entropy of the waypoints.
        example of H [[0.         0.         0.         0.         0.         0.
  0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.        ]
 [0.         0.         0.         0.         0.         0.00444444
  0.         0.        ]
 [0.         0.         0.         0.         0.         0.00888889
  0.         0.        ]
 [0.         0.         0.         0.         0.         0.00444444
  0.         0.        ]
 [0.         0.         0.00444444 0.         0.         0.
  0.00444444 0.        ]
 [0.         0.         0.         0.         0.00444444 0.
  0.00444444 0.        ]
 [0.         0.         0.         0.         0.         0.
  0.00444444 0.        ]]
        """
        H = H / np.sum(H)
        E = -np.sum(H * np.log(H + 1e-10))
        return E


    def objective_function(self, waypoints):
            # This now only takes waypoints as an input.
            waypoints = np.vstack(([0, 0], waypoints.reshape(-1, 2), [0, 0]))
            length = self.calculate_path_length(waypoints)
            budget_penalty = max(0, length - self.distance_budget) * 10
            workspace_penalty_value = self.workspace_penalty(waypoints) * 10
            entropy_penalty = -self.calculate_entropy(waypoints)  # Directly use instance method
            return entropy_penalty + budget_penalty + workspace_penalty_value

    def workspace_penalty(self, G):
        """
        // Compute the workspace penalty of the waypoints distance from the workspace boundary.
        Input: Set of waypoints G
        & g_\mathcal{W}(\mathcal{P}_i, \varepsilon) = \Gamma \left (\int_0 ^{|\mathcal{P}|} \bar  \phi (\mathcal{P}|_l, \varepsilon)dl \right )     \Gamma(x) = \begin{cases}
    cx^2 & \text{if } x  <0 \\ 
    0 & \text{if } x \geq 0
\end{cases}\\
        """
        # Gamma
        def gamma(x):
            if x < 0:
                return 1 * x ** 2
            else:
                return 0
        # Compute the workspace penalty
        penalty = 0
        for i in range(len(G)):
            penalty += gamma(self.workspace_size[0] / 2 - G[i][0])
            penalty += gamma(self.workspace_size[1] / 2 - G[i][1])
        return penalty
    
    
    def calculate_path_length(self, N):
        """
        Input: B-spline path N from scipy.interpolate.BSpline
        Output: Length of the path
        """
        t = np.linspace(0, 1, 100)
        x_new, y_new = N(t).T
        d = np.sqrt(np.diff(x_new) ** 2 + np.diff(y_new) ** 2)
        return np.sum(d)
    
    def ComputeWaypoints(self, N, d_waypoint):
        """
        Input: B-spline path N from scipy.interpolate.BSpline, inter-waypoint distance d_waypoint
        Output: Waypoints W
        """
        # Compute the waypoints from the B-spline path
        t = np.linspace(0, 1, 100)
        x_new, y_new = N(t).T
        # Compute the inter-waypoint distances
        d = np.sqrt(np.diff(x_new) ** 2 + np.diff(y_new) ** 2)
        # Compute the waypoints
        W = np.array([x_new[0], y_new[0]])
        for i in range(1, len(t) - 1):
            if np.sum(d[:i]) >= d_waypoint:
                W = np.vstack((W, [x_new[i], y_new[i]]))
                d = d[i:]
        W = np.vstack((W, [x_new[-1], y_new[-1]]))
        return W
    
    def Calculate2DHistogram(self, W):
        """
        // Compute the 2D histogram of the waypoints.
        """
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
                c = np.array(c).reshape(-1, 2)
                # Convert the control vertices into a B-spline path
                N = self.generate_b_spline_path(c)
                # print("N", N)
                # Compute the waypoints
                G = self.ComputeWaypoints(N, 1)
                # print("G", G)
                # Compute the 2D histogram of the waypoints
                H = self.Calculate2DHistogram(np.array(G))
                # Compute the entropy of the 2D histogram
                # print("H", H)
                E = self.calculate_entropy(np.array(H))
                # print("E", E)
                # Evaluate the fitness of the solution and add it to the solutions' fitness set
                E_budget_penalty = max(0, self.calculate_path_length(N) - self.distance_budget) * 10
                E_workspace_penalty = self.workspace_penalty(G) * 10
                u = E - E_budget_penalty - E_workspace_penalty
                # print("u", u)
                fit.append(u)
            # update parameters of CMAES using S and U
            try:
                es.tell(S, fit)
                es.logger.add()  # write data to disc to be plotted
                es.disp()
            except:
                pass
        # Return the best solution
        C = es.result.xbest.reshape((-1, 2))
        # transform the middle array into normal 
        print(C)
        N = self.generate_b_spline_path(C)
        self.nominal_path = N
        return N
   
    def generate_nominal_path(self):
        adjustable_waypoints = self.n_waypoints - 2
        x0 = np.random.rand(adjustable_waypoints * 2) * self.workspace_size[0]
        es = cma.CMAEvolutionStrategy(x0, 0.05, {'bounds': [0, self.workspace_size[0]]})
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

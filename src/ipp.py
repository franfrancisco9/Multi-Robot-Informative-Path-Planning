import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

class InformativePathPlanning:
    def __init__(self, workspace_size=(40, 40), n_waypoints=200, distance_budget=2000):
        self.workspace_size = workspace_size
        self.n_waypoints = n_waypoints
        self.distance_budget = distance_budget
        self.nominal_path = None

    def Boustrophedon(self, start, end):
        waypoints = [start]
        x, y = start
        num_vertical_lines = 11
        dx = self.workspace_size[0] / num_vertical_lines
        dy = self.workspace_size[1] / ((self.n_waypoints // num_vertical_lines) // 2)

        direction = 1  # Initial direction up
        for _ in range(num_vertical_lines):
            for _ in range((self.n_waypoints // num_vertical_lines) // 2):
                if len(waypoints) + 1 >= self.n_waypoints: break  # Reserve space for the last point
                y += direction * dy
                if (y >= self.workspace_size[1] - 0.5 and direction == 1) or (y <= 0.5 and direction == -1):
                    # Adjust for boundary condition with an intermediary waypoint
                    middle_y = y - direction * dy / 2  # Middle point for the curve
                    waypoints.append((x, middle_y))
                    waypoints.append((x, max(0.5, min(y, self.workspace_size[1] - 0.5))))
                    break
                waypoints.append((x, max(0.5, min(y, self.workspace_size[1] - 0.5))))

            if len(waypoints) + 1 >= self.n_waypoints: break  # Reserve space for the last point
            direction *= -1  # Change direction for the next vertical
            x += dx
            if x < self.workspace_size[0] and len(waypoints) + 1 < self.n_waypoints:
                waypoints.append((x, waypoints[-1][1]))  # Move horizontally at the current height

        # Ensure the last point is reached
        if waypoints[-1] != end:
            waypoints.append(end)

        self.nominal_path = np.array(waypoints)

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
        x_new, y_new = self.generate_spline_path()
        plt.figure(figsize=(10, 10))
        plt.plot(x_new, y_new, 'b-', label='Nominal Path')
        plt.plot(self.nominal_path[:, 0], self.nominal_path[:, 1], 'ro', label='Waypoints')
        plt.xlim(0, self.workspace_size[0])
        plt.ylim(0, self.workspace_size[1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Boustrophedon Path')
        # LOCATE LEGEND OUTSIDE OF PLOT
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

# Example usage
if __name__ == "__main__":
    ipp = InformativePathPlanning(workspace_size=(40, 40), n_waypoints=200, distance_budget=2000)
    ipp.Boustrophedon((0.5, 0.5), (39.5, 39.5))
    ipp.plot_path()

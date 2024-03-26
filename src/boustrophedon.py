"""
Boustrophedon path planner.
- Created by: Francisco Fonseca on March 2024
"""
import numpy as np
from bspline import bspline

class Boustrophedon:
    def __init__(self, scenario, d_waypoint_distance=2.5, budget=375, line_spacing=5):
        """
        Initializes a Boustrophedon path planner.

        Parameters:
        - scenario: The radiation field scenario to plan a path for.
        - d_waypoint_distance: Desired distance between waypoints.
        - budget: The total distance budget available for the path.
        - line_spacing: The spacing between lines in the Boustrophedon pattern.
        """
        self.scenario = scenario
        self.workspace_size = scenario.workspace_size
        self.d_waypoint_distance = d_waypoint_distance
        self.budget = budget
        self.full_path = None
        self.obs_wp = None
        self.name = "Boustrophedon"
        self.line_spacing = line_spacing

    def run(self):
        """
        Generates a Boustrophedon path within a specified distance budget.
        """
        x_coords = np.arange(0.5, self.workspace_size[0], self.line_spacing)
        if x_coords[-1] + 0.5 != self.workspace_size[0]:
            x_coords = np.append(x_coords, self.workspace_size[0] - 0.5)
        
        y_up = self.workspace_size[1] - 0.5
        y_down = 0.5

        cv = []
        for i, x in enumerate(x_coords):
            cv.append([x, y_down if i % 2 == 0 else y_up])
            cv.append([x, y_up if i % 2 == 0 else y_down])

        cv = np.array(cv)
        p = bspline(cv, n=1000, degree=2)

        # Select waypoints based on budget and desired distance between them
        self.obs_wp = [p[0]]
        distance_covered = 0.0
        
        for i in range(1, len(p)):
            segment_length = np.linalg.norm(p[i] - self.obs_wp[-1])
            dist = np.linalg.norm(p[i] - p[i-1])
            if distance_covered + dist > self.budget:
                break  # Stop if adding this segment would exceed the budget
           
            if segment_length >= self.d_waypoint_distance:
                self.obs_wp.append(p[i])
            
            distance_covered += dist

        # print("Distance covered: ", distance_covered)
        self.obs_wp = np.array(self.obs_wp)
        self.full_path = p[:i+1].T
        measurements = self.scenario.simulate_measurements(self.obs_wp)
        return self.scenario.predict_spatial_field(self.obs_wp, measurements)

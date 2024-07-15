"""
Boustrophedon path planner.
- Created by: Francisco Fonseca on March 2024
"""
import numpy as np
from bspline import bspline

class Boustrophedon():
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
        self.obs_wp = np.array(self.obs_wp)
        self.full_path = p[:i+1].T
        self.measurements = self.scenario.simulate_measurements(self.obs_wp)
        return self.scenario.predict_spatial_field(self.obs_wp, self.measurements)


class MultiAgentBoustrophedon:
    def __init__(self, scenario, num_agents=1, d_waypoint_distance=2.5, budget=375, line_spacing=5):
        """
        Initializes a Boustrophedon path planner for multiple agents.
        """
        self.scenario = scenario
        self.workspace_size = scenario.workspace_size
        self.d_waypoint_distance = d_waypoint_distance
        self.total_budget = budget
        self.num_agents = num_agents
        self.line_spacing = line_spacing
        self.agents_full_path = [[] for _ in range(num_agents)]
        self.agents_measurements = [[] for _ in range(num_agents)]
        self.agents_obs_wp = [[] for _ in range(num_agents)]
        self.name = "Multi-Agent Boustrophedon"

        if num_agents > 1:
            # divide the space in num_agents equal parts and set each start at the start of the next segment
            self.agent_positions = [np.array([i * self.workspace_size[0] / num_agents, 0.5]) for i in range(num_agents)]
        else:
            self.agent_positions = [np.array([0.5, 0.5])]

    def run(self):
        """
        Generates Boustrophedon paths for multiple agents within specified distance budgets.
        """
        for i in range(self.num_agents):
            start_x = self.agent_positions[i][0]
            end_x = self.workspace_size[0] - 0.5 if i == self.num_agents - 1 else self.agent_positions[i + 1][0] - self.line_spacing
            budget_per_agent = self.total_budget 
            # print("Start x: ", start_x, "End x: ", end_x, "Budget: ", budget_per_agent)
            x_coords = np.arange(start_x, end_x, self.line_spacing)
            if x_coords[-1] + 0.5 != end_x:
                x_coords = np.append(x_coords, end_x)

            y_up = self.workspace_size[1] - 0.5
            y_down = 0.5

            cv = []
            for j, x in enumerate(x_coords):
                cv.append([x, y_down if j % 2 == 0 else y_up])
                cv.append([x, y_up if j % 2 == 0 else y_down])

            cv = np.array(cv)
            p = bspline(cv, n=1000, degree=2)

            obs_wp = [p[0]]
            distance_covered = 0.0

            for k in range(1, len(p)):
                segment_length = np.linalg.norm(p[k] - obs_wp[-1])
                dist = np.linalg.norm(p[k] - p[k-1])
                if distance_covered + dist > budget_per_agent:
                    break

                if segment_length >= self.d_waypoint_distance:
                    obs_wp.append(p[k])

                distance_covered += dist

            self.agents_obs_wp[i] = np.array(obs_wp)
            self.agents_full_path[i] = p[:k+1]

        return self.finalize_all_agents()

    def finalize_all_agents(self):
        """
        Aggregates measurements from all agents and computes the final predicted spatial field.
        """
        aggregated_obs_wp = np.vstack(self.agents_obs_wp)
        aggregated_paths = np.vstack(self.agents_full_path)
        self.full_path = aggregated_paths.T
        self.obs_wp = aggregated_obs_wp
        measurements = self.scenario.simulate_measurements(aggregated_obs_wp)
        return self.scenario.predict_spatial_field(aggregated_obs_wp, measurements)





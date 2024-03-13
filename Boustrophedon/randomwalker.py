import numpy as np

class RandomWalker:
    def __init__(self, workspace_size=(40, 40), d_waypoint_distance=2.5, n_waypoints=200):
        self.workspace_size = workspace_size
        self.d_waypoint_distance = d_waypoint_distance
        self.n_waypoints = n_waypoints
        self.full_path = None  # This will store the direct path between waypoints
        self.obs_wp = None  # This stores the observation waypoints
        self.name = "RandomWalker"
        self.generate_random_walk()

    def generate_random_walk(self):
        current_position = np.array([0, 0])
        waypoints = [current_position]

        for _ in range(1, self.n_waypoints-1):
            theta = np.random.uniform(0, 2 * np.pi)
            next_position = current_position + self.d_waypoint_distance * np.array([np.cos(theta), np.sin(theta)])
            
            if 0 <= next_position[0] <= self.workspace_size[0] and 0 <= next_position[1] <= self.workspace_size[1]:
                waypoints.append(next_position)
                current_position = next_position
            else:
                continue  # Skip this waypoint if it leads outside the workspace
        # return to the starting point
        waypoints.append(waypoints[0])
        self.obs_wp = np.array(waypoints).reshape(-1, 2)
        # For RandomWalker, full_path is essentially the same as obs_wp since we're connecting waypoints with straight lines
        self.full_path = np.array(waypoints).reshape(-1, 2).T
        # print(f"Generated {self.n_waypoints} waypoints for {self.name}.")
        # print(f'Full path: {self.full_path}')

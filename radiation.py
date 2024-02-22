import numpy as np

class RadiationField:
    def __init__(self, num_sources=1, workspace_size=(40, 40), intensity_range=(10000, 100000)):
        self.sources = self.generate_sources(num_sources, workspace_size, intensity_range)
        self.r_s = 0.5  # Source radius
        self.r_d = 0.5  # Detector radius
        self.T = 100  # Transmission factor

    def generate_sources(self, num_sources, workspace_size, intensity_range):
        """Generate random sources within the workspace."""
        sources = []
        for _ in range(num_sources):
            rand_x = np.random.uniform(0, workspace_size[0])
            rand_y = np.random.uniform(0, workspace_size[1])
            rand_A = np.random.uniform(*intensity_range)
            sources.append([rand_x, rand_y, rand_A])
        return sources

    def update_source(self, source_index, new_x, new_y, new_A):
        """Update a specific source's parameters."""
        if source_index < len(self.sources):
            self.sources[source_index] = [new_x, new_y, new_A]
        else:
            print("Source index out of range.")

    def get_sources_info(self):
        """Return information about the current sources."""
        return self.sources

    def intensity(self, r):
        """Compute intensity at a point r."""
        I = 0
        for source in self.sources:
            r_n, A_n = np.array(source[:2]), source[2]
            dist = np.linalg.norm(r - r_n)
            if dist <= self.r_s:
                I += A_n / (4 * np.pi * self.r_s**2)
            else:
                I += A_n * self.T / (4 * np.pi * dist**2)
        return I

    def response(self, r):
        """Compute response at a point r."""
        R = 0
        for source in self.sources:
            r_n, A_n = np.array(source[:2]), source[2]
            dist = np.linalg.norm(r - r_n)
            if dist <= self.r_d:
                R += 0.5 * A_n
            else:
                theta = np.arcsin(self.r_d / dist)
                R += 0.5 * A_n * (1 - np.cos(theta))
        return R

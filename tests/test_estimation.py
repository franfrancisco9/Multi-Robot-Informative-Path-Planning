import sys
import os
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.point_source.point_source import PointSourceField
from src.estimation.estimation import estimate_sources_bayesian

def test_estimation_with_perfect_values():
    # Define a scenario with known sources
    num_sources = 2
    workspace_size = (0, 40, 0, 40)
    intensity_range = (10000, 1000000)
    known_sources = [
        [5, 5, 100000],
        [30, 30, 100000]
    ]

    # Create a PointSourceField with the known sources
    scenario = PointSourceField(num_sources=num_sources, workspace_size=workspace_size, intensity_range=intensity_range)
    for i, source in enumerate(known_sources):
        scenario.update_source(i, *source)

    # Simulate observations at various waypoints horizontal lines spaces 0.5
    waypoints = [(x, y) for x in np.arange(workspace_size[0], workspace_size[1], 1.0) for y in np.arange(workspace_size[2], workspace_size[3], 1.0)]
    print(f"Simulating measurements at {len(waypoints)} waypoints...")
    measurements = scenario.simulate_measurements(waypoints, noise_level=0.5)  # No noise for perfect values

    # Run the estimation 10 times and collect the number of estimated sources
    num_runs = 10
    estimated_sources_counts = []

    for _ in range(num_runs):
        # Estimate the sources using the Bayesian estimation method
        estimated_sources, estimated_num_sources, estimated_bic = estimate_sources_bayesian(
            obs_wp=np.array(waypoints),
            obs_vals=np.array(measurements),
            lambda_b=1.0,  # Assume a background radiation rate of 1.0 for simplicity
            max_sources=3,  # Allow the estimator to consider up to 5 sources
            n_samples=2000,  # Number of samples for importance sampling
            s_stages=25,  # Number of stages for progressive correction
            scenario=scenario
        )
        print(f"Estimated {estimated_num_sources} sources with BIC: {estimated_bic}, {estimated_sources}")
        estimated_sources_counts.append(estimated_num_sources)

    # Calculate and print the average number of estimated sources
    avg_estimated_sources = np.mean(estimated_sources_counts)

    print("Known sources:")
    for source in known_sources:
        print(f"Location: ({source[0]}, {source[1]}), Intensity: {source[2]}")

    print("\nEstimated sources (last run):")
    estimated_sources = estimated_sources.reshape(-1, 3)
    for source in estimated_sources:
        print(f"Location: ({source[0]}, {source[1]}), Intensity: {source[2]}")

    print("\nAverage number of estimated sources over 10 runs:", avg_estimated_sources)

if __name__ == "__main__":
    test_estimation_with_perfect_values()

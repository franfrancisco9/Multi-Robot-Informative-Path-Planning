# Multi-Robot Informative Path Planning
 Code for thesis on Multi-Robot Informative Path Planning for mapping unkown spatial fields, with a focus on managing exploration versus exploitation.


# Path Planning Strategies

This project explores various path planning strategies to navigate scenarios within a budget, balancing exploration and exploitation based on the scenario's needs. It includes implementations of Boustrophedon, Informative Path Planning, Naive RRT, Strategic RRT, Bias RRT, Bias Beta RRT, and Adaptive RRT.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.6+
- requeriments.txt

### Installing

1. Clone the repository to your local machine.
2. Ensure you have Python 3.6+ installed.
3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Running the code


Running the code
To run the simulations, use the following command with the necessary arguments:

```bash
python main.py -config example_setup.json
```

Where `example_setup.json` should be replaced with the path to your JSON configuration file. The JSON configuration allows you to easily specify the scenarios and strategies you want to test, along with any other parameters for the simulation. Check the example JSON configuration file for more information on how to set up your own.

Key Components of the Configuration:

- Scenarios: Define the simulation scenarios. Each scenario has params for general parameters like number of sources and workspace size. specific_params allows you to specify the location and strength of specific sources.
- Strategies: List of strategy class names you wish to test. Ensure the strategy names match the class names in the code.
- Args: General arguments for the simulation such as the number of rounds, beta value for trade-off, whether to save/show the results, seed for randomness, budget for path planning, waypoint distance, and budget iterations for RRT strategies:

    - `rounds`: Number of rounds to run the simulation.
    - `beta_t`: Beta value for the Bias Beta RRT strategy.
    - `save`: Flag to save the results to a file.
    - `show`: Flag to show the results in a plot.
    - `seed`: Seed for random number generation.
    - `budget`: Budget for path planning.
    - `d_waypoint_distance`: Waypoint distance.
    - `budget_iter`: Budget iterations for RRT strategies.
    - `run_number`: Number of the run to control the folder name for saving the results.

## Note

This code is in active development and may change frequently. If you have any questions or suggestions, please feel free to reach out.



[![CodeFactor](https://www.codefactor.io/repository/github/franfrancisco9/multi-robot-informative-path-planning/badge)](https://www.codefactor.io/repository/github/franfrancisco9/multi-robot-informative-path-planning)

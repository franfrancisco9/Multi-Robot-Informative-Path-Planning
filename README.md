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

```bash
python main.py -r [NUMBER_OF_ROUNDS] -beta [BETA_T_VALUE] -save -show -seed [SEED] -budget [BUDGET] -d [WAYPOINT_DISTANCE] -budget-iter [BUDGET_ITER]
```

- `NUMBER_OF_ROUNDS`: Number of rounds to run the simulation.
- `BETA_T_VALUE`: Beta value for the Bias Beta RRT strategy.
- `-save`: Flag to save the results to a file.
- `-show`: Flag to show the results in a plot.
- `-seed`: Seed for random number generation.
- `-budget`: Budget for path planning.
- `-d`: Waypoint distance.
- `-budget_iter`: Budget iterations for RRT strategies.

## Note

This code is in active development and may change frequently. If you have any questions or suggestions, please feel free to reach out.





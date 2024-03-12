import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from radiation import RadiationField


# Scenario 1 - Empty field 
scenario_1 = RadiationField(num_sources=0)
# Scenario 2 - Single source
scenario_2 = RadiationField(num_sources=1)
# Scenario 3 - Two sources
scenario_3 = RadiationField(num_sources=2)
# Scenario 4 - Seven sources
scenario_4 = RadiationField(num_sources=7)

# Define the spatial domain for visualization
Z_true_1 = scenario_1.ground_truth()
Z_true_2 = scenario_2.ground_truth()
Z_true_3 = scenario_3.ground_truth()
Z_true_4 = scenario_4.ground_truth()

# For each scneario create a figure, get the max_log_value
# Create the levels and the colormap
# save the fig to ./images with scenario_i.png
# show the fig
for i in range(1, 5):
    Z_true = globals()[f'Z_true_{i}']
    fig, axs = plt.subplots(1, 1, figsize=(16, 6))
    if Z_true.max() == 0:
        max_log_value = 1
    else:
        max_log_value = np.ceil(np.log10(Z_true.max()))
    levels = np.logspace(0, max_log_value, int(max_log_value) + 1)
    cmap = plt.get_cmap('Greens_r', len(levels) - 1)
    cs_true = axs.contourf(scenario_1.X, scenario_1.Y, Z_true, 
                          levels=levels, cmap=cmap, 
                          norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
    fig.colorbar(cs_true, ax=axs, format=ticker.LogFormatterMathtext())
    axs.set_title(f'Scenario {i} Radiation Field')
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    plt.tight_layout()
    plt.savefig(f'./images/scenario_{i}.png')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from radiation import intensity, response
from cma import CMAEvolutionStrategy as CMAES



# Covariance Matrix Adaption Evolution Strate
# Define the parameters of the radiation field
N = 1  # Number of sources
# random positions all with random strenghts between 100000 and 1000000
sources = []
for i in range(N):
    rand_x = np.random.uniform(0, 40)
    rand_y = np.random.uniform(0, 40)
    rand_A = np.random.uniform(10000, 100000)
    # rand_x = 20
    # rand_y = 20
    # rand_A = 100000
    sources.append([rand_x, rand_y, rand_A])

r_s = 0.5  # Source radius
r_d = 0.5  # Detector radius
T = 100  # Transmission factor

# Define the spatial domain
x = np.linspace(0, 40, 200)
y = np.linspace(0, 40, 200)
X, Y = np.meshgrid(x, y)

# Compute the radiation field
Z = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        r = np.array([X[i, j], Y[i, j]])
        Z[i, j] = intensity(r, sources, r_s, T) + 50 * response(r, sources, r_d)

# Mask negative values if any
Z = np.ma.masked_where(Z <= 0, Z)

# Define manual levels for the contour plot to cover each order of magnitude from 10^0
max_log_value = np.ceil(np.log10(Z.max()))
levels = np.logspace(0, max_log_value, int(max_log_value) + 1)

# Define a colormap with distinct shades of green for each order of magnitude
cmap = plt.get_cmap('Greens_r', len(levels) - 1)  # Ensuring we get a discrete range of greens

# Plotting
fig, ax = plt.subplots()
# Use the custom levels and colormap, with normalization over the specified levels
cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True))
cbar = fig.colorbar(cs, ticks=levels, format=ticker.LogFormatterMathtext())
cbar.ax.set_yticklabels([f'$10^{{{int(np.log10(l))}}}$' for l in levels])

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Radiation Field with Discrete Color Scale')
plt.show()

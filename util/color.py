from matplotlib.colors import LinearSegmentedColormap
import numpy as np

colors = []
for j in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255,j))
for j in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,j))

red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)
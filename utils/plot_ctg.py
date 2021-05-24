import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

def plot(ctg, name):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    assert len(ctg.shape) == 2
    dimX = ctg.shape[0]
    dimY = ctg.shape[1]
    X = np.arange(-dimX // 2, dimX // 2, 1)
    Y = np.arange(-dimY // 2, dimY // 2, 1)
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, ctg, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-.1, np.max(ctg) + .1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.0f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    #plt.show()
    fig.savefig(f'outputs/figures/{name}_ctg_plot.png')

if __name__=="__main__":
    name = sys.argv[1]
    with open(f"outputs/precompute/{name}_ctg.npy", "rb") as f:
        ctg = np.load(f)
    plot(ctg, name)

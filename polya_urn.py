import random
from matplotlib import pyplot, mpl
from collections import Counter

"""
Polya urn model:
    1) generate colors Θ_1, ..., Θ_N ~ Polya(G_0, α)
    2) generate each datapoint p_i ~ F(Θ_i)
"""

def gen_colors(): # G_0
    """ return a random [0..1] RGB tuple """
    return (random.random(), random.random(), random.random())


def polya_urn(colors_distribution, N, alpha): # 1)
    """
    Parameters:
     - colors_distribution, the distrib which we sample new colors from
     - N, the number of balls in total (in the end)
     - alpha, dispersion parameter
    Returns the list of balls in the urn.
    """

    if N <= 0:
        return []
    balls = []

    for b in range(1, N+1):
        if random.random() <= alpha * 1.0 / (alpha + len(balls)):
            balls.append(colors_distribution())
        else:
            balls.append(balls[random.randint(0, len(balls)-1)])
    return balls


def color_bar(urn):
    """ Outputs a bar-coloring of the urn distribution in file polya.png """
    fig = pyplot.figure(figsize=(8,2))
    c = Counter(urn)
    cmap = mpl.colors.ListedColormap(c.keys())
    bounds = [-sum(c.values())/2]
    for v in c.itervalues():
        bounds.append(bounds[len(bounds)-1] + v)
    print c
    print bounds
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb = mpl.colorbar.ColorbarBase(fig.add_axes([0.05, 0.5, 0.9, 0.15]), 
            cmap=cmap,
            norm=norm,
            boundaries=bounds,
            ticks=bounds,
            spacing='proportional',
            orientation='horizontal')
    cb.set_label("Distribution of balls' colors in the Polya-urn")
    pyplot.savefig("polya.png")


if __name__ == "__main__":
    color_bar(polya_urn(gen_colors, 20, 2.0))


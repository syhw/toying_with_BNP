# -*- coding: utf-8 -*-
import random
from matplotlib import pyplot, mpl
from collections import Counter

"""
Polya urn:
    1) place α black balls in an urn
    2) repeat N times:
        - draw for the urn, in any case put back this ball:
            - if the ball if black, place one ball of a new color in the urn
            - else put one ball of the same color in the urn
    Notes :
        - we can place more than one ball in the urn, in both cases.
        - we can start with a non-empty (non-black balls) urn (as we could 
          place customers in the CRP instead of starting empty).
        - we can specify the colors distribution of "new color" balls (prior)
    
Polya urn mixture model (== DPMM): TODO
    1) generate colors Θ_1, ..., Θ_N ~ Polya(G_0, α)
    2) generate each datapoint y_i ~ F(Θ_i)
"""

def uniform_colors():
    """ return a random [0..1] RGB tuple """
    return (random.random(), random.random(), random.random())


def polya_urn(colors_distribution, N, alpha, init=[]): # 1)
    """
    Parameters:
     - colors_distribution, the distrib which we sample new colors from
     - N, the number of balls in total (in the end)
     - alpha, dispersion parameter
     - optionally can be initialized with balls already inside
    Returns the list of balls in the urn.
    """

    if N <= 0:
        return init
    balls = init

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
    color_bar(polya_urn(uniform_colors, 20, 2.0))
    c1 = uniform_colors()
    c2 = uniform_colors()
    color_bar(polya_urn(uniform_colors, 30, 2.0, 
        init=[c1,c1,c1,c1,c1,c1,c2,c2,c2,c2]))


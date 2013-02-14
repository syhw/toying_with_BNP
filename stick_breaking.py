# -*- coding: utf-8 -*-
import random
from scipy.stats import beta
from crp import histogram

"""
Stick-breaking process:
    broken pieces = weights = β
    β_k = β'_k \prod_{i=1}^{k-1} (1 - β'_i), with β'_k ~ Beta(1, α)
    i.e. 
    1) Take a stick of length 1 (unit)
    2) Repeat:
        - Break the remaining part at a proportion sampled from Beta(1, α)
"""

def stick_breaking_process(num_weights, alpha): # 1)
    """
    Parameters:
     - num_weights, the number of stick pieces, i.e. the number of different 
     tables in the CRP, or the number of colors in the Polya urn.
     - alpha, the dispersion parameter
    Returns a vector of weights, i.e. a (fractional) number of customers per 
    tables for the CRP, of a (fractional) number of balls of each colors for
    the Polya urn: weights[indice] = proportion_of_indice
    """

    if num_weights <= 0:
        return []
    
    betas = beta.rvs(1, alpha, size=num_weights)
    remaining_stick = 1.0
    weights = []
    for b in betas: # could use numpy.cumprod instead
        piece = remaining_stick * b
        weights.append(piece)
        remaining_stick -= piece
    return weights


if __name__ == "__main__":
    weights = stick_breaking_process(20, 3.0)
    print weights
    histogram(weights, 
            plot_name="stick_breaking",
            plot_xlabel="stick length (from unit stick)",
            plot_title="Stick-breaking process distribution")


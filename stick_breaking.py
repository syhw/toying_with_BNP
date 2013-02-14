# -*- coding: utf-8 -*-
import random
from scipy.stats import beta
from crp import histogram

"""
Stick-breaking process:
    1) generate group probabilities (stick lengths) w_1, ..., w_∞ ~ Stick(α)
    2) generate group parameters θ_1, ..., θ_N ~ G_0 [base distribution]
    3) generate group assignments g_1, ..., g_N ~ Categorical(w_1, ...,w_∞)
    4) generate each datapoint p_i ~ F(θ_i)
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


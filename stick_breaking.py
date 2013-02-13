import random
from scipy.stats import beta
from crp import histogram


def stick_breaking_process(num_weights, alpha):
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


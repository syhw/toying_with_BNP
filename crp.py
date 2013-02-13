import random
from collections import Counter

"""
Chinese-restaurant process:
    - generate table assignments g_1, ..., g_N ~ CRP(N, alpha)
    - generate table parameters theta_1, ..., theta_N ~ G_0 [base distribution]
    - generate each datapoint p_i ~ F(theta_{g_i})
    for instance F is a Gaussian and theta_i = (mean_i, var_i)
"""

def chinese_restaurant_process(N, alpha):
    """ 
    Parameters:
     - N, number of customers
     - alpha, dispersion parameter
    Returns a list of customers (in order of arrival) and their tables:
    tables[customer_indice] = table_indice 
    """

    if N <= 0:
        return []
    tables = [1] # first customer at table 1
    next_open = 2

    for c in range(1, N+1):
        if random.random() < alpha * 1.0 / (alpha + c):
            tables.append(next_open)
            next_open += 1
        else:
            tables.append(tables[random.randint(0, len(tables)-1)])
    return tables


def histogram(c, plot_name="test", plot_title="", plot_xlabel=""):
    import pylab
    pylab.figure(1)
    pos = pylab.arange(len(c))+.5
    pylab.barh(pos, c, align='center')
    pylab.yticks(pos, range(1, len(c)+1))
    pylab.xlabel(plot_xlabel)
    pylab.title(plot_title)
    pylab.grid(True)
    pylab.savefig(plot_name+".png")


if __name__ == "__main__":
    tables = chinese_restaurant_process(100, 3.0)
    print tables
    c = Counter(tables)
    histogram(c.values(), 
            plot_name="crp", 
            plot_xlabel="number of customers",
            plot_title="Chinese restaurant process tables distribution")

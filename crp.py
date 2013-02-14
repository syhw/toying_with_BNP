# -*- coding: utf-8 -*-
import random
from collections import Counter

"""
Chinese-restaurant process:
    tables = clusters = c
    dishes = parameters = Φ
    customers = datapoints = y
    
     - initially the restaurant is empty
     - the first person to enter sits down at a table (selects a cluster) and
       orders food for the table (parameters of the cluster)
     - the second person to enter:
        - with probability α/(1+α) sits down at a new table
        - with probability 1/(1+α) sits down with the first customer
     - the (n+1)th person sits down:
        - with probability α/(n+α) at a new table 
        - with probability n_k/(n+α) at a the kth table (with n_k persons)
"""

def chinese_restaurant_process(N, alpha): # 1)
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

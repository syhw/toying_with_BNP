import random
import pylab
from collections import Counter


def chinese_restaurant_process(N, alpha):
    """ 
    Parameters:
     - N, number of customers
     - alpha, dispersion parameter
    returns a list of customers (in order of arrival) and their tables:
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


if __name__ == "__main__":
    tables = chinese_restaurant_process(100, 3.0)
    print tables
    c = Counter(tables)
    pylab.figure(1)
    pos = pylab.arange(len(c))+.5
    pylab.barh(pos, c.values(), align='center')
    pylab.yticks(pos, range(1, len(c)+1))
    pylab.xlabel('number of customers')
    pylab.title('Chinese restaurant process tables distribution')
    pylab.grid(True)
    pylab.savefig("crp.png")

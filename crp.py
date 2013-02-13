import random

def chinese_restaurant_process(N, alpha):
    """ returns a list of customers (in order of arrival) and their tables:
    tables[customer_indice] = table_indice """

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
    print chinese_restaurant_process(100, 3.0)

import random

def gen_colors():
    """ return a random [0..1] RGB tuple """
    return (random.random(), random.random(), random.random())


def polya_urn(colors_distribution, N, alpha):
    """
    Parameters:
     - colors_distribution, the distrib which we sample new colors from
     - N, the number of balls in total (in the end)
     - alpha, dispersion parameter
    returns the list of balls in the urn
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


if __name__ == "__main__":
    print polya_urn(gen_colors, 20, 3.0)

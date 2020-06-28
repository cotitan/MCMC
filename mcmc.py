import numpy as np
import random
import matplotlib.pyplot as plt

def continuous():
    """ Metropolis-Hastings for continuous distribution
    """
    def target_pdf(x):
        mean = 3
        sigma = 2
        return np.exp(-(x-mean)**2/(2*sigma**2))\
            / (np.sqrt(2*np.pi) * sigma)

    T = 5000
    xs = [0]
    sigma = 1
    for _ in range(T):
        next_x = np.random.normal(loc=xs[-1], scale=sigma)
        # there is no transition matrix in this case
        # intuitively, if next is closer to mean_value compare to x,
        # then next_x is tend to be accepted
        alpha = min(1, (target_pdf(next_x) / target_pdf(xs[-1])))

        u = random.uniform(0, 1)
        if u < alpha:
            xs.append(next_x)
        else:
            xs.append(xs[-1])

    xs = np.array(xs)
    plt.scatter(xs, target_pdf(xs))
    num_bins = 50
    plt.hist(xs, num_bins, normed=1, facecolor='red', alpha=0.7)
    plt.show()

def discrete():
    """ Metropolis-Hastings for descrete distribution
    """
    dim = 5
    # target distribution
    pi = np.random.rand(dim)
    pi /= pi.sum()
    # markov transition matrix
    P = np.random.rand(dim, dim)
    P = P / P.sum(axis=1, keepdims=True)

    record = np.zeros(dim)
    # init state
    x = np.random.choice(list(range(dim)), p=pi)

    for i in range(100000):
        next_x = np.random.choice(list(range(dim)), p=P[x,:])
        # MCMC
        alpha = pi[next_x] * P[next_x][x]
        # Metropolis-Hasting. M-H sampling
        # alpha = min([pi[next_x]*P[next_x][x] / (pi[x]*P[x][next_x]), 1])
        u = np.random.rand(1)
        if u < alpha:
            x = next_x
        record[x] += 1
    print(pi)
    print(record / record.sum())
    print(record)

if __name__ == "__main__":
    discrete()
    continuous()

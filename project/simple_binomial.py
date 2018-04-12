"""Simple ABC example with a binomial model.

   Comparison of MCMC-ABC, conventional MH MCMC and an anlytical solution.

   Copyright (c) 2018, Felix Held
"""
import numpy as np
import scipy.stats as stats

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns


def logistic(theta):
    """Logistic function."""
    return 1. / (1. + np.exp(-theta))


def loglogistic(theta):
    """Log of the logistic function."""
    return -np.log(1. + np.exp(-theta))


def mcmc_abc(data, M=10000, sigma=1e-1, eps=1e-1):
    """Implementation of MCMC-ABC for the particular problem.

    Beta(2, 2) prior is assumed for p. Parameters are inferred on
    an unbounded parameter space.

    Arguments:
        data - data from experiment
        M - number of iterations
        sigma - random walk standard deviation
        eps - ABC epsilon

    Returns:
        ps - samples for p
    """
    ps = np.zeros((M + 1,), dtype='float64')

    # Summary statistics from data
    D = np.array([np.mean(data), np.std(data)])

    # Transform to unbounded space
    ps[0] = -np.log(1. / stats.beta(2, 2).rvs() - 1.)

    for m in tqdm(range(M)):
        while True:
            pnew = ps[m] + sigma * np.random.randn()

            data_sim = np.random.binomial(3, logistic(pnew), size=5)
            Dsim = np.array([np.mean(data_sim), np.std(data_sim)])

            if np.linalg.norm(D - Dsim) <= eps:
                break

        lacc = stats.beta(2, 2).logpdf(logistic(pnew)) + \
            loglogistic(pnew) + np.log(1 - logistic(pnew)) - \
            (stats.beta(2, 2).logpdf(logistic(ps[m])) + \
            loglogistic(ps[m]) + np.log(1 - logistic(ps[m])))
        # Metropolis-Hastings acceptance step
        h = min(1, np.exp(lacc))
        if np.random.rand() <= h:
            ps[m + 1] = pnew
        else:
            ps[m + 1] = ps[m]

    return logistic(ps)


def mh(data, M=10000, sigma=1e-1):
    """Implementation of Metropolis-Hastings for the particular problem.

    Beta(2, 2) prior is assumed for p. Parameters are inferred on
    an unbounded parameter space.

    Arguments:
        data - data from the experiment
        M - number of iterations
        sigma - random walk std dev

    Returns:
        ps - samples for p
    """
    ps = np.zeros((M + 1,), dtype='float64')

    # Transform to unbounded space
    ps[0] = -np.log(1. / stats.beta(2, 2).rvs() - 1.)

    for m in tqdm(range(M)):
        pnew = ps[m] + sigma * np.random.randn()

        lacc = np.sum(stats.binom.logpmf(data, 3, logistic(pnew))) + \
            stats.beta(2, 2).logpdf(logistic(pnew)) + \
            loglogistic(pnew) + np.log(1 - logistic(pnew)) - \
            (np.sum(stats.binom.logpmf(data, 3, logistic(ps[m]))) + \
             stats.beta(2, 2).logpdf(logistic(ps[m])) + \
             loglogistic(ps[m]) + np.log(1 - logistic(ps[m])))

        # Metropolis-Hastings acceptance step
        h = min(1, np.exp(lacc))
        if np.random.rand() <= h:
            ps[m + 1] = pnew
        else:
            ps[m + 1] = ps[m]

    return logistic(ps)


def ep_abc(data, M=10000, eps=1e-1):
    """Implementation of EP-ABC for the particular problem.

    Very naive implementation of EP-ABC for one parameter.

    Parameters:
        data - data from the experiment
        M - number of samples for moment matching
        eps - ABC epsilon
    """



if __name__ == '__main__':
    sns.set_style()

    # Simulate data
    data = np.random.binomial(3, 0.3, size=5)
    print(data)

    # Infere parameters
    ps_abc = mcmc_abc(data)
    ps_mh = mh(data)

    # Exact solution with conjugate prior
    xs = np.arange(0., 1., 1e-2)
    ys = stats.beta(2 + np.sum(data), 2 + np.sum(3 - data)).pdf(xs)

    # Kernel density estimates for the samples
    kde_abc = stats.gaussian_kde(ps_abc, 'silverman')
    kde_mh = stats.gaussian_kde(ps_mh, 'silverman')

    # Plot solutions
    plt.plot(xs, ys)
    plt.plot(xs, kde_abc(xs))
    plt.plot(xs, kde_mh(xs))
    # plt.hist(logistic(ps_abc), bins=30, density=True, alpha=0.4, color='b');
    # plt.hist(logistic(ps_mh), bins=30, density=True, alpha=0.4, color='r');

    plt.savefig('p_densities.png')

    # Save traces
    plt.clf()
    plt.plot(ps_abc)
    plt.savefig('abc_trace.png')

    plt.clf()
    plt.plot(ps_mh)
    plt.savefig('mh_trace.png')

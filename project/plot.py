"""Plot the resulting densities from stable_dist.

   Copyright (c) 2018, Felix Held
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import scipy.stats as stats

from stable_dist import stable_rvs
from binomial import logistic, logit


def histogram_binomial():
    """Plot histogram of the data in the binomial case."""
    with h5py.File('simple_binomial.h5', 'r') as f:
        data = f['run5/data'][...]

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.hist(data, bins=11, range=(-0.5, 10.5), density=True)
    ax.set_xlabel('Binomial outcome')
    ax.set_ylabel('Frequency')

    fig.tight_layout()
    fig.savefig('figures/histogram_binomial.png')


def traces_binomial():
    """Plot traces for Metropolis Hastings and MCMC-ABC."""
    with h5py.File('simple_binomial.h5', 'r') as f:
        ps_abc = f['run5/abc'][...]
        ps_mh = f['run5/abc'][...]

    fig, ax = plt.subplots(figsize=(4, 3))

    # Save traces
    ax.plot(ps_abc)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Probability $p$')
    fig.tight_layout()
    fig.savefig('figures/mcmc_abc_trace.png')

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(ps_mh)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Probability $p$')
    fig.tight_layout()
    fig.savefig('figures/mh_trace.png')


def density_binomial():
    """Plot histogram of the data in the binomial case."""
    with h5py.File('simple_binomial.h5', 'r') as f:
        data = f['run5/data'][...]

        # EP results
        ep = f['run5/ep'][...]
        mu = ep[0]
        sigma = ep[1]

        # Markov chains
        ps_abc = f['run5/abc'][...]
        ps_mh = f['run5/abc'][...]

    # Problem setup
    n = 10

    fig, ax = plt.subplots(figsize=(5, 3))

    # Kernel density estimates for the samples (2000 samples burnin)
    kde_abc = stats.gaussian_kde(logistic(ps_abc[2000:]), 'silverman')
    kde_mh = stats.gaussian_kde(logistic(ps_mh[2000:]), 'silverman')

    # Zoomed in solution
    fig.clf()
    ax = fig.add_subplot(111)
    # Exact solution with conjugate prior
    xs = np.arange(0.285, 0.325, 1e-4)
    ys = stats.beta(1 + np.sum(data), 1 + np.sum(n - data)).pdf(xs)

    # Solution from expectation propagation
    ys_ep = stats.norm(mu, sigma).pdf(logit(xs)) / (xs * (1 - xs))

    ax.plot(xs, kde_abc(xs))
    ax.plot(xs, kde_mh(xs), '--')
    ax.plot(xs, ys_ep, '-.')
    ax.plot(xs, ys, '--')

    ax.legend(['MCMC-ABC', 'MH', 'EP-ABC', 'Analytic'])
    ax.set_xlabel('Probability $p$')
    ax.set_ylabel('Density')

    fig.tight_layout()
    fig.savefig('figures/p_densities_zoomed.png')


def data_stable():
    """Plot data used for estimation of stable distribution."""
    # Import return data
    zt = pd.read_csv('euro_in_sterling.csv').zt.values
    # Calculate scaled log-return rates
    yt = 100. * np.log(zt[1:] / zt[:-1])

    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    axs[0].plot(zt, 'o-', markersize=2, linewidth=1)
    axs[0].set_xlabel('Day')
    axs[0].set_ylabel('Exchange rate')

    axs[1].plot(yt, 'o-', markersize=2, linewidth=1)
    axs[1].set_xlabel('Day')
    axs[1].set_ylabel('Scaled log-return rate')

    fig.tight_layout()
    fig.savefig('figures/data_stable.png')


def density_stable():
    """Plot the marginal densities for the estimated parameters."""
    # Reload data saved as below
    # np.savez('stable_dist_eur_in_sterling', yt, mu, sigma, total_samples)
    data = np.load('stable_dist_eur_in_sterling.npz')
    mu = data['arr_1']
    sigma = data['arr_2']

    fig, axs = plt.subplots(2, 2, figsize=(9, 5))

    # alpha
    xs = np.arange(0.29, 0.4, 1e-4)
    axs[0, 0].plot(xs, stats.norm.pdf(
        stats.norm.ppf(xs / 2), mu[0], np.sqrt(sigma[0, 0])) /
        (stats.norm.pdf(stats.norm.ppf(xs / 2)) * 2))
    axs[0, 0].set_title('$\\alpha$')

    # beta
    xs = np.arange(-0.5, -0.35, 1e-4)
    axs[0, 1].plot(xs, stats.norm.pdf(
        stats.norm.ppf((xs + 1) / 2), mu[1], np.sqrt(sigma[1, 1])) /
        (stats.norm.pdf(stats.norm.ppf((xs + 1) / 2)) * 2))
    axs[0, 1].set_title('$\\beta$')

    # mu
    xs = np.arange(0.05, .125, 1e-4)
    axs[1, 0].plot(xs, stats.norm.pdf(xs, mu[2], np.sqrt(sigma[2, 2])))
    axs[1, 0].set_title('$\\mu$')

    # c
    xs = np.arange(3.31e-3, 3.38e-3, 1e-6)
    axs[1, 1].plot(xs, stats.norm.pdf(
        np.log(xs), mu[3], np.sqrt(sigma[3, 3])) / xs)
    axs[1, 1].set_title('$c$')

    fig.tight_layout()
    fig.savefig('figures/stable_dist_densities_euro_in_sterling.png')


def histograms_stable():
    """Plot histograms of data and simulated values at MAP estimate.

    The MLE solution was calculated with help of the
    R package `StableEstim`.
    """
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))

    # Reload data saved as below
    # np.savez('stable_dist_eur_in_sterling', yt, mu, sigma, total_samples)
    data = np.load('stable_dist_eur_in_sterling.npz')
    yt = data['arr_0']
    mu = data['arr_1']

    # MAP estimate from EP-ABC
    alpha = 2 * stats.norm.cdf(mu[0])
    beta = 2 * stats.norm.cdf(mu[1]) - 1
    loc = mu[2]
    scale = np.exp(mu[3])

    smpls = stable_rvs(alpha, beta, loc, scale, size=int(1e7))

    # MLE solution from R
    alpha = 1.9227
    beta = 0.0581
    loc = 0.0111
    scale = 0.3389

    smpls_r = stable_rvs(alpha, beta, loc, scale, size=int(1e7))

    axs[0].hist(yt, density=True, alpha=0.3, bins=30)
    axs[0].hist(smpls, density=True, alpha=0.3, range=(-2, 2), bins=30)
    axs[0].set_xlabel('Scaled log-return')
    axs[0].set_ylabel('Frequency')
    axs[0].legend(['Data', 'Simulated'])

    axs[1].hist(yt, density=True, alpha=0.3, bins=30)
    axs[1].hist(smpls_r, density=True, alpha=0.3, range=(-2, 2), bins=30)
    axs[1].set_xlabel('Scaled log-return')
    axs[1].set_ylabel('Frequency')

    fig.tight_layout()
    fig.savefig('figures/histograms_euro_in_sterling.png')


if __name__ == '__main__':
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.serif'] = 'STIX Two Text'
    mpl.rcParams['font.family'] = 'serif'

    # # ####### Binomial distribution example
    # # Plot histogram
    # histogram_binomial()

    # # Plot MCMC traces
    # traces_binomial()

    # # Plot densities
    # density_binomial()

    # # ####### Stable distribution example
    # # Plot data
    # data_stable()

    # Plot density
    density_stable()

    # Plot histogram
    histograms_stable()

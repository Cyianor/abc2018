"""Plot the resulting densities from stable_dist.

   Copyright (c) 2018, Felix Held
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from stable_dist import stable_rvs

if __name__ == '__main__':
    # Import return data
    zt = pd.read_csv('euro_in_sterling.csv').zt.values
    # Calculate scaled log-return rates
    yt = 100. * np.log(zt[1:] / zt[:-1])

    # Save data from this run
    # np.savez('stable_dist_eur_in_sterling', yt, mu, sigma, total_samples)
    data = np.load('stable_dist_eur_in_sterling.npz')
    yt = data['arr_0']
    mu = data['arr_1']
    sigma = data['arr_2']
    total_samples = data['arr_3']

    print(mu, sigma, total_samples)

    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.serif'] = 'STIX Two Text'
    mpl.rcParams['font.family'] = 'serif'

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

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
    fig.savefig('stable_dist_densities_euro_in_sterling.png')

    fig, ax = plt.subplots(figsize=(12, 8))

    alpha = 2 * stats.norm.cdf(mu[0])
    beta = 2 * stats.norm.cdf(mu[1]) - 1
    loc = mu[2]
    scale = np.exp(mu[3])

    smpls = stable_rvs(alpha, beta, loc, scale, size=int(1e7))

    ax.hist(smpls, density=True, alpha=0.3, range=(-2, 2), bins=30)
    ax.hist(yt, density=True, alpha=0.3, bins=30)

    fig.tight_layout()
    fig.savefig('histograms_euro_in_sterling.png')


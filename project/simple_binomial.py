"""Simple ABC example with a binomial model.

   Comparison of MCMC-ABC, conventional MH MCMC and an analytical solution.

   Copyright (c) 2018, Felix Held
"""
import numpy as np
import scipy.stats as stats

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import h5py


def logit(theta):
    return np.log(theta / (1. - theta))


def logistic(theta):
    """Logistic function."""
    return 1. / (1. + np.exp(-theta))


def loglogistic(theta):
    """Log of the logistic function."""
    return -np.log(1. + np.exp(-theta))


def mcmc_abc(data, n, M=10000, sigma=1e-1, eps=1e-1):
    """Implementation of MCMC-ABC for the particular problem.

    Uniform prior (i.e. Beta(1, 1)) is assumed for p.
    Parameters are inferred on an unbounded parameter space.

    :Arguments:
        data : NumPy array
            Simulated data
        n : int
            Number of Bernoulli experiments
        M : int
            Number of MCMC iterations
        sigma : float
            Random walk std dev
        eps : float
            ABC epsilon

    :Returns:
        NumPy array : Samples for logit(p)
    """
    ps = np.zeros((M + 1,), dtype='float64')

    # Summary statistics from data
    D = np.array([np.mean(data), np.std(data)])

    # Initialize around correct answer, slightly cheating
    ps[0] = -0.9 * np.random.randn()

    total_samples = 0

    for m in tqdm(range(M)):
        while True:
            pnew = ps[m] + sigma * np.random.randn()

            data_sim = np.random.binomial(n, logistic(pnew), size=len(data))
            Dsim = np.array([np.mean(data_sim), np.std(data_sim)])
            total_samples += len(data)

            if np.linalg.norm(D - Dsim, ord=1) <= eps:
                break

        # Beta(1, 1) prior
        lacc = stats.beta(1, 1).logpdf(logistic(pnew)) + \
            loglogistic(pnew) + np.log(1 - logistic(pnew)) - \
            (stats.beta(1, 1).logpdf(logistic(ps[m])) +
             loglogistic(ps[m]) + np.log(1 - logistic(ps[m])))

        # Metropolis-Hastings acceptance step
        h = min(1, np.exp(lacc))
        if np.random.rand() <= h:
            ps[m + 1] = pnew
        else:
            ps[m + 1] = ps[m]

    return ps, total_samples


def mh(data, n, M=10000, sigma=1e-1):
    """Implementation of Metropolis-Hastings for the particular problem.

    Uniform prior (i.e. Beta(1, 1)) is assumed for p.
    Parameters are inferred on an unbounded parameter space.

    :Arguments:
        data : NumPy array
            Simulated data
        n : int
            Number of Bernoulli experiments
        M : int
            Number of MCMC iterations
        sigma : float
            Random walk std dev

    :Returns:
        NumPy array : Samples for logit(p)
    """
    ps = np.zeros((M + 1,), dtype='float64')

    # Initialize around correct answer, slightly cheating
    ps[0] = -0.9 * np.random.randn()

    for m in tqdm(range(M)):
        pnew = ps[m] + sigma * np.random.randn()

        # Beta(1, 1) prior
        lacc = np.sum(stats.binom.logpmf(data, n, logistic(pnew))) + \
            stats.beta(1, 1).logpdf(logistic(pnew)) + \
            loglogistic(pnew) + np.log(1 - logistic(pnew)) - \
            (np.sum(stats.binom.logpmf(data, n, logistic(ps[m]))) +
             stats.beta(1, 1).logpdf(logistic(ps[m])) +
             loglogistic(ps[m]) + np.log(1 - logistic(ps[m])))

        # Metropolis-Hastings acceptance step
        h = min(1, np.exp(lacc))
        if np.random.rand() <= h:
            ps[m + 1] = pnew
        else:
            ps[m + 1] = ps[m]

    return ps


def ep_abc(data, n, passes=3, M=10000, Mbatch=1000, eps=1e-1):
    """Implementation of EP-ABC for the particular problem.

    Very naive implementation of EP-ABC for one parameter.
    Normal(0, 2.5) prior is used on the unbounded scale, making
    the prior only weakly informative.

    :Arguments:
        data : NumPy array
            Simulated data
        n : int
            Number of Bernoulli experiments
        passes : int
            Number of passes through the dataset
        M : int
            Minimum number of accepted samples for moment matching
        Mbatch : int
            Number of samples created in every batch
        eps : float
            ABC epsilon

    :Returns:
        (float, float, int) : Normal approximation mu, sigma and total number
                              of simulated data
    """
    # Length of data
    N = len(data)

    # For each data point an approximating distribution N(r, q) is sought after
    # In the beginning assume r = 0 and q = 0 for all but prior distribution
    # Initial approximation is then q = f0, i.e. the prior

    # Save total number of sampled values
    total_samples = 0

    # Create arrays for precisions and precision means
    r = np.zeros((N + 1,), dtype='float64')
    q = np.zeros((N + 1,), dtype='float64')

    # Prior Normal(0, 2.5)
    r[0] = 0.           # Precision mean
    q[0] = 1. / 2.5**2  # Precision

    for k in tqdm(range(passes)):
        for i in tqdm(range(1, N + 1)):
            # Cavity distribution with respect to i-th data point
            r_cavity = np.sum(r[:i]) + np.sum(r[(i + 1):])
            q_cavity = np.sum(q[:i]) + np.sum(q[(i + 1):])

            if q_cavity <= 0.:
                # Only continue if positive precision
                print(
                    'Negative precision: Skipping site %d in pass %d' % (i, k))
                continue

            m_acc = 0
            theta_acc = np.array([], dtype='float64')
            while True:
                # ABC scheme to calculate new r and q
                theta = stats.norm(r_cavity / q_cavity,
                                   np.sqrt(1. / q_cavity)).rvs(Mbatch)
                sim_data = stats.binom.rvs(3, logistic(theta))

                acc_vec = (np.abs(sim_data - data[i - 1]) <= eps)
                m_acc += np.sum(acc_vec.astype('int'))

                theta_acc = np.concatenate((theta_acc, theta[acc_vec]))

                total_samples += 1

                if m_acc >= M:
                    break

            mu_tilted = np.sum(theta_acc) / m_acc
            var_tilted = np.sum(theta_acc**2) / m_acc - mu_tilted**2

            r[i] = mu_tilted / var_tilted - r_cavity
            q[i] = 1. / var_tilted - q_cavity

    return np.sum(r) / np.sum(q), np.sqrt(1. / np.sum(q)), total_samples


def ep_abc_iid(data, n, passes=3, M=10000, Mbatch=1000,
               ess_min=3000, eps=1e-1, a=1.):
    """Implementation of iid optimised EP-ABC for the particular problem.

    IID optimised implementation of EP-ABC for one parameter.
    Normal(0, 2.5) prior is used on the unbounded scale, making
    the prior only weakly informative.

    :Arguments:
        data : NumPy array
            Simulated data
        n : int
            Number of Bernoulli experiments
        passes : int
            Number of passes through the dataset
        M : int
            Minimum number of accepted samples for moment matching
        Mbatch : int
            Number of samples created in every batch
        ess_min : int
            Minimal allowed ESS value
        eps : float
            ABC epsilon
        a : float
            Determines how quickly the global approximation is updated

    :Returns:
        (float, float, int) : Normal approximation mu, sigma and total number
                              of simulated data
    """
    # Length of data
    N = len(data)

    # For each data point an approximating distribution N(r, q) is sought after
    # In the beginning assume r = 0 and q = 0 for all but prior distribution
    # Initial approximation is then q = f0, i.e. the prior

    # Save total number of sampled values
    total_samples = 0

    # Create arrays for precisions and precision means
    r = np.zeros((N + 1,), dtype='float64')
    q = np.zeros((N + 1,), dtype='float64')

    # Prior Normal(0, 2.5)
    r[0] = 0.           # Precision mean
    q[0] = 1. / 2.5**2  # Precision

    # Global approximation
    R = np.sum(r)
    Q = np.sum(q)

    def sample(mu_gen, sigma_gen, i):
        # Access total_samples from outer scope
        nonlocal total_samples

        theta = np.array([], dtype='float64')
        sim_data = np.array([], dtype='int')
        m_acc = 0
        while True:
            theta_tmp = stats.norm(mu_gen, sigma_gen).rvs(Mbatch)
            sim_data_tmp = stats.binom.rvs(n, logistic(theta_tmp))
            total_samples += Mbatch

            acc = (np.abs(sim_data_tmp - data[i]) < eps)
            m_acc += np.sum(acc.astype('int'))

            # Save "good" accepted samples
            theta = np.concatenate((theta, theta_tmp[acc]))
            sim_data = np.concatenate((sim_data, sim_data_tmp[acc]))

            if m_acc >= M:
                break

        return theta, sim_data, np.ones_like(theta)

    for k in tqdm(range(passes)):
        for i in tqdm(range(1, N + 1)):
            # Cavity distribution with respect to i-th data point
            r_cavity = R - r[i]
            q_cavity = Q - q[i]

            if q_cavity <= 0.:
                # Only continue if positive precision
                print(
                    'Negative precision: Skipping site %d in pass %d' % (i, k))
                continue
            else:
                sigma_cavity = np.sqrt(1. / q_cavity)
                mu_cavity = r_cavity / q_cavity

            if k == 0 and i == 1:
                # Set new generative mu and sigma
                mu_gen = mu_cavity
                sigma_gen = sigma_cavity

                # New samples
                theta, sim_data, ws = sample(mu_gen, sigma_gen, i - 1)
                lpdf = stats.norm(mu_gen, sigma_gen).logpdf(theta)
            else:
                # Calculate importance sampling weights
                ws = np.exp(
                    stats.norm(
                        mu_cavity, sigma_cavity).logpdf(theta) - lpdf) * \
                    (np.abs(sim_data - data[i - 1]) < eps)

            # Calculate effective sample size
            ws_sum = np.sum(ws)
            if ws_sum == 0. or (ws_sum**2 / np.sum(ws**2) < ess_min):
                # If ESS is too low then resample
                # Set new generative mu and sigma
                mu_gen = mu_cavity
                sigma_gen = sigma_cavity

                # New samples
                theta, sim_data, ws = sample(mu_gen, sigma_gen, i - 1)
                lpdf = stats.norm(mu_gen, sigma_gen).logpdf(theta)

            mu_tilted = np.average(theta, weights=ws)
            var_tilted = np.average((theta - mu_tilted)**2, weights=ws)

            q_tilted = 1. / var_tilted
            r_tilted = q_tilted * mu_tilted

            Q = a * q_tilted + (1 - a) * Q
            R = a * r_tilted + (1 - a) * R

            r[i] = R - r_cavity
            q[i] = Q - q_cavity

    return R / Q, np.sqrt(1. / Q), total_samples


if __name__ == '__main__':
    sns.set_style()

    # Simulate data
    n = 10
    p = 0.3
    data = np.random.binomial(n, p, size=1000)
    print('Data', data)
    print(np.mean(data), np.std(data))

    mu, sigma, ep_total_samples = ep_abc_iid(
        data, n=10, M=int(1e4), Mbatch=int(1e4),
        ess_min=int(2e4), eps=.1, passes=1)
    print('\nEP-ABC IID-opt', mu, sigma)
    print('\nEP-ABC total samples', ep_total_samples)

    # Infere parameters
    ps_abc, mcmc_total_samples = mcmc_abc(data, n, M=30000,
                                          sigma=5e-1, eps=.01)
    print('\nMCMC-ABC total samples', mcmc_total_samples)
    ps_mh = mh(data, n, M=30000)

    # Save data, EP-ABC estimate and MCMC-ABC as well as MH traces
    with h5py.File('simple_binomial.h5', 'a') as f:
        if 'last' in f.attrs:
            last = f.attrs['last']
            last += 1
            f.attrs['last'] = last
        else:
            last = 0
            f.attrs['last'] = last

        grp = f.create_group('run%d' % last)

        grp.attrs['p'] = p
        grp.attrs['ep-ttl-smpls'] = ep_total_samples
        grp.attrs['mcmc-ttl-smpls'] = mcmc_total_samples

        data_ds = grp.create_dataset(
            'data', data.shape, dtype='float64')
        data_ds[...] = data

        ep_ds = grp.create_dataset(
            'ep', (2,), dtype='float64')
        ep_ds[...] = [mu, sigma]

        abc_ds = grp.create_dataset(
            'abc', ps_abc.shape, dtype='float64')
        abc_ds[...] = ps_abc

        mh_ds = grp.create_dataset(
            'mh', ps_mh.shape, dtype='float64')
        mh_ds[...] = ps_mh

    # Kernel density estimates for the samples
    kde_abc = stats.gaussian_kde(logistic(ps_abc[2000:]), 'silverman')
    kde_mh = stats.gaussian_kde(logistic(ps_mh[2000:]), 'silverman')

    # Exact solution with conjugate prior
    xs = np.arange(0.01, 1., 1e-3)
    ys = stats.beta(1 + np.sum(data), 1 + np.sum(n - data)).pdf(xs)

    # Solution from expectation propagation
    ys_ep = stats.norm(mu, sigma).pdf(logit(xs)) / (xs * (1 - xs))

    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.serif'] = 'STIX Two Text'
    mpl.rcParams['font.family'] = 'serif'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Plot solutions
    ax.plot(xs, kde_abc(xs))
    ax.plot(xs, kde_mh(xs), '--')
    ax.plot(xs, ys_ep, '-.')
    ax.plot(xs, ys, ':')

    ax.legend(['MCMC-ABC', 'MH', 'EP-ABC', 'Analytic'])
    ax.set_xlabel('Probability $p$')
    ax.set_ylabel('Density')

    fig.tight_layout()
    fig.savefig('p_densities.png')

    # Zoomed in solution
    fig.clf()
    ax = fig.add_subplot(111)
    # Exact solution with conjugate prior
    xs = np.arange(0.25, 0.35, 1e-3)
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
    fig.savefig('p_densities_zoomed.png')

    # Save traces
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(ps_abc)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Probability $p$')
    fig.tight_layout()
    fig.savefig('abc_trace.png')

    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(ps_mh)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Probability $p$')
    fig.tight_layout()
    fig.savefig('mh_trace.png')

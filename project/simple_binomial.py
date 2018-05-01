"""Simple ABC example with a binomial model.

   Comparison of MCMC-ABC, conventional MH MCMC and an analytical solution.

   Copyright (c) 2018, Felix Held
"""
import numpy as np
import scipy.stats as stats

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns


def logit(theta):
    return np.log(theta / (1. - theta))


def logistic(theta):
    """Logistic function."""
    return 1. / (1. + np.exp(-theta))


def loglogistic(theta):
    """Log of the logistic function."""
    return -np.log(1. + np.exp(-theta))


def mcmc_abc(data, M=10000, sigma=1e-1, eps=1e-1):
    """Implementation of MCMC-ABC for the particular problem.

    Uniform prior (i.e. Beta(1, 1)) is assumed for p.
    Parameters are inferred on an unbounded parameter space.

    Arguments:
        data - data from experiment
        M - number of iterations
        sigma - random walk standard deviation
        eps - ABC epsilon

    Returns:
        ps - samples for logit(p)
    """
    ps = np.zeros((M + 1,), dtype='float64')

    # Summary statistics from data
    D = np.array([np.mean(data), np.std(data)])

    # Initialize around zero
    ps[0] = 0.1 * np.random.randn()

    total_samples = 0

    for m in tqdm(range(M)):
        while True:
            pnew = ps[m] + sigma * np.random.randn()

            data_sim = np.random.binomial(3, logistic(pnew), size=len(data))
            Dsim = np.array([np.mean(data_sim), np.std(data_sim)])
            total_samples += 1

            if np.linalg.norm(D - Dsim, ord=1) <= eps:
                break

        # Beta(1, 1) prior
        lacc = stats.beta(1, 1).logpdf(logistic(pnew)) + \
            loglogistic(pnew) + np.log(1 - logistic(pnew)) - \
            (stats.beta(1, 1).logpdf(logistic(ps[m])) + \
            loglogistic(ps[m]) + np.log(1 - logistic(ps[m])))

        # Metropolis-Hastings acceptance step
        h = min(1, np.exp(lacc))
        if np.random.rand() <= h:
            ps[m + 1] = pnew
        else:
            ps[m + 1] = ps[m]

    return ps, total_samples


def mh(data, M=10000, sigma=1e-1):
    """Implementation of Metropolis-Hastings for the particular problem.

    Uniform prior (i.e. Beta(1, 1)) is assumed for p.
    Parameters are inferred on an unbounded parameter space.

    Arguments:
        data - data from the experiment
        M - number of iterations
        sigma - random walk std dev

    Returns:
        ps - samples for logit(p)
    """
    ps = np.zeros((M + 1,), dtype='float64')

    # Initialize around zero
    ps[0] = 0.1 * np.random.randn()

    for m in tqdm(range(M)):
        pnew = ps[m] + sigma * np.random.randn()

        # Beta(1, 1) prior
        lacc = np.sum(stats.binom.logpmf(data, 3, logistic(pnew))) + \
            stats.beta(1, 1).logpdf(logistic(pnew)) + \
            loglogistic(pnew) + np.log(1 - logistic(pnew)) - \
            (np.sum(stats.binom.logpmf(data, 3, logistic(ps[m]))) + \
             stats.beta(1, 1).logpdf(logistic(ps[m])) + \
             loglogistic(ps[m]) + np.log(1 - logistic(ps[m])))

        # Metropolis-Hastings acceptance step
        h = min(1, np.exp(lacc))
        if np.random.rand() <= h:
            ps[m + 1] = pnew
        else:
            ps[m + 1] = ps[m]

    return ps


def ep_abc(data, passes=3, M=10000, Mbatch=1000, eps=1e-1):
    """Implementation of EP-ABC for the particular problem.

    Very naive implementation of EP-ABC for one parameter.
    Normal(0, 2.5) prior is used on the unbounded scale, making
    the prior only weakly informative.

    Parameters:
        data - data from the experiment
        passes - number of passes through the dataset
        M - minimum number of accepted samples for moment matching
        Mbatch - number of samples created in every batch
        eps - ABC epsilon
    """
    # Length of data
    N = len(data)

    # For each data point an approximating distribution N(r, q) is sought after
    # In the beginning assume r = 0 and q = 0 for all but prior distribution
    # Initial approximation is then q = f0, i.e. the prior

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
                print('Negative precision: Skipping site %d in pass %d' %
                    (i, k))
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


def ep_abc_iid(data, passes=3, M=10000, Mbatch=1000, eps=1e-1, ess_min=3000):
    """Implementation of iid optimised EP-ABC for the particular problem.

    IID optimised implementation of EP-ABC for one parameter.
    Normal(0, 2.5) prior is used on the unbounded scale, making
    the prior only weakly informative.

    Parameters:
        data - data from the experiment
        passes - number of passes through the dataset
        M - minimum number of accepted samples for moment matching
        Mbatch - number of samples created in every batch
        eps - ABC epsilon
    """
    # Length of data
    N = len(data)

    # For each data point an approximating distribution N(r, q) is sought after
    # In the beginning assume r = 0 and q = 0 for all but prior distribution
    # Initial approximation is then q = f0, i.e. the prior

    total_samples = 0

    # Create arrays for precisions and precision means
    r = np.zeros((N + 1,), dtype='float64')
    q = np.zeros((N + 1,), dtype='float64')

    # Prior Normal(0, 2.5)
    r[0] = 0.           # Precision mean
    q[0] = 1. / 2.5**2  # Precision

    def sample(mu_gen, sigma_gen, i, total_samples):
        theta = np.array([], dtype='float64')
        sim_data = np.array([], dtype='int')
        acc_vec = np.array([], dtype='int')
        m_acc = 0
        while True:
            theta_tmp = stats.norm(mu_gen, sigma_gen).rvs(Mbatch)
            sim_data_tmp = stats.binom.rvs(3, logistic(theta_tmp))

            acc_vec_tmp = (np.abs(sim_data_tmp - data[i]) <= eps)
            m_acc += np.sum(acc_vec.astype('int'))

            theta = np.concatenate((theta, theta_tmp))
            sim_data = np.concatenate((sim_data, sim_data_tmp))
            acc_vec = np.concatenate((acc_vec, acc_vec_tmp))

            total_samples += Mbatch

            if m_acc >= M:
                break

        return theta, sim_data, acc_vec.astype('float64'), total_samples

    ess = np.zeros((passes, N), dtype='float64')

    for k in tqdm(range(passes)):
        for i in tqdm(range(1, N + 1)):
            # Cavity distribution with respect to i-th data point
            r_cavity = np.sum(r[:i]) + np.sum(r[(i + 1):])
            q_cavity = np.sum(q[:i]) + np.sum(q[(i + 1):])

            if q_cavity <= 0.:
                # Only continue if positive precision
                print('Negative precision: Skipping site %d in pass %d' %
                    (i, k))
                continue

            if k == 0 and i == 1:
                # Set new generative mu and sigma
                mu_gen = r_cavity / q_cavity
                sigma_gen = np.sqrt(1. / q_cavity)

                # New samples
                theta, sim_data, ws, total_samples = sample(
                    mu_gen, sigma_gen, i - 1, total_samples)
            else:
                # Calculate importance sampling weights
                ws = np.exp(
                    stats.norm(r_cavity / q_cavity,
                               np.sqrt(1. / q_cavity)).logpdf(theta) -
                    stats.norm(mu_gen, sigma_gen).logpdf(theta)) * \
                    (np.abs(sim_data - data[i - 1]) <= eps)

            # Calculate effective sample size
            ess[k, i - 1] = np.sum(ws)**2 / np.sum(ws**2)
            if ess[k, i - 1] < ess_min:
                # If ESS is too low then resample
                # Set new generative mu and sigma
                mu_gen = r_cavity / q_cavity
                sigma_gen = np.sqrt(1. / q_cavity)

                # New samples
                theta, sim_data, ws, total_samples = sample(
                    mu_gen, sigma_gen, i - 1, total_samples)

            z_norm = np.sum(ws)
            mu_tilted = np.sum(ws * theta) / z_norm
            var_tilted = np.sum(ws * theta**2) / z_norm - mu_tilted**2

            r[i] = mu_tilted / var_tilted - r_cavity
            q[i] = 1. / var_tilted - q_cavity

    return np.sum(r) / np.sum(q), np.sqrt(1. / np.sum(q)), ess, total_samples


if __name__ == '__main__':
    sns.set_style()

    # Simulate data
    data = np.random.binomial(3, 0.3, size=500)
    print('Data', data)
    print(np.mean(data), np.std(data))

    mu, sigma, ess, total_samples = ep_abc_iid(
        data, M=int(1e6), Mbatch=10000,
        ess_min=int(2e-2 * 1e6), eps=1e-1, passes=4)
    print('\nEP-ABC IID-opt', mu, sigma)
    print('\nEP-ABC total samples', total_samples)

    # mu, sigma = ep_abc(data, M=100000, Mbatch=10000, eps=1e-1)
    # print('\nEP-ABC', mu, sigma)

    # Infere parameters
    ps_abc, total_samples = mcmc_abc(data, M=10000,
                                     sigma=5e-1, eps=1e-2)
    print('\nMCMC-ABC total samples', total_samples)
    ps_mh = mh(data, M=10000)

    # Kernel density estimates for the samples
    kde_abc = stats.gaussian_kde(logistic(ps_abc[2000:]), 'silverman')
    kde_mh = stats.gaussian_kde(logistic(ps_mh[2000:]), 'silverman')

    # Exact solution with conjugate prior
    xs = np.arange(0.01, 1., 1e-3)
    ys = stats.beta(1 + np.sum(data), 1 + np.sum(3 - data)).pdf(xs)

    # Solution from expectation propagation
    ys_ep = stats.norm(mu, sigma).pdf(logit(xs)) / (xs * (1 - xs))

    # Plot solutions
    plt.plot(xs, kde_abc(xs))
    plt.plot(xs, kde_mh(xs), '--')
    plt.plot(xs, ys_ep, '-.')
    plt.plot(xs, ys, ':')
    plt.legend(['MCMC-ABC', 'MH', 'EP-ABC', 'Analytic'])

    plt.savefig('p_densities.png')

    # Zoomed in solution
    plt.clf()
    # Exact solution with conjugate prior
    xs = np.arange(0.2, 0.4, 1e-3)
    ys = stats.beta(1 + np.sum(data), 1 + np.sum(3 - data)).pdf(xs)

    # Solution from expectation propagation
    ys_ep = stats.norm(mu, sigma).pdf(logit(xs)) / (xs * (1 - xs))

    plt.plot(xs, kde_abc(xs))
    plt.plot(xs, kde_mh(xs), '--')
    plt.plot(xs, ys_ep, '-.')
    plt.plot(xs, ys, '--')
    plt.legend(['MCMC-ABC', 'MH', 'EP-ABC', 'Analytic'])

    plt.savefig('p_densities_zoomed.png')

    # Save traces
    plt.clf()
    plt.plot(ps_abc)
    plt.savefig('abc_trace.png')

    plt.clf()
    plt.plot(ps_mh)
    plt.savefig('mh_trace.png')

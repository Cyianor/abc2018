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

    Beta(2, 2) prior is assumed for p. Parameters are inferred on
    an unbounded parameter space.

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

    # # Initialize with prior
    # ps[0] = -np.log(1. / stats.beta(2, 2).rvs() - 1.)
    # Initialize around zero
    ps[0] = 0.1 * np.random.randn()

    for m in tqdm(range(M)):
        while True:
            pnew = ps[m] + sigma * np.random.randn()

            data_sim = np.random.binomial(3, logistic(pnew), size=len(data))
            Dsim = np.array([np.mean(data_sim), np.std(data_sim)])

            if np.linalg.norm(D - Dsim, ord=1) <= eps:
                break

        # Beta(2, 2) prior
        lacc = stats.beta(2, 2).logpdf(logistic(pnew)) + \
            loglogistic(pnew) + np.log(1 - logistic(pnew)) - \
            (stats.beta(2, 2).logpdf(logistic(ps[m])) + \
            loglogistic(ps[m]) + np.log(1 - logistic(ps[m])))
        # # Normal(0, 1.6) prior on unbounded scale
        # lacc = stats.norm(0, 1.6).logpdf(pnew) - \
        #     stats.norm(0, 1.6).logpdf(ps[m])

        # Metropolis-Hastings acceptance step
        h = min(1, np.exp(lacc))
        if np.random.rand() <= h:
            ps[m + 1] = pnew
        else:
            ps[m + 1] = ps[m]

    return ps


def mh(data, M=10000, sigma=1e-1):
    """Implementation of Metropolis-Hastings for the particular problem.

    Beta(2, 2) prior is assumed for p. Parameters are inferred on
    an unbounded parameter space.

    Arguments:
        data - data from the experiment
        M - number of iterations
        sigma - random walk std dev

    Returns:
        ps - samples for logit(p)
    """
    ps = np.zeros((M + 1,), dtype='float64')

    # # Initialize with prior
    # ps[0] = -np.log(1. / stats.beta(2, 2).rvs() - 1.)
    # Initialize around zero
    ps[0] = 0.1 * np.random.randn()

    for m in tqdm(range(M)):
        pnew = ps[m] + sigma * np.random.randn()

        # Beta(2, 2) prior
        lacc = np.sum(stats.binom.logpmf(data, 3, logistic(pnew))) + \
            stats.beta(2, 2).logpdf(logistic(pnew)) + \
            loglogistic(pnew) + np.log(1 - logistic(pnew)) - \
            (np.sum(stats.binom.logpmf(data, 3, logistic(ps[m]))) + \
             stats.beta(2, 2).logpdf(logistic(ps[m])) + \
             loglogistic(ps[m]) + np.log(1 - logistic(ps[m])))
        # # Normal(0, 1.6) prior on unbounded scale
        # lacc = np.sum(stats.binom.logpmf(data, 3, logistic(pnew))) + \
        #     stats.norm(0, 1.6).logpdf(pnew) - \
        #     (np.sum(stats.binom.logpmf(data, 3, logistic(ps[m]))) + \
        #      stats.norm(0, 1.6).logpdf(ps[m]))

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
    Normal(0, 1.6) prior is used on the unbounded scale.
    This prior has very similar properties as a Beta(2, 2)
    prior on the bounded scale.

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

    # Create arrays for precisions and precision means
    r = np.zeros((N + 1,), dtype='float64')
    q = np.zeros((N + 1,), dtype='float64')

    # Prior Normal(0, 1.6)
    r[0] = 0.           # Precision mean
    q[0] = 1. / 1.6**2  # Precision

    for k in tqdm(range(passes)):
        for i in tqdm(range(1, N + 1)):
            # Cavity distribution with respect to i-th data point
            r_cavity = np.sum(r[:i]) + np.sum(r[(i + 1):])
            q_cavity = np.sum(q[:i]) + np.sum(q[(i + 1):])

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

                if m_acc >= M:
                    break

            mu_tilted = np.sum(theta_acc) / m_acc
            var_tilted = np.sum(theta_acc**2) / m_acc - mu_tilted**2

            r[i] = mu_tilted / var_tilted - r_cavity
            q[i] = 1. / var_tilted - q_cavity

    return np.sum(r) / np.sum(q), np.sqrt(1. / np.sum(q))


def ep_abc_iid(data, passes=3, M=10000, Mbatch=1000, eps=1e-1, ess_min=3000):
    """Implementation of iid optimised EP-ABC for the particular problem.

    IID optimised implementation of EP-ABC for one parameter.
    Normal(0, 1.6) prior is used on the unbounded scale.
    This prior has very similar properties as a Beta(2, 2)
    prior on the bounded scale.

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

    # Create arrays for precisions and precision means
    r = np.zeros((N + 1,), dtype='float64')
    q = np.zeros((N + 1,), dtype='float64')

    # Prior Normal(0, 1.6)
    r[0] = 0.           # Precision mean
    q[0] = 1. / 1.6**2  # Precision

    def sample(mu_gen, sigma_gen, i):
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

            if m_acc >= M:
                break

        return theta, sim_data, acc_vec.astype('float64')

    ess = np.zeros((passes, N), dtype='float64')

    for k in tqdm(range(passes)):
        for i in tqdm(range(1, N + 1)):
            # Cavity distribution with respect to i-th data point
            r_cavity = np.sum(r[:i]) + np.sum(r[(i + 1):])
            q_cavity = np.sum(q[:i]) + np.sum(q[(i + 1):])

            if k == 0 and i == 1:
                # Set new generative mu and sigma
                mu_gen = r_cavity / q_cavity
                sigma_gen = np.sqrt(1. / q_cavity)

                # New samples
                theta, sim_data, ws = sample(mu_gen, sigma_gen, i - 1)
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
                theta, sim_data, ws = sample(mu_gen, sigma_gen, i - 1)

            z_norm = np.sum(ws)
            mu_tilted = np.sum(ws * theta) / z_norm
            var_tilted = np.sum(ws * theta**2) / z_norm - mu_tilted**2

            r[i] = mu_tilted / var_tilted - r_cavity
            q[i] = 1. / var_tilted - q_cavity

    return np.sum(r) / np.sum(q), np.sqrt(1. / np.sum(q)), ess


if __name__ == '__main__':
    sns.set_style()

    # Simulate data
    data = np.random.binomial(3, 0.3, size=100)
    print('Data', data)
    print(np.mean(data), np.std(data))

    mu, sigma, ess = ep_abc_iid(data, M=100000, Mbatch=10000,
                                ess_min=60000, eps=1e-1)
    print('\nEP-ABC IID-opt', mu, sigma)

    # mu, sigma = ep_abc(data, M=100000, Mbatch=10000, eps=1e-1)
    # print('\nEP-ABC', mu, sigma)

    # Infere parameters
    ps_abc = mcmc_abc(data, M=50000, eps=.15)[5000:]
    ps_mh = mh(data, M=50000)[5000:]

    # Exact solution with conjugate prior
    xs = np.arange(0.01, 1., 1e-3)
    ys = stats.beta(2 + np.sum(data), 2 + np.sum(3 - data)).pdf(xs)

    # Solution from expectation propagation
    ys_ep = stats.norm(mu, sigma).pdf(logit(xs)) / (xs * (1 - xs))

    # Kernel density estimates for the samples
    kde_abc = stats.gaussian_kde(logistic(ps_abc), 'silverman')
    kde_mh = stats.gaussian_kde(logistic(ps_mh), 'silverman')

    # Plot solutions
    plt.plot(xs, ys)
    plt.plot(xs, kde_abc(xs))
    plt.plot(xs, kde_mh(xs))
    plt.plot(xs, ys_ep)
    # plt.hist(logistic(ps_abc), bins=30, density=True, alpha=0.4, color='b');
    # plt.hist(logistic(ps_mh), bins=30, density=True, alpha=0.4, color='r');
    plt.legend(['Analytic', 'MCMC-ABC', 'MH', 'EP-ABC'])

    plt.savefig('p_densities.png')

    # Save traces
    plt.clf()
    plt.plot(ps_abc)
    plt.savefig('abc_trace.png')

    plt.clf()
    plt.plot(ps_mh)
    plt.savefig('mh_trace.png')

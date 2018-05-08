"""ABC example for a stable distribution.

   Copyright (c) 2018, Felix Held
"""
import numpy as np
import scipy.stats as stats
import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl


def stable_rvs(alpha, beta, mu=0, c=1, size=1):
    """Simulates a stable random variable.

    :Arguments:
        alpha : float
            Stability parameter, 0 < alpha <= 2
        beta : float
            Skewness parameter, -1 <= beta <= 1
        mu : float
            Location parameter
        c : float
            Scale parameter, 0 < c

    :Returns:
        float or NumPy array : Simulations from the specified stable
            distribution
    """
    if size == 1:
        # Use that NumPy spits out floats if size is None
        s = None
    else:
        s = size

    U = (np.random.uniform(size=s) - 0.5) * np.pi
    if not (alpha == 1. and beta == 0.):
        W = np.random.exponential(size=s)

    pih = np.pi / 2.

    if alpha == 1.:
        if beta == 0.:
            X = np.tan(U)
            Y = c * X + mu
        else:
            X = 1 / pih * (
                (pih + beta * U) * np.tan(U) - beta * np.log(
                    pih * W * np.cos(U) / (pih + beta * U)))
            Y = c * X + 1 / pih * beta * c * np.log(c) + mu
    else:
        if beta == 0.:
            X = np.sin(alpha * U) / np.cos(U)**(1 / alpha) * \
                (np.cos(U * (1 - alpha)) / W)**((1 - alpha) / alpha)
        else:
            zeta = beta * np.tan(np.pi * alpha / 2.)
            chi = np.arctan(zeta)
            X = (1 + zeta**2)**(1 / (2 * alpha)) * \
                np.sin(alpha * U + chi) / np.cos(U)**(1 / alpha) * \
                (np.cos(U * (1 - alpha) - chi) / W)**((1 - alpha) / alpha)
        Y = c * X + mu

    return Y


def stable_rvs_unbnd_vec(theta):
    """Support parameter vector for stable random variates.

    Algorithm by Chambers, Mallows and Stuck (from Wikipedia,
    https://en.wikipedia.org/wiki/Stable_distribution#Simulation_of_stable_variables)

    :Arguments:
        theta : NumPy array (2dim)
            Parameters (probit(alpha / 2), probit((beta + 1) / 2), mu, log(c))
            Each row represents one set of parameters

    :Returns:
        NumPy array : random variates from the specified
            stable distribution
    """
    alpha = 2. * stats.norm.cdf(theta[:, 0])
    beta = 2. * stats.norm.cdf(theta[:, 1]) - 1.
    mu = theta[:, 2]
    c = np.exp(theta[:, 3])
    # mu = loc
    # c = scale
    N = theta.shape[0]

    U = (np.random.rand(N) - 0.5) * np.pi
    W = -np.log(np.random.rand(N))

    zeta = beta * np.tan((np.pi * alpha) / 2.)
    chi = np.arctan(zeta)
    X = (1 + zeta**2)**(1 / (2 * alpha)) * \
        np.sin(alpha * U + chi) / np.cos(U)**(1 / alpha) * \
        (np.cos(U * (1 - alpha) - chi) / W)**((1 - alpha) / alpha)

    return c * X + mu


def cholesky_inv(A):
    """Calculate Cholesky inverse of a matrix.

    Let U be the Cholesky decomposition of a `n x n` matrix A. Then

    ::
        Ainv = U \\ (U' \\ identity(n))

    where `\\` indicates the solution of a linear equation system.

    :Arguments:
        A : NumPy array
            Symmetric real-valued matrix to be inverted

    :Returns:
        NumPy array : Inverse to A
    """
    # U = np.linalg.cholesky(A)
    # return np.linalg.solve(U, np.linalg.solve(U.T, np.identity(A.shape[0])))
    return np.linalg.inv(A)


def ep_abc_iid(data, M=10000, Mbatch=1000, ess_min=3000,
               passes=3, eps=1, a=1.):
    """Implementation of iid optimised EP-ABC for the particular problem.

    IID optimised implementation of EP-ABC for four parameters.

    Parameters are converted to the real line via
    (probit(alpha / 2), probit((beta + 1) / 2), mu, log(c)), where
    probit is the inverse of the N(0, 1) CDF.

    Normal(0, diag(1, 1, 10, 10)) prior is used on the unbounded scale, making
    the prior only weakly informative.

    :Arguments:
        data : NumPy array
            Simulated data
        M : int
            Minimum number of accepted samples for moment matching
        Mbatch : int
            Number of samples created in every batch
        ess_min : int
            Minimal allowed ESS value
        passes : int
            Number of passes through the dataset
        eps : float
            ABC epsilon
        a : float
            Determines how quickly the global approximation is updated.
            Parameter is chosen to be between (0, 1). 1.0 means to apply no
            damping at all and close to zero means to update the global
            approximation only very slowly (default: 1.0)

    :Returns:
        (NumPy array, NumPy array, int) : Normal approximation mu, sigma
            and total number of simulations
    """
    # Length of data
    N = len(data)

    # For each data point an approximating distribution N(r, q) is sought after
    # In the beginning assume r = 0 and q = 0 for all but prior distribution
    # Initial approximation is then q = f0, i.e. the prior

    # Save total number of sampled values
    total_samples = 0

    # Create arrays for precision matrices and precision means
    r = np.zeros((N + 1, 4), dtype='float64')
    q = np.zeros((N + 1, 4, 4), dtype='float64')

    # Prior
    r[0, :] = np.zeros((4,), dtype='float')             # Precision mean
    q[0, :, :] = cholesky_inv(np.diag([1, 1, 10, 10]))  # Precision

    # Global approximations
    R = np.sum(r, axis=0)
    Q = np.sum(q, axis=0)

    # Natural parameterisation
    sigma = cholesky_inv(Q)
    mu = sigma.dot(R)

    progress = tqdm(range(passes))

    def sample(mu_gen, sigma_gen, i):
        # Access total_samples and progress from outer scope
        nonlocal total_samples
        nonlocal progress

        # Preparations
        m = len(mu_gen)
        mu_gen = np.tile(mu_gen, (Mbatch, 1))

        theta = np.empty((0, m), dtype='float64')
        sim_data = np.empty((0,), dtype='int')
        m_acc = 0

        # Cholesky decomposition of covariance matrix
        U = np.linalg.cholesky(sigma_gen)
        # Constant factor for likelihood calculation
        lconst = -(m / 2) * np.log(2 * np.pi) - np.log(np.linalg.det(U))
        # Likelihood values
        lpdf = np.empty((0,), dtype='float64')

        while True:
            z = np.random.randn(Mbatch, m)
            theta_tmp = z.dot(U) + mu_gen

            sim_data_tmp = stable_rvs_unbnd_vec(theta_tmp)
            total_samples += Mbatch

            acc = (np.abs(sim_data_tmp - data[i]) < eps)
            m_acc += np.sum(acc.astype('int'))

            theta = np.concatenate((theta, theta_tmp[acc, :]))
            sim_data = np.concatenate((sim_data, sim_data_tmp[acc]))
            lpdf = np.concatenate(
                (lpdf, (lconst - .5 * np.sum((z * z).T, axis=0))[acc]))

            progress.set_postfix_str('%.1f' % (m_acc / M * 100.))
            if m_acc >= M:
                break

        return theta, sim_data, lpdf, \
            np.ones((theta.shape[0],), dtype='float64')

    for k in progress:
        for i in tqdm(range(1, N + 1)):
            # Cavity distribution with respect to i-th data point
            r_cavity = R - r[i, :]
            q_cavity = Q - q[i, :, :]

            try:
                sigma_cavity = cholesky_inv(q_cavity)
                mu_cavity = sigma_cavity.dot(r_cavity)
            except np.linalg.LinAlgError:
                # Only continue if positive definite precision matrix
                print(
                    'Negative precision: Skipping site %d in pass %d' % (i, k))
                continue

            if k == 0 and i == 1:
                # Set new generative mu and sigma
                mu_gen = mu_cavity
                sigma_gen = sigma_cavity

                # New samples
                theta, sim_data, lpdf_gen, ws = sample(
                    mu_gen, sigma_gen, i - 1)
                ess = theta.shape[0]
            else:
                # Calculate importance sampling weights
                l_ws = stats.multivariate_normal(
                    mu_cavity, sigma_cavity).logpdf(theta) - lpdf_gen
                l_ws -= np.max(l_ws)
                acc = (np.abs(sim_data - data[i - 1]) < eps).astype('int')
                ws = np.exp(l_ws) * acc
                ws_sum = np.sum(ws)
                if ws_sum > 0.:
                    ess = ws_sum**2 / np.sum(ws**2)
                else:
                    ess = 0

            # Check minimum effective sample size
            if ess < ess_min:
                # If ESS is too low then resample
                # Set new generative mu and sigma
                mu_gen = mu_cavity
                sigma_gen = sigma_cavity

                # New samples
                theta, sim_data, lpdf_gen, ws = sample(
                    mu_gen, sigma_gen, i - 1)

            mu_tilted = np.average(theta, 0, ws)
            sigma_tilted = np.cov(theta, rowvar=0, ddof=0, aweights=ws)

            q_tilted = cholesky_inv(sigma_tilted)
            r_tilted = q_tilted.dot(mu_tilted)

            # Update global approximation (weighted with old parameters)
            Q = a * q_tilted + (1 - a) * Q
            R = a * r_tilted + (1 - a) * R

            # Natural parameterisation
            sigma = cholesky_inv(Q)
            mu = sigma.dot(R)

            # if (i - 1) % 100 == 0:
            #     print('\n', mu)

            r[i, :] = R - r_cavity
            q[i, :, :] = Q - q_cavity

    return mu, sigma, total_samples


if __name__ == '__main__':
    # Import return data
    zt = pd.read_csv('euro_in_sterling.csv').zt.values
    # Calculate scaled log-return rates
    yt = 100. * np.log(zt[1:] / zt[:-1])

    # Seed the RNG
    np.random.seed(20180503)
    # Turn of some annoying warnings
    np.seterr(over='ignore', divide='ignore', invalid='ignore')

    # # Generate some data
    yt = stable_rvs(1.61, -0.2, 0., 0.41, size=500)

    # Visualise data
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.plot(yt, '-o', markersize=2)
    ax.set_ylabel('Scaled log-return')
    fig.tight_layout()
    fig.savefig('scaled_log_return_sim2.png')

    mu, sigma, total_samples = ep_abc_iid(
        yt, M=int(8e6), Mbatch=int(5e6),
        ess_min=int(2e4), eps=1., passes=1)

    # Save data from this run
    np.savez('stable_dist_sim2', yt, mu, sigma, total_samples)

    print(mu, sigma, total_samples)

    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.serif'] = 'STIX Two Text'
    mpl.rcParams['font.family'] = 'serif'

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # alpha
    xs = np.arange(0, 2.1, 1e-3)
    axs[0, 0].plot(xs, stats.norm.pdf(
        stats.norm.ppf(xs / 2), mu[0], np.sqrt(sigma[0, 0])) /
        (stats.norm.pdf(stats.norm.ppf(xs / 2)) * 2))
    axs[0, 0].set_title('$\\alpha$')

    # beta
    xs = np.arange(-1, 1.1, 1e-3)
    axs[0, 1].plot(xs, stats.norm.pdf(
        stats.norm.ppf((xs + 1) / 2), mu[1], np.sqrt(sigma[1, 1])) /
        (stats.norm.pdf(stats.norm.ppf((xs + 1) / 2)) * 2))
    axs[0, 1].set_title('$\\beta$')

    # mu
    xs = np.arange(-1, 1.1, 1e-3)
    axs[1, 0].plot(xs, stats.norm.pdf(xs, mu[2], np.sqrt(sigma[2, 2])))
    axs[1, 0].set_title('$\\mu$')

    # c
    xs = np.arange(0.01, 0.5, 1e-3)
    axs[1, 1].plot(xs, stats.norm.pdf(
        np.log(xs), mu[3], np.sqrt(sigma[3, 3])) / xs)
    axs[1, 1].set_title('$c$')

    fig.tight_layout()
    fig.savefig('stable_dist_densities_sim2.png')

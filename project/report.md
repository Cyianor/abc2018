---
title: >
  Expectation Propagation in the framework of Approximate Bayesian Computation:
  Two case studies
author: Felix Held
date: 2018-05-08
documentclass: scrartcl
bibliography: refs.bib
biblio-title: References
biblio-style: authoryear
biblatexoptions:
  - backend=biber
header-includes:
  - \usepackage{unicode-math}
  - \setmainfont{STIX Two Text}
  - \setmathfont{STIX Two Math}
  - \usepackage{microtype}
  - \usepackage{dsfont}
xnos-capitalise: On
cleveref: On
---

# Introduction

Approximate Bayesian Computation (ABC) aims to extend
Bayesian posterior inference to problems where no explicit likelihood function
is available or its computation is costly. Explicit evaluation of the
likelihood can be circumvented if an efficient simulation mechanism
is available.

The aim of this project was to investigate
the *Expectation Propagation* (EP) algorithm for likelihood-free
inference. EP was originally described in @Minka2001 and extended to
the likelihood-free case in @Barthelme2014a.

In this project, EP-ABC was implemented for two case studies. It's performance
was explored and compared to other established methods.

# Algorithm

EP in itself calculates an approximation to the posterior distribution.
Assume that the posterior distribution factors like
$$
\pi(\theta | y_{1:n}) \propto \prod_i l_i(\theta).
$${#eq:posterior}
A common assumption is that $l_0(\theta)$ is the prior for $\theta$ and
$l_i(\theta) = p(y_i | y_{1:i - 1}, \theta)$ are the likelihood factors.
EP then constructs parametric approximations $f_i(\theta)$ 
for each $l_i(\theta)$. These approximations can be chosen from the 
exponential family and in typical examples the normal distribution is chosen.
Let therefore
$$
f_i(\theta) \propto \exp\left(-\frac{1}{2} \theta^T Q_i \theta + 
  r_i^T \theta\right),
$${#eq:ep-factors}
where $Q_i$ is the precision matrix 
(i.e. the inverse of the covariance matrix of $f_i$) and $r_i$ is the
precision mean (i.e. $r_i = Q_i \mu_i$ where $\mu_i$ is the mean of $f_i$).
Parameterising the $f_i$ in the so called *natural parameterisation* of the
normal distribution makes it very easy to describe the global approximation.
Since every $f_i$ approximates one of the $l_i$ the global approximation
for $\pi(\theta | y_{1:n})$ is
$$
q(\theta) \propto 
\exp\left(-\frac{1}{2} \theta^T \left(\sum_i Q_i\right) \theta +
\left(\sum_i r_i\right)^T \theta\right)
$${#eq:global-approx}
To arrive at the normal distribution approximations $f_i$, 
EP performs the following iterative updates

1. Determine the cavity distribution
$$
q_{-i}(\theta) = \prod_{j \neq i} f_j(\theta) \propto 
\exp\left(-\frac{1}{2} \theta^T Q_{-i} \theta + r_{-i}^T \theta\right)
$$
   where $Q_{-i} = \sum_{j \neq i} Q_j$ and $r_{-i} = \sum_{j \neq i} r_j$.
2. Form the tilted/hybrid distribution 
$q_{/i}(\theta) \propto q_{-i}(\theta) l_i(\theta)$.
3. Calculate the updates
$$
\begin{aligned}
  Z &= \int l_i(\theta) q_{-i}(\theta) \mathrm{d}\theta \\
  \mu &= \frac{1}{Z} \int \theta l_i(\theta) q_{-i}(\theta) \mathrm{d}\theta \\
  \Sigma &= \frac{1}{Z} \int \theta \theta^T l_i(\theta) q_{-i}(\theta)
  \mathrm{d} \theta - \mu \mu^T
\end{aligned}
$${#eq:ep-moment-updates}
4. Update site with
$$
Q_i = \Sigma^{-1} - Q_{-i}, \quad r_i = \Sigma^{-1} \mu - r_{-i}
$${#eq:ep-site-updates}

It is common to sweep through the data in multiple passes to ensure proper 
convergence. According to @Barthelme2014a it is often enough with 3-4 passes.

This algorithm depends on the evaluation of the integrals in 
@eq:ep-moment-updates. These can only be evaluated if $l_i(\theta)$ is
accessible. In a likelihood-free inference however, these might not be
available. The idea, presented in @Barthelme2014a, suggests the following
solution:

1. Let $\varepsilon > 0$, $M \in \mathbb{N}$, $\Sigma_{-i} = Q_{-i}^{-1}$ and
   $\mu_{-i} = \Sigma_{-i} r_{-i}$.
2. Simulate $M$ parameters $\theta^{(m)} \sim N(\mu_{-i}, \Sigma{-i})$
3. For every $\theta^{(k)}$, $k = 1, \dots, M$ simulate
   $\hat{y}_{i}^{(m)}$ from
   $p(\hat{y}_i^{(m)} | y_{1:i - 1}, \theta^{(m)})$.
4. Calculate
$$
\begin{aligned}
M_{\mathrm{acc}} &= \sum_{m = 1}^M
  \mathds{1}_{\{\|\hat{y}^{(m)}_i - y_i\| \leq \varepsilon\}} \\
\hat{\mu} &= \frac{1}{M_{\mathrm{acc}}}
  \sum_{m = 1}^M \theta^{(m)}
  \mathds{1}_{\{\|\hat{y}^{(m)}_i - y_i\| \leq \varepsilon\}} \\
\hat{\Sigma} &= \frac{1}{M_{\mathrm{acc}}}
  \sum_{m = 1}^M \theta^{(m)} (\theta^{(m)})^T
  \mathds{1}_{\{\|\hat{y}^{(m)}_i - y_i\| \leq \varepsilon\}} -
  \hat{\mu}\hat{\mu}^T
\end{aligned}
$${#eq:ep-moment-approx}
5. Update 
$$
Q_i = \hat{\Sigma}^{-1} - Q_{-i}, \quad
r_i = \hat{\Sigma}^{-1} \hat{\mu} - r_{-i}
$${#eq:ep-site-approx}

To ensure numerical stability, a minimum number $M_{\min}$ of accepted 
samples was specified. 

To speed up the algorithm, parameters and data simulated
for one site, can be re-used through a importance sampling scheme.
This is possible as long as data are iid.
Weights are calculated like
$$
w_{i + 1}^{(m)} = \frac{q_{-(i + 1)}(\theta^{(m)})}{q_{-i}(\theta^{(m)})}
    \mathds{1}_{\{\|y^{(m)} - y_{i + 1}\| \leq \varepsilon\}}.
$${#eq:ep-weights}
If the effective sample size
$$
  \mathrm{ESS} = \frac{\left(\sum_{m = 1}^M w_i^{(m)}\right)}{%
  \sum_{m = 1}^M \left(w_i^{(m)}\right)^2}
$${#eq:ep-ess}
is above a pre-specified threshold it is reasonable to re-use the samples.
Calculation of approximated moments is then modified to
$$
\begin{aligned}
\hat{Z} &= \sum_{m = 1}^M w_i^{(m)}, \\
\hat{\mu} &= \frac{1}{\hat{Z}}
  \sum_{m = 1}^M w_i^{(m)} \theta^{(m)}, \\
\hat{\Sigma} &= \frac{1}{\hat{Z}}
  \sum_{m = 1}^M w_i^{(m)} \theta^{(m)} (\theta^{(m)})^T -
  \hat{\mu}\hat{\mu}^T.
\end{aligned}
$${#eq:ep-moment-approx-weighted}
New samples are generated if ESS is below the specified threshold.

# Case-studies

## Success probability of a binomial distribution

To test the algorithm on a very simple example, a binomial variable was 
simulated with $n = 10$ and $p = 0.3$. The simulated data can be seen
in @fig:binomial-data.

![Data from $N = 1000$ simulations of a binomial variable with $n = 10$
  and $p = 3$.](figures/histogram_binomial.png){#fig:binomial-data}

To formulate this as a Bayesian inference problem, the parameter $n$ was
assumed to be known and the task was to re-estimate $p$ from the data. 
No prior knowledge was assumed and therefore $p$ was given a 
Beta$(1, 1)$ prior, which is effectively a uniform prior. The analytical 
solution can then be calculated from the data as
$$
\pi(\theta | y_{1:N}) = \mathrm{Beta}\left(\theta; 1 + \sum_{i = 1}^{N} k_i, 
1 + \sum_{i = 1}^N (N - k_i)\right)
$${#eq:binomial-analytic-sol}
where $k_i$ is the $i$-th binomial outcome.

To investigate the performance of the EP algorithm, it was compared to
the analytical solution of the Bayesian posterior, a solution from
the Metropolis-Hastings algorithm [@Robert2004], utilising a likelihood, and 
a solution from a likelihood-free implementation of MCMC-ABC [@Marjoram2003].

In all cases inference was done on an unbounded scale. To achieve this, the
logit transform
$$
\mathrm{logit}(p) = \log\left(\frac{p}{1 - p}\right)
$$
was used to transform the probability $p$ to the real axis. To transform the
unbounded parameter back to the interval $(0, 1)$ the logistic function
$$
\mathrm{logistic}(x) = \frac{1}{1 + \exp(-x)}
$$
was used.

For the EP algorithm it was more convenient to choose a normal prior, since
the prior enters the approximation as $f_0$. Experiments showed that a
uniform prior on the interval $(0, 1)$ approximately lead to a normal prior
with mean zero and standard deviation 1.8. Since many data points were used
the prior was deemed less important and $N(0, 2.5)$ was chosen as an even 
broader prior.

For EP-ABC the distance function was chosen as the absolute distance between
simulated and observed data. $\varepsilon$ was chosen below 1, which effectively
lead to exact sampling. Since only one data point is compared at a time
this was deemed acceptable. Furthermore, $M_{\min} = 8 \cdot 10^5$ was chosen
as the minimum number of accepted samples per site-update, ESS minimum was
set to $2 \cdot 10^4$ and through trials it was found that one pass through
the data is sufficient.

For MCMC-ABC the sufficient statistics were chosen as the sample mean and 
sample standard deviation. This is weakly justifiable by the central limit 
theorem. For $np > 5$ and $np(1 - p) > 5$ it is justifiable to approximate a 
binomial distribution by a normal distribution. A normal distribution in
turn has the sample mean and sample variance as sufficient statistics if
both mean and variance of the normal distribution are unknown.
The inequalities are not properly fulfilled in this particular case, since 
$np = 3$ and $np(1 - p) = 2.1$. Sample mean and standard deviation
were chosen for estimation nonetheless. Balancing speed and accuracy
$\varepsilon = 0.01$ was chosen.

The standard deviation of the random walk in MCMC-ABC was set $\sigma = 0.5$
and to $\sigma = 0.1$ for Metropolis-Hastings. To accomodate for burn-in, 
2000 iterations were discarded at the beginning of the chains for MCMC-ABC 
and Metropolis-Hastings. Both chains were run for 30000 iterations.

## Parameters of a stable distribution

For a second case study exchange rates were downloaded from the Bank of England.
Specifically, exchange rates for Euro in Pound Sterling from 1st January 2010
to the 31st December 2015. From these exchange rates $z_t$, scaled log-return 
rates $y_t = 100 \cdot \log(z_t / z_{t - 1})$ were calculated. This
led to 1514 data points $y_t$ that are visualised in @fig:stable-data.

![Exchange rates $z_t$ (left) and scaled log-return rates $y_t$ (right) per
  day. Downloaded from the Bank of England for Euro in Pound Sterling
  for the period 1st January 2010 to 31st December 2015.
  ](figures/data_stable.png){#fig:stable-data}

It was attempted to fit a four-parameter stable distribution [@Nolan2018] to
the scaled log-return rates $y_t$. Stable distributions do not have a
density in closed-form. However, they can be cheaply simulated with the
algorithm described in @Chambers1976.

The four parameters of a stable distribution are

- $0 < \alpha \leq 2$, determining the weight of the tails
- $-1 \leq \beta \leq 1$, determining the skewness of the distribution
- $c > 0$, scale parameter
- $\mu$, location parameter

As in the first case study, parameter inference was done on an unbounded scale.
The parameters were transformed as
$$
\theta = \left(\Phi^{-1}\left(\frac{\alpha}{2}\right), 
               \Phi^{-1}\left(\frac{\beta + 1}{2}\right),
               \log(c), \mu\right)
$$
where $\Phi^{-1}$ is the inverse CDF of the standard normal distribution.
The prior was chosen as
$$
  f_0(\theta) = N(\mathbf{0}_4, \mathrm{diag}(1, 1, 10, 10))
$$
which is only weakly informative.

The likelihood-free EP algorithm was implemented for this problem to estimate
an approximation to the posterior distribution. By visual inspection of the
data and by balancing runtime with expected accuracy $\varepsilon = 1$ was
chosen. Additionally, $M_{\min} = 8 \cdot 10^6$ and a minimum ESS of
$2 \cdot 10^4$ were chosen. Given the computational intensity only one pass
was performed through the data set.

The mode of this distribution was compared to the result of the
R package `StableEstim` [@StableEstim], which implements various methods
to estimate parameters of stable distributions. The algorithm that
was chosen for comparison is the maximum likelihood algorithm described 
in @Nolan2001.

# Results

## Success probability of a binomial distribution

All three numerical methods recovered the posterior distribution for $p$. 
The posterior densities can be seen in @fig:p-posteriors.

![The posterior distributions from all four methods described above. Results
  from MCMC-ABC, Metropolis-Hastings and the analytical solution overlap 
  almost exactly. The solution derived from the EP approximation is slightly
  broader in the left tail and shows a slight bias in the direction of $p = 0.3$, 
  the exact solution.
 ](figures/p_densities_zoomed.png){#fig:p-posteriors}

The densities for MCMC-ABC, Metropolis-Hastings and the analytical solution
overlap almost perfectly. The solution for EP-ABC is slightly broader in the
left tail and shows a slight bias towards $p = 0.3$, the exact solution. 
It was observed during experiments that this bias was common behaviour 
for EP-ABC for this particular example. 

Computationally, the EP-ABC algorithm took about 22 min to complete, while
MCMC-ABC took just about 1 hour. Likelihood-supported Metropolis-Hastings
took just only one minute to complete.

During the execution of the EP-ABC algorithm about $6.6 \cdot 10^{9}$ samples 
were generated. MCMC-ABC needed $33.9 \cdot 10^{9}$ samples, which is 
about a factor 5 more than EP-ABC required.

## Parameters of a stable distribution

Estimation of the parameters $\alpha$, $\beta$, $\mu$ and $c$ for a stable
distribution from the scaled log-return rates did not properly succeed. 
EP-ABC ran for about 13 hours to achieve one pass through the data and 
generated $7.7 \cdot 10^{10}$ samples along the way. EP-ABC estimated
a normal distribution with
$$
\mu = (-0.9698, -0.5560,  0.09019, -5.701)
$$
and
$$
\Sigma = 10^{-4} \begin{pmatrix}
 6.1 & -4.1 \cdot 10^{-2} &  -9.4 \cdot 10^{-4} & 1.4 \cdot 10^{-3} \\
-4.1 \cdot 10^{-2} & 6.0 & -1.6 \cdot 10^{-3} & -5.4 \cdot 10^{-4} \\
-9.4 \cdot 10^{-4} & -1.6 \cdot 10^{-3} & 2.8 \cdot 10^{-1} & -6.2 \cdot 10^{-4} \\
 1.4 \cdot 10^{-3} & -5.4 \cdot 10^{-4} & -6.2 \cdot 10^{-4} & 7.4 \cdot 10^{-2}
\end{pmatrix}
$${#eq:ep-cov}
on the unbounded scale. Note the very small factor $10^{-4}$ in front
of the the matrix in @eq:ep-cov. By applying suitable re-transformations, the 
marginal densities on the bounded scales can be recovered. These are shown 
in @fig:densities-stable.

![Marginal posterior densities for the parameters $\alpha$, $\beta$, $\mu$ and
  $c$ of the estimated stable distribution. These were determined on the
  unbounded scale by EP-ABC and transformed back to the respective bounded
  scales.
  ](figures/stable_dist_densities_euro_in_sterling.png){#fig:densities-stable}

For reference purposes, the MLE of the parameters based on the same data 
was calculated with the R package `StableEstim`. The estimated parameters and
95% confidence intervals can be seen in @tbl:mle-estimates. 
Confidence intervals turned out to be narrow, except for $\beta$.

Table: MLE parameter values and 95% confidence intervals
  estimated by `StableEstim`. {#tbl:mle-estimates}

Parameter MLE     95% CI
--------- ------- -----------------
$\alpha$  1.92    (1.88, 1.97)
$\beta$   0.0581  (-0.398, 0.514)
$\mu$     0.0111  (-0.0127, 0.0349)
$c$       0.339   (0.327, 0.351)

@fig:hist-stable shows a comparison of the data to simulated values from a 
stable distribution. Distribution parameters were taken as those estimated
from EP-ABC and the MLE from `StableEstim`. It is clearly visible that samples
from the MLE solution match the data much more closely. The solution derived
from EP-ABC is located correctly but shows much less spread than the data.

![Histograms of data and simulated values. The parameter values for
  simulation were taken as the MAP estimate determined by EP-ABC (left) and
  the MLE estimate determined by the R package `StableEstim` (right).
  ](figures/histograms_euro_in_sterling.png){#fig:hist-stable}

# Conclusion

EP-ABC worked nicely for the example of the success probability 
of a binomial distribution. EP-ABC had a clear speed advantage over 
MCMC-ABC while producing exact samples during the whole estimation process.
However, EP-ABC showed a slightly heavier left tail and slight bias towards
the left. A possible explanation is a slight right-skewness in the analytical
posterior which cannot be by the Gaussian approximation.
Using the re-sampling scheme led to a noticable speed-up (results not shown).

It is not obvious to me why my implementation of EP-ABC for stable distributions
delivers substantially worse results than the MLE solution. For one, 
the estimated covariance matrix in @eq:ep-cov has very small entries.
Possibly, the global approximation constructed in EP-ABC, contracts too
quickly and this makes it difficult for new parameter values to be 
explored. However, further analysis of this problem would be necessary.

EP-ABC seems promising as a quick method to explore ABC problems. However,
the restriction to Gaussian posteriors might be too restrictive or lead
to biases for some problems.
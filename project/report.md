---
title: >
  Expectation Propagation in the framework of Approximate Bayesian Computation
author: Felix Held
date: 2018-05-01
documentclass: scrartcl
bibliography: refs.bib
biblio-title: References
biblio-style: authoryear
biblatexoptions:
  - backend=biber
header-includes:
  - \usepackage{mathpazo}
  - \usepackage{palatino}
  - \usepackage{microtype}
---

# Introduction

Approximate Bayesian Computation (ABC) aims to extend
Bayesian posterior inference to problems where no explicit likelihood function
is available or its computation is costly. Explicit evaluation of the
likelihood can be circumvented if an efficient simulation mechanism
is available.

The aim of this project was to investigate
the *Expectation Propagation* (EP) algorithm for likelihood-free
inference. This algorithm was originally described in @Barthelme2014a.

# Algorithm

EP in itself calculates an approximation to the posterior distribution.
It works by sampling


# Case-studies

## Success probability of a binomial distribution

## Tail and skew parameters of a stable distribution

# Conclusion


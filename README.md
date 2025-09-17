[![CI](https://github.com/esbenscriver/Estimate-matching-model/actions/workflows/ci.yml/badge.svg)](https://github.com/esbenscriver/Estimate-matching-model/actions/workflows/ci.yml)
[![CD](https://github.com/esbenscriver/Estimate-matching-model/actions/workflows/cd.yml/badge.svg)](https://github.com/esbenscriver/Estimate-matching-model/actions/workflows/cd.yml)

# Description
Estimate by maximum likelihood a one-to-one matching model with transferable utility where the choice probabilities of the agents on both sides of the matching market are given by the logit model and the transfers are assumed to be observed with a measurment error. See see e.g. [Andersen (2025)](https://arxiv.org/pdf/2409.05518) for a model description.

The model and estimator are implemented in JAX. We leverage the [SQUAREM](https://github.com/esbenscriver/squarem-JAXopt) accelerator to efficiently solve the systemt of fixed-point equations that characterize the equilibrium transfers. Finally, we rely on the [JAXopt](https://github.com/google/jaxopt) implementation of implicit differentiation when calculating the gradient of the log-likelihood function.

Let $\theta$ denote the parameters to be estimated. $\theta$ is estimated by maximum likelihood. Similar to [Rust (1987)](https://doi.org/10.2307/1911259) the estimation procedures via a nested fixed-point algorithm with an outer loop that search over different values of $\hat{\theta}$ to maximize the log-likelihood function, and an inner loop that for $\hat{\theta}$ solves for the equilibrium transfer and evaluates the log-likelihood function.
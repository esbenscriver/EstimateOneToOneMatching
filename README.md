[![CI](https://github.com/esbenscriver/EstimateOneToOneMatching/actions/workflows/ci.yml/badge.svg)](https://github.com/esbenscriver/EstimateOneToOneMatching/actions/workflows/ci.yml)
[![CD](https://github.com/esbenscriver/EstimateOneToOneMatching/actions/workflows/cd.yml/badge.svg)](https://github.com/esbenscriver/EstimateOneToOneMatching/actions/workflows/cd.yml)

# Description
Estimate by maximum likelihood a one-to-one matching model with transferable utility where the choice probabilities of the agents on both sides of the matching market are given by the logit model. The transfers are assumed to be observed with a measurment error. See see e.g. [Andersen (2025)](https://arxiv.org/pdf/2409.05518) for a model description.

The model and estimator are implemented in JAX. We leverage the [SQUAREM](https://github.com/esbenscriver/squarem-JAXopt) accelerator to efficiently solve the systemt of fixed-point equations that characterize the equilibrium transfers. Finally, we rely on the [JAXopt](https://github.com/google/jaxopt) implementation of implicit differentiation when calculating the gradient of the log-likelihood function automatically.

Let $\theta$ denote the parameters to be estimated. $\theta$ is estimated by maximum likelihood. Similar to [Rust (1987)](https://doi.org/10.2307/1911259) the estimation procedures via a nested fixed-point algorithm with an outer loop that search over different values of $\hat{\theta}$ to maximize the log-likelihood function, and an inner loop that for $\hat{\theta}$ solves for the equilibrium transfer and evaluates the log-likelihood function.

The full log-likelihood function is given by the sum of the log-likelihood of transfers, matched agents of type X, matched agents of type Y, unmatched agents of type X, and unmatched agents of type Y

$$
    \log L(\theta) = \log L_{t}(\theta) + \log L_{m}^{X}(\theta) + \log L_{m}^{Y}(\theta) + \log L_{u}^{X}(\theta) + \log L_{u}^{Y}(\theta).
$$

The log-likelihood of transfers are given in terms of the squared difference between the model consistent and observed transfer

$$
    \log L_t(\theta) = - \tfrac{1}{2XY} \sum_x^X \sum_y^Y \log((t^{*}_{xy}(\theta) - t_{xy})^2) ,
$$

the log-likelihood of the matched agents of type X is given as

$$
    \log L_{m}^{X}(\theta) = \sum_x^X \sum_y^Y n_{xy} \log p^{X}_{xy}(\theta),
$$

the log-likelihood of the matched agents of type Y is given as

$$
    \log L_{m}^{Y}(\theta) = \sum_x^X \sum_y^Y n_{xy} \log p^{Y}_{xy}(\theta),
$$

the log-likelihood of the unmatched agents of type X is given as

$$
    \log L_{u}^{X}(\theta) = \sum_x^X n_{x0} \log p^{X}_{x0}(\theta),
$$

the log-likelihood of the unmatched agents of type Y is given as

$$
    \log L_{u}^{Y}(\theta) = \sum_y^Y n_{0y} \log p^{Y}_{0y}(\theta).
$$
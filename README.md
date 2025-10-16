[![CI](https://github.com/esbenscriver/EstimateOneToOneMatching/actions/workflows/ci.yml/badge.svg)](https://github.com/esbenscriver/EstimateOneToOneMatching/actions/workflows/ci.yml)
[![CD](https://github.com/esbenscriver/EstimateOneToOneMatching/actions/workflows/cd.yml/badge.svg)](https://github.com/esbenscriver/EstimateOneToOneMatching/actions/workflows/cd.yml)

# Estimate one-to-one matching model
This package estimate by maximum likelihood a one-to-one matching model with transferable utility where the choice probabilities of the agents on both sides of the matching market are given by the logit model. The estimator assumes that the distribution of transfers and matches are observed.

The model and estimator are implemented in JAX. We leverage the [SQUAREM](https://github.com/esbenscriver/squarem-JAXopt) accelerator to efficiently solve the system of fixed-point equations that characterize the equilibrium transfers. Finally, we rely on the [JAXopt](https://github.com/google/jaxopt) implementation of implicit differentiation when calculating the gradient of the log-likelihood function automatically.

## Model description
The matching market consists of agent of type X and Y on both sides of the market. Each agent choose who they want to match with. The deterministic match-specific payoffs of the agents of type X and Y are given as

$$
    v^{X}_{xy} = z^{X}_{xy} \beta^{X} + t_{xy},
$$
$$
    v^{Y}_{xy} = z^{Y}_{xy} \beta^{Y} - t_{xy},
$$

where $t_{xy}$ is a match-specific transfer from agent y to agent x. The corresponding choice probabilities are given by the logit expressions

$$
    p^{X}_{xy}(v^{X}_{x \cdot}) = \frac{\exp{(v^{X}_{xy}/\sigma^{X})}}{1 + \sum_{j} \exp{(v^{X}_{xj}/\sigma^{X})}}, 
$$
$$
    p^{Y}_{xy}(v^{Y}_{\cdot y}) = \frac{\exp{(v^{Y}_{xy}/\sigma^{Y})}}{1 + \sum_{i} \exp{(v^{Y}_{iy}/\sigma^{Y})}},
$$

where $\sigma^{X}$ and $\sigma^{Y}$ are scale parameters. Note that for identification the deterministic payoffs of being unmatched is normalized to zero

$$
    v^{X}_{x0} = v^{Y}_{0y} = 0.
$$

In turn, the choice probabilities of being unmatched are given as

$$
    p^{X}_{x0}(v^{X}_{x \cdot}) = \frac{1}{1 + \sum_{j} \exp{(v^{X}_{xj}/\sigma^{X})}},
$$

$$
    p^{Y}_{0y}(v^{Y}_{\cdot y}) = \frac{1}{1 + \sum_{i} \exp{(v^{Y}_{iy}/\sigma^{Y})}}.
$$ 

Finally, the transfers, $t_{xy}$, are determined from a set of market clearing conditions

$$
    n^{X}_{x} p^{X}_{xy}(v^{X}_{x \cdot}) = n^{Y}_{y} p^{Y}_{xy}(v^{Y}_{\cdot y}),
$$

where 

$$
    (n^{X}_{x}, n^{Y}_{y}),
$$

are the marginal distribution of agents of type X and Y. The distribution of equilibrium transfers can be determined from a system of fixed-point equations

$$
    t_{xy} = t_{xy} + \tfrac{\sigma^{X}\sigma^{Y}}{\sigma^{X} + \sigma^{Y}} \log \left( \frac{ n^{Y}_{y} p^{Y}_{xy} } { n^{X}_{x} p^{X}_{xy} } \right),
$$

that can be shown to be a contraction mapping, see [Andersen (2025)](https://arxiv.org/pdf/2409.05518). Hence, iterating on this expression is guaranteed to converge to an unique solution, $t^{*}_{xy}$.

## Maximum likelihood estimator
Let $\theta = (\beta^X, \beta^Y,\sigma^X,\sigma^Y)$ denote the vector of parameters to be estimated and let $\theta_{0}$ denote the true but unobserved vector of parameter values. $\theta$ is estimated by maximum likelihood, where transfers are assumed to be observed with an iid normal distributed measurement error, $\varepsilon_{xy} \sim \mathcal{N}(\mu,\sigma^{2})$,  

$$
    \tilde{t}_{xy} = t^{*}_{xy}(\theta_{0}) + \varepsilon_{xy}.
$$

We will use the well known fact that $\mu$ and $\sigma^{2}$ can be concentrated out of the log-likelihood function

$$
    \hat{\varepsilon}_{xy}(\theta) = \tilde{t}_{xy} - t^{*}_{xy}(\theta),
$$
$$
    \hat{\mu}(\theta) = \tfrac{1}{XY} \sum_{x}^{X} \sum_{y}^{Y} \hat{\varepsilon}_{xy}(\theta),
$$
$$
    \hat{\sigma}^{2}(\theta) = \tfrac{1}{XY} \sum_{x}^{X} \sum_{y}^{Y} \left(\hat{\varepsilon}_{xy}(\theta) - \hat{\mu}(\theta) \right)^2.
$$

The full log-likelihood function is given by the sum of the log-likelihood of transfers, matched and unmatched agents of type X, and matched and unmatched agents of type Y

$$
    \max_{\theta} \log L(\theta) = \log L_{t}(\theta) + \log L_{m}^{X}(\theta) + \log L_{m}^{Y}(\theta),
$$

where the log-likelihood of transfers is proportional to the variance of the measurement errors,

$$
    \log L_t(\theta) = - \tfrac{XY}{2} \log \hat{\sigma}^{2}(\theta),
$$

the log-likelihood of the matched and unmatched agents of type X is given as the negative Kullback-Leibler divergence between the observed choices, $(m_{x0},m_{xy})$, and the model consistent choice probabilities of agents of type X

$$
    \log L_{m}^{X}(\theta) = \sum_{x}^{X}\left[ m_{x0} \log p^{X}_{x0}(\theta) + \sum_{y}^{Y} m_{xy} \log p^{X}_{xy}(\theta) \right],
$$

and the log-likelihood of the matched and unmatched agents of type Y is given as the negative Kullback-Leibler divergence between the observed choices, $(m_{0y},m_{xy})$, and the model consistent choice probabilities of agents of type Y

$$
    \log L_{m}^{Y}(\theta) = \sum_{y}^{Y}\left[ m_{0y} \log p^{Y}_{0y}(\theta) + \sum_{x}^{X} m_{xy} \log p^{Y}_{xy}(\theta) \right].
$$


Similar to [Rust (1987)](https://doi.org/10.2307/1911259) the estimation procedures via a nested fixed-point algorithm with an outer loop that search over different values of $\theta$ to maximize the full log-likelihood function, $\log L(\theta)$, and an inner loop that for $\theta$ solves for the equilibrium transfer, $t^{*}_{xy}(\theta)$.
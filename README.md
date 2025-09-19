[![CI](https://github.com/esbenscriver/EstimateOneToOneMatching/actions/workflows/ci.yml/badge.svg)](https://github.com/esbenscriver/EstimateOneToOneMatching/actions/workflows/ci.yml)
[![CD](https://github.com/esbenscriver/EstimateOneToOneMatching/actions/workflows/cd.yml/badge.svg)](https://github.com/esbenscriver/EstimateOneToOneMatching/actions/workflows/cd.yml)

# Description
Estimate by maximum likelihood a one-to-one matching model with transferable utility where the choice probabilities of the agents on both sides of the matching market are given by the logit model. See see e.g. [Andersen (2025)](https://arxiv.org/pdf/2409.05518) for a model description.

The model and estimator are implemented in JAX. We leverage the [SQUAREM](https://github.com/esbenscriver/squarem-JAXopt) accelerator to efficiently solve the systemt of fixed-point equations that characterize the equilibrium transfers. Finally, we rely on the [JAXopt](https://github.com/google/jaxopt) implementation of implicit differentiation when calculating the gradient of the log-likelihood function automatically.

The match-specific deterministic payoffs of the agents of type X and Y are given as

$$
    v^{X}_{xy} = z^{X}_{xy} \beta^{X} + t_{xy}, \\\\
    v^{Y}_{xy} = z^{Y}_{xy} \beta^{Y} - t_{xy},
$$

and the correponding choice probabilities are given by the logit expressions

$$
    p^{X}_{xy}(v^{X}_{x \cdot}) = \frac{\exp{(v^{X}_{xy})}}{1 + \sum_{j} \exp{(v^{X}_{xj})}}, \\\\
    p^{Y}_{xy}(v^{Y}_{\cdot y}) = \frac{\exp{(v^{Y}_{xy})}}{1 + \sum_{i} \exp{(v^{Y}_{iy})}}.
$$

Note thate the deterministic payoffs of the outside options are normalized to zero, $v^{X}_{x0}=v^{Y}_{0y}=0$. 

Finally, the transfers, $t_{xy}$, are determined from a set of market clearing conditions

$$
    n^{X}_{x} p^{X}_{xy}(v^{X}_{xy}) = n^{Y}_{y} p^{Y}_{xy}(v^{Y}_{xy}),
$$


where $(n^{X}_{x}, n^{Y}_{y})$ are the marginal distribution of agents of type X and Y.


Let $\theta \in (\beta^X,beta^Y)$ denote the parameters to be estimated. $\theta$ is estimated by maximum likelihood, where transfers are assumed to be observed with a iid normal distributed measurment error, $\varepsilon_{xy} \sim \mathcal{N}(0,\sigma^{2})$  

$$
    t_{xy}(\theta) = t^{*}_{xy}(\theta) + \varepsilon_{xy}.
$$

Similar to [Rust (1987)](https://doi.org/10.2307/1911259) the estimation procedures via a nested fixed-point algorithm with an outer loop that search over different values of $\hat{\theta}$ to maximize the log-likelihood function, and an inner loop that for $\hat{\theta}$ solves for the equilibrium transfer, $t^{*}_{xy}(\hat{\theta})$, and evaluates the full log-likelihood function, $\log L(\hat{\theta})$.

The full log-likelihood function is given by the sum of the log-likelihood of transfers, matched agents of type X, matched agents of type Y, unmatched agents of type X, and unmatched agents of type Y

$$
    \log L(\theta) = \log L_{t}(\theta) + \log L_{m}^{X}(\theta) + \log L_{m}^{Y}(\theta) + \log L_{u}^{X}(\theta) + \log L_{u}^{Y}(\theta).
$$

The log-likelihood of transfers are given in terms of the squared difference between the model consistent equilibrium transfer and the observed transfer,

$$
    \log L_t(\theta) = - \tfrac{XY}{2} \log \left(\tfrac{1}{XY} \sum_x^X \sum_y^Y \left(t^{*}_{xy}(\theta) - t_{xy}\right)^2 \right) ,
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
    \log L_{u}^{Y}(\theta) = \sum_y^Y n_{0y} \log p^{Y}_{0y}(\theta),
$$

where 

$$
    \left(p^{X}_{xy}(\theta), p^{Y}_{xy}(\theta), p^{X}_{x0}(\theta), p^{Y}_{0y}(\theta)\right)
$$ 

are the choice probabilities of agents of type X and Y consistent with $t^{*}_{xy}(\theta)$.
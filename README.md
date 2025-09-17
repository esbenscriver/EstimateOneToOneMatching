[![CI](https://github.com/esbenscriver/Estimate-matching-model/actions/workflows/ci.yml/badge.svg)](https://github.com/esbenscriver/Estimate-matching-model/actions/workflows/ci.yml)
[![CD](https://github.com/esbenscriver/Estimate-matching-model/actions/workflows/cd.yml/badge.svg)](https://github.com/esbenscriver/Estimate-matching-model/actions/workflows/cd.yml)

# Description
Estimate by maximum likelihood a one-to-one matching model with transferable utility where the choice probabilities of the agents on both sides of the matching market are given by the logit model and the transfers are assumed to be observed with a measurment error. See see e.g. [Andersen (2025)](https://arxiv.org/pdf/2409.05518) for a model description.

The model and estimator are implemented in JAX. We leverage the [SQUAREM](https://github.com/esbenscriver/squarem-JAXopt) accelerator to efficiently solve the systemt of fixed-point equations that characterize the equilibrium transfers. Finally, we rely on the [JAXopt](https://github.com/google/jaxopt) implementation of implicit differentiation when calculating the gradient of the log-likelihood function.

### Matching model
The economy consists of agents of type X and Y that face a discrete choice of who to match with. Let $\mathscr{X}$ denote the set of unique types of agents of type X, and let $\mathscr{Y}$ denote the set of unique types of agents of type Y. We will assume that the number of types, $\left|\mathscr{X}\right|$
and $\left|\mathscr{Y}\right|$, are finite.

Let $\mathscr{Y}_0$ denote the full choice set of the agents of type X. The agent $a$ of type $x$ face the discrete choice of remain unmatched, $y=0$, or match with one of the $|\mathscr{Y}|$ types of agents, $y=1,...,|\mathscr{Y}|$,
\begin{equation}
    \max_{y \in \mathscr{Y}_0} \bigg\{ v_{xy}^X + \sigma_x^X \varepsilon_{axy}^X \bigg\},
\end{equation}
where $v_{xy}^X$ and $\varepsilon_{axy}^X$ are the deterministic and stochastic part of the payoff function, and $\sigma_x^X > 0$ is the scale parameter. If the agent chooses to remain unmatched, the agent derives a deterministic payoff of zero. When the agent of type $x$ chooses to match with an agent of type $y$, the agent's deterministic payoff is given by the sum of the match-specific utility, $\beta_{xy}^X$, and transfer, $t_{xy}$,
\begin{aligned}
    v_{x0}^X &= 0, \\
    v_{xy}^X &= \beta_{xy}^X + t_{xy} \; \forall \; y \in \mathscr{Y}.
\end{aligned}
As we assume, that there exist a continuum mass of each type of agents, the share of agents of type $x$ matched with a agent of type $y$ is a function $p_{xy}^X: \mathbb{R}^{|\mathscr{Y}|} \rightarrow \mathbb{R}$ of the vector of wages, that the agents face, $t_{x \cdot}=(t_{x1},\cdots,t_{x|\mathscr{Y}|})$  
\begin{equation}
    p_{xy}^X(t_{x \cdot})
    = Pr\bigg[ y = \underset{j\in \mathscr{Y}_0}{\operatorname{argmax}} \bigg\{v_{xj}^X + \sigma_x^X \varepsilon_{axj}^X \bigg\} \Big|t_{x \cdot} \bigg] \; \forall \; y \in \mathscr{Y}_0.
\end{equation}

$\mathscr{X}_0$ denotes the full choice set of the agents of type Y.
$$
    \max_{x \in \mathscr{X}_0} \bigg\{ v_{xy}^Y + \sigma_y^Y \varepsilon_{bxy}^Y \bigg\}.
$$
When the agent of type $y$ chooses to match with an agent of type $x$, the agent's deterministic payoff is given by the the match-specific utility, $\beta_{xy}^Y$, minus the match-specific transfer, $t_{xy}$
\begin{aligned}
    v_{0y}^Y &= 0, \\
    v_{xy}^Y &= \beta_{xy}^Y - t_{xy} \; \forall \; x \in \mathscr{X}.
\end{aligned}
The share of agents of type $y$ matched with agents of type $x$ is a function of the vector of transfers that the agent faces, $p_{xy}^Y: \mathbb{R}^{|\mathscr{X}|} \rightarrow \mathbb{R}$.  
\begin{equation}
    p_{xy}^Y(t_{\cdot y})
    = Pr\bigg[ x = \underset{i \in \mathscr{X}_0}{\operatorname{argmax}} \bigg\{v_{iy}^Y + \sigma_y^Y \varepsilon_{biy}^X \bigg\} \Big|t_{\cdot y} \bigg] \; \forall \; y \in \mathscr{X}_0.
\end{equation}
where $t_{\cdot y}=(t_{1y},\cdots,t_{|\mathscr{X}|y})$. Following the literature on discrete choice modeling, $p_{xy}^X$ and $p_{xy}^Y$ will be refer to market shares as the choice probabilities of the agents of type X and y.

The discrete choices of the agents of type X and Y are connected through the transfers that are determined in a competitive equilibrium. Let the mass of agents of type $x$ be denoted by $n_x^X$, and the mass of agents of type $y$ be denoted $n_y^Y$. The vector of equilibrium transfers, $T^*=(t_{11}^*,\cdots,t_{1|\mathscr{Y}|}^*,\cdots,t_{|\mathscr{X}|1}^*,\cdots,t_{|\mathscr{X}||\mathscr{Y}|}^*)$, and the vector of equilibrium matches, $\mu=(\mu_{11},\cdots,\mu_{1|\mathscr{Y}|},\cdots,\mu_{|\mathscr{X}|1},\cdots,\mu_{|\mathscr{X}|\mathscr{Y}|})$, are jointly determined from a set of market clearing conditions, such that agents of type x's demand for agent of type y equate the agents of type y's demand for agents of type x across all combinations of $x$ and $y$,
$$
    \mu_{xy}(W^*) = n_x^X p_{xy}^X(t_{x \cdot}^*) = n_y^Y p_{xy}^Y(t_{\cdot y}^*) \; \forall \; (x,y) \in \mathscr{X} \times \mathscr{Y}.
$$
We find the equilibrium transfer by applying the following iterative algorithm
$$
    t_{xy}^{k+1} = t_{xy}^{k} + \tfrac{c_{xy}^X\sigma_x^X c_{xy}^Y \sigma_y^Y}{c_{xy}^X\sigma_x^X + c_{xy}^Y \sigma_y^Y} \log \left[ \frac{n_y^Y p_{xy}^Y(t_{\cdot y}^{k})}{n_x^X p_{xy}^X(t_{x \cdot}^{k})} \right] \; \forall \; (x,y) \in \mathscr{X} \times \mathscr{Y}.
$$

### Estimator
The model is estimated by maximum likelihood. Similar to [Rust (1987)](https://doi.org/10.2307/1911259) the estimation procedures via a nested fixed-point algorithm with an outer loop that search over different values of $\hat{\theta}$, and an inner loop that for $\hat{\theta}$ solves for the equilibrium transfer.
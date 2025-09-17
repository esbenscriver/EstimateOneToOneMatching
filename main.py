"""
Solve and estimate a one-to-one matching model
"""

# import JAX
import jax
import jax.numpy as jnp
from jax import random

# import solver for one-to-one matching model
from estimate_matching_model.matching_model import MatchingModel, ObservedData

from tabulate import tabulate

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

types_X, types_Y = 400, 200
number_of_parameters_X, number_of_parameters_Y = 2, 3

# Simulate covariates of the agents' utility function
covariates_X = -random.uniform(
    key=random.PRNGKey(111), shape=(types_X, types_Y, number_of_parameters_X)
)
covariates_Y = random.uniform(
    key=random.PRNGKey(112), shape=(types_X, types_Y, number_of_parameters_Y)
)

# Simulate parameters of the agents' utility function
parameters = random.uniform(
    key=random.PRNGKey(113), shape=(number_of_parameters_X + number_of_parameters_Y,)
)

# Simulate marginal distribution of agents
marginal_distribution_X = random.uniform(key=random.PRNGKey(114), shape=(types_X, 1))
marginal_distribution_Y = random.uniform(key=random.PRNGKey(115), shape=(1, types_Y))

# Solve a matching model with logit demand
model = MatchingModel(
    covariates_X=covariates_X,
    covariates_Y=covariates_Y,
    marginal_distribution_X=marginal_distribution_X,
    marginal_distribution_Y=marginal_distribution_Y,
)

utility_X, utility_Y = model.Utilities_of_agents(params=parameters)

transfer = model.solve(utility_X=utility_X, utility_Y=utility_Y)

pX_xy, pX_x0 = model.ChoiceProbabilities_X(transfer, utility_X)
pY_xy, pY_0y = model.ChoiceProbabilities_Y(transfer, utility_Y)

# Simulate data
sigma = 1.0
measurement_errors = sigma * random.normal(random.PRNGKey(211), shape=transfer.shape)
observed_transfer = transfer + measurement_errors

data = ObservedData(
    transfer=observed_transfer,
    matched=model.marginal_distribution_X * pX_xy,
    unmatched_X=model.marginal_distribution_X * pX_x0,
    unmatched_Y=model.marginal_distribution_Y * pY_0y,
)

guess = jnp.zeros_like(parameters)

# Estimate parameters by maximum likelihood estimation
parameter_estimates = model.fit(guess, data, verbose=True)

# Combine into rows
table = list(zip(parameters, parameter_estimates))

# Print with headers
print(tabulate(table, headers=["True parameters", "Estimated parameters"], tablefmt="grid"))



"""
Solve and estimate a one-to-one matching model
"""

# import JAX
import jax
import jax.numpy as jnp
from jax import random

# import solver for one-to-one matching model
from estimate_matching_model.matching_model import MatchingModel, ObservedData

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

# Set number of types of agents on both sides of the market
types_X, types_Y = 40, 60

number_of_parameters_X, number_of_parameters_Y = 2, 3

# Simulate choice-specific utilities
covariates_X = -random.uniform(
    key=random.PRNGKey(111), shape=(types_X, types_Y, number_of_parameters_X)
)
covariates_Y = random.uniform(
    key=random.PRNGKey(112), shape=(types_X, types_Y, number_of_parameters_Y)
)

# Simulate choice-specific utilities
parameters = random.uniform(
    key=random.PRNGKey(113), shape=(number_of_parameters_X + number_of_parameters_Y,)
)

# Simulate distribution of agents
marginal_distribution_X = random.uniform(key=random.PRNGKey(115), shape=(types_X, 1))
marginal_distribution_Y = random.uniform(key=random.PRNGKey(116), shape=(1, types_Y))

print("-----------------------------------------------------------------------")
print("Solve a matching model with logit demand:")

model = MatchingModel(
    covariates_X=covariates_X,
    covariates_Y=covariates_Y,
    marginal_distribution_X=marginal_distribution_X,
    marginal_distribution_Y=marginal_distribution_Y,
)

utility_X, utility_Y = model.Utilities_of_agents(params=parameters)

transfer = model.solve(utility_X=utility_X, utility_Y=utility_Y, verbose=True)

pX_xy, pX_x0 = model.ChoiceProbabilities_X(transfer, utility_X)
pY_xy, pY_0y = model.ChoiceProbabilities_Y(transfer, utility_Y)

data = ObservedData(
    transfer=transfer,
    matched=model.marginal_distribution_X * pX_xy,
    unmatched_X=model.marginal_distribution_X * pX_x0,
    unmatched_Y=model.marginal_distribution_Y * pY_0y,
)

assert jnp.allclose(data.matched, model.marginal_distribution_Y * pY_xy), (
    "demand do not match"
)

guess = jnp.zeros_like(parameters)

parameter_estimates = model.fit(guess, data)
print(f"{parameters = }")
print(f"{parameter_estimates = }")

print(f"negative log-lik (true): {model.neg_log_likelihood(parameters, data)}")
print(
    f"negative log-lik (estimated): {model.neg_log_likelihood(parameter_estimates, data)}"
)

assert jnp.allclose(parameter_estimates, parameters), (
    "true parameters and estimated parameters do no match"
)

"""
Solve and estimate a one-to-one matching model
"""

# import JAX
import jax
import jax.numpy as jnp
from jax import random

# import solver for one-to-one matching model
from estimate_matching_model.matching_model import MatchingModel, Data

from tabulate import tabulate

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

types_X, types_Y = 200, 300
number_of_parameters_X, number_of_parameters_Y = 2, 3

parameter_names_X = [f"beta_X ({x})" for x in range(number_of_parameters_X)]
parameter_names_Y = [f"beta_Y ({y})" for y in range(number_of_parameters_Y)]
parameter_names = (
    parameter_names_X + parameter_names_Y + ["log(scale_X)", "log(scale_Y)"]
)

# Simulate covariates of the agents' utility function
covariates_X = -random.uniform(
    key=random.PRNGKey(111), shape=(types_X, types_Y, number_of_parameters_X)
)
covariates_Y = random.uniform(
    key=random.PRNGKey(112), shape=(types_X, types_Y, number_of_parameters_Y)
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

# Simulate parameters of the agents' utility function
parameters = random.uniform(
    key=random.PRNGKey(211),
    shape=(number_of_parameters_X + number_of_parameters_Y + 2,),
)

solution = model.predict(params=parameters)

# Simulate data
mu, sigma = 0.0, 1.0
measurement_errors = mu + sigma * random.normal(
    random.PRNGKey(311), shape=solution.transfer.shape
)

data = Data(
    transfer=solution.transfer + measurement_errors,
    matched=solution.matched,
    unmatched_X=solution.unmatched_X,
    unmatched_Y=solution.unmatched_Y,
)

guess = jnp.zeros_like(parameters)

parameter_estimates = model.fit(guess, data, verbose=True)

log_lik = -model.neg_log_likelihood(parameter_estimates, data)

table_estimates = tabulate(
    list(zip(parameter_names, parameters, parameter_estimates)),
    headers=["names", "True parameters", "Estimated parameters"],
    tablefmt="grid",
)

print(f"\n{table_estimates}")
print(f"log-likelihood value: {log_lik:.4f}\n")

predictions = model.predict(params=parameter_estimates)

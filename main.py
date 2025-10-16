"""
Solve and estimate a one-to-one matching model
"""

# import JAX
import jax
import jax.numpy as jnp
from jax import random

# import solver for one-to-one matching model
from estimate_matching_model.matching_model import MatchingModel, Data
from estimate_matching_model.segment_matching_model import MatchingModel as segment_MatchingModel

from tabulate import tabulate
import time

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)

types_X, types_Y = 200, 300
number_of_parameters_X, number_of_parameters_Y = 2, 3

parameter_names_X = [f"beta_X ({x})" for x in range(number_of_parameters_X)]
parameter_names_Y = [f"beta_Y ({y})" for y in range(number_of_parameters_Y)]
parameter_names = parameter_names_X + parameter_names_Y + ["scale_X", "scale_Y"]

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

start = time.time()
solution = model.predict(params=parameters)
print(f"time: {time.time() - start} sec.")

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
log_lik0 = -model.neg_log_likelihood(guess, data)
print(f"log-likelihood value: {log_lik0:.6f}\n")

estimates_unrestricted = model.fit(guess, data, verbose=True)

log_lik = -model.neg_log_likelihood(estimates_unrestricted, data)

estimates_restricted = model.restricted_parameters(estimates_unrestricted)
parameters_restricted = model.restricted_parameters(parameters)

table_estimates = tabulate(
    list(zip(parameter_names, parameters_restricted, estimates_restricted)),
    headers=["names", "True parameters", "Estimated parameters"],
    tablefmt="grid",
    floatfmt=".4f",
)
print(f"\n{table_estimates}")
print(f"log-likelihood value: {log_lik:.4f}\n")

predictions = model.predict(params=estimates_restricted)

mp = model.extract_parameters(parameters)
utility_X, utility_Y = model.Utilities_of_agents(mp)
utility_X_ravel, utility_Y_ravel = utility_X.ravel(), utility_Y.ravel()
transfer_ravel = model.UpdateTransfers(solution.transfer,utility_X,utility_Y,mp).ravel()

type_idx_X = jnp.tile(jnp.arange(types_X)[:, jnp.newaxis], (1, types_Y)).ravel()
type_idx_Y = jnp.tile(jnp.arange(types_Y)[:, jnp.newaxis], (1, types_X)).T.ravel()

segment_model = segment_MatchingModel(
    covariates_X=covariates_X.reshape((types_X * types_Y, number_of_parameters_X)),
    covariates_Y=covariates_Y.reshape((types_X * types_Y, number_of_parameters_Y)),
    marginal_distribution_X=marginal_distribution_X.squeeze(),
    marginal_distribution_Y=marginal_distribution_Y.squeeze(),
    types_idx_X=type_idx_X,
    types_idx_Y=type_idx_Y,
)

utility_X_segment, utility_Y_segment = segment_model.Utilities_of_agents(mp)
print(f"{utility_X_ravel.shape = }, {utility_Y_ravel.shape = }")
print(f"{utility_X_segment.shape = }, {utility_Y_segment.shape = }")
print(f"{jnp.allclose(utility_X_ravel, utility_X_segment)}")
print(f"{jnp.allclose(utility_Y_ravel, utility_Y_segment)}")

transfer_ravel = model.UpdateTransfers(solution.transfer,utility_X,utility_Y,mp).ravel()
transfer_segment = segment_model.UpdateTransfers(solution.transfer.ravel(),utility_X_segment,utility_Y_segment,mp)
print(f"{jnp.allclose(transfer_ravel, transfer_segment)}")
print(f"{data.unmatched_X.shape = }, {data.unmatched_Y.shape = }")

segment_data = Data(
    transfer=data.transfer.ravel(),
    matched=data.matched.ravel(),
    unmatched_X=data.unmatched_X.ravel(),
    unmatched_Y=data.unmatched_Y.ravel(),
)

log_lik0 = -segment_model.neg_log_likelihood(guess, segment_data)
print(f"log-likelihood value: {log_lik0:.6f}\n")

segment_estimates_unrestricted = segment_model.fit(guess, segment_data, verbose=True)

log_lik = -segment_model.neg_log_likelihood(segment_estimates_unrestricted, segment_data)

segment_estimates_restricted = segment_model.restricted_parameters(segment_estimates_unrestricted)
parameters_restricted = segment_model.restricted_parameters(parameters)

table_estimates = tabulate(
    list(zip(parameter_names, parameters_restricted, segment_estimates_restricted)),
    headers=["names", "True parameters", "Estimated parameters"],
    tablefmt="grid",
    floatfmt=".4f",
)
print(f"\n{table_estimates}")
print(f"log-likelihood value: {log_lik:.4f}\n")

start = time.time()
segment_solution = segment_model.predict(params=parameters)
print(f"time: {time.time() - start} sec.")

# import JAX
import jax
import jax.numpy as jnp
from jax import random

# import solver for one-to-one matching model
from estimate_matching_model.matching_model import MatchingModel

import pytest

# Increase precision to 64 bit
jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "types_X, types_Y, number_of_parameters_X, number_of_parameters_Y",
    [
        (100, 200, 20, 30),
    ],
)
def test_mle(types_X, types_Y, number_of_parameters_X, number_of_parameters_Y):
    # Simulate choice-specific utilities
    covariates_X = -random.uniform(
        key=random.PRNGKey(111), shape=(types_X, types_Y, number_of_parameters_X)
    )
    covariates_Y = random.uniform(
        key=random.PRNGKey(112), shape=(types_X, types_Y, number_of_parameters_Y)
    )

    # Simulate distribution of agents
    marginal_distribution_X = random.uniform(
        key=random.PRNGKey(113), shape=(types_X, 1)
    )
    marginal_distribution_Y = random.uniform(
        key=random.PRNGKey(114), shape=(1, types_Y)
    )

    # Simulate parameters
    parameters = random.uniform(
        key=random.PRNGKey(115),
        shape=(number_of_parameters_X + number_of_parameters_Y + 2,),
    )

    model = MatchingModel(
        covariates_X=covariates_X,
        covariates_Y=covariates_Y,
        marginal_distribution_X=marginal_distribution_X,
        marginal_distribution_Y=marginal_distribution_Y,
    )

    solution = model.predict(params=parameters)

    guess = jnp.zeros_like(parameters)

    parameter_estimates = model.fit(guess, solution, verbose=False)

    assert jnp.allclose(parameter_estimates, parameters), (
        f"true parameters and estimated parameters do no match:\n{parameter_estimates = }\n{parameters = }"
    )

"""
JAX implementation of fixed-point iteration algoritm to solve and one-to-one matching model with transferable utility

Reference:
Esben Scriver Andersen, Note on solving one-to-one matching models with linear transferable utility, 2025 (https://arxiv.org/pdf/2409.05518)
"""

import jax
import jax.numpy as jnp
from jax import Array

# import simple_pytree (used to store variables)
from simple_pytree import Pytree, dataclass

# import solvers
from jaxopt import FixedPointIteration, AndersonAcceleration, LBFGS
from squarem_jaxopt import SquaremAcceleration

SolverTypes = (
    type[SquaremAcceleration] | type[AndersonAcceleration] | type[FixedPointIteration]
)


@dataclass
class ObservedData(Pytree, mutable=False):
    """Observed data used for maximum likelihood estimation

    Attributes:
        transfer (Array): observed transfers between matched agents
        matched (Array): observed numbers of matched agents
        unmatched_X (Array): observed numbers of unmatched agents of type X
        unmatched_Y (Array): observed numbers of unmatched agents of type Y
    """

    transfer: Array
    matched: Array
    unmatched_X: Array
    unmatched_Y: Array


@dataclass
class MatchingModel(Pytree, mutable=False):
    """Matching model

    Attributes:
        covariates_X (Array): covariates of utility function of agents of type X
        covariates_Y (Array): covariates of utility function of agents of type Y
        marginal_distribution_X (Array): marginal distribution of agents of type X
        marginal_distribution_Y (Array): marginal distribution of agents of type Y
    """

    covariates_X: Array
    covariates_Y: Array

    marginal_distribution_X: Array
    marginal_distribution_Y: Array

    def ChoiceProbabilities(self, v: Array, axis: int) -> tuple[Array, Array]:
        """Compute the logit choice probabilities for inside and outside options

        Args:
            v (Array): choice-specific payoffs
            axis (int): axis that describes the choice set

        Returns:
        P_inside (Array):
            choice probabilities of inside options.
        P_outside (Array):
            choice probabilities of outside option.
        """
        # v_max = jnp.max(v, axis=axis, keepdims=True)

        # exponentiated centered payoffs of inside options
        nominator = jnp.exp(v)

        # denominator of choice probabilities
        denominator = 1 + jnp.sum(nominator, axis=axis, keepdims=True)
        return nominator / denominator, 1 / denominator

    def Utility(self, covariates: Array, parameters: Array) -> Array:
        """Computes match-specific utilities

        Args:
            covariates (Array): covariates of utility function
            parameters (Array): parameters of utility function

        Returns:
            demand (Array): match-specific utilities
        """
        return jnp.einsum("ijk, k -> ij", covariates, parameters)

    def ChoiceProbabilities_X(self, transfer: Array, utility_X: Array) -> tuple[Array, Array]:
        """Computes choice probabilities of agents of type X

        Args:
            transfer (Array): match-specific transfers
            utilities (Array): match-specific utilities for agents of type X

        Returns:
            ChoiceProbabilities (Array): match-specific choice probabilities for agents of type X
        """
        v_X = jax.lax.add(utility_X, transfer)
        return self.ChoiceProbabilities(v_X, axis=1)

    def ChoiceProbabilities_Y(
        self, transfer: Array, utility_Y: Array
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Computes choice probabilities of agents of type Y

        Args:
            transfer (Array): match-specific transfers
            utilities (Array): match-specific utilities for agents of type Y

        Returns:
            ChoiceProbabilities (Array): match-specific choice probabilities for agents of type Y
        """
        v_Y = jax.lax.sub(utility_Y, transfer)
        return self.ChoiceProbabilities(v_Y, axis=0)

    def Demand_X(self, transfer: Array, utility_X: Array) -> Array:
        """Computes agents of type X's demand for agents of type Y

        Args:
            transfer (Array): match-specific transfers
            utility_X (Array): match-specific utilities

        Returns:
            demand (Array): demand for inside options
        """
        return (
            self.marginal_distribution_X
            * self.ChoiceProbabilities_X(transfer, utility_X)[0]
        )

    def Demand_Y(self, transfer: Array, utility_Y: Array) -> Array:
        """Computes agents of type Y's demand for agents of type X

        Args:
            transfer (Array): match-specific transfers
            utility_Y (Array): match-specific utilities

        Returns:
            demand (Array): demand for inside options
        """
        return (
            self.marginal_distribution_Y
            * self.ChoiceProbabilities_Y(transfer, utility_Y)[0]
        )

    def UpdateTransfers(
        self,
        t_initial: Array,
        utility_X: Array,
        utility_Y: Array,
    ) -> Array:
        """Updates fixed point equation for transfers

        Args:
            t_initial (Array): initial transfers
            utility_X (Array): utility of agents of type X
            utility_Y (Array): utility of agents of type Y

        Returns:
            t_updated (Array): updated transfers
        """
        # Calculate demand for both sides of the market
        demand_X = self.Demand_X(t_initial, utility_X)  # type X's demand for type Y
        demand_Y = self.Demand_Y(t_initial, utility_Y)  # type Y's demand for type X

        # Update transfer
        t_updated = t_initial + 1 / 2 * jnp.log(demand_Y / demand_X)
        return t_updated

    def solve(
        self,
        utility_X: Array,
        utility_Y: Array,
        fixed_point_solver: SolverTypes = SquaremAcceleration,
        tol: float = 1e-10,
        maxiter: int = 1000,
        verbose: bool = False,
    ) -> Array:
        """Solve for equilibrium transfer

        Args:
            utility_X (Array): utilities of agents of type X
            utility_Y (Array): utilities of agents of type Y
            fixed_point_solver (SolverTypes): solver used for solving fixed point equation (FixedPointIteration, AndersonAcceleration, SquaremAcceleration)
            tol (float): stopping tolerance for step length of fixed-point iterations, x_{i+1} - x_{i}
            maxiter (int): maximum number of iterations
            verbose (bool): whether to print information on every iteration or not.

        Returns:
            transfers (Array): equilibrium transfers
        """
        # Initial guess for equilibrium transfers
        transfer_init = jnp.zeros(self.covariates_X.shape[:-1])

        # Find equilibrium transfers
        result = fixed_point_solver(
            self.UpdateTransfers,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
        ).run(transfer_init, utility_X, utility_Y)
        return result.params

    def Utilities_of_agents(self, params: Array) -> tuple[Array, Array]:
        """Compute match-specific utilities for agents of type X and Y

        Args:
            params (Array): parameters of agents' utility functions

        Returns:
        utility_X (Array):
            utilities for agents of type X
        utility_Y (Array):
            utilities for agents of type Y
        """
        number_of_covariates_X = self.covariates_X.shape[-1]

        parameters_X = params[:number_of_covariates_X]
        parameters_Y = params[number_of_covariates_X:]

        utility_X = self.Utility(self.covariates_X, parameters_X)
        utility_Y = self.Utility(self.covariates_Y, parameters_Y)
        return utility_X, utility_Y

    def neg_log_likelihood(self, params: Array, data: ObservedData) -> Array:
        """Computes the negative log-likelihood function

        Args:
            params (Array): parameters of agents' utility functions
            data (ObservedData): observed transfers and numbers of matched and unmatched agents

        Returns:
            neg_log_lik (Array): negative log-likelihood value
        """
        utility_X, utility_Y = self.Utilities_of_agents(params)

        transfer = self.solve(utility_X, utility_Y)

        pX_xy, pX_x0 = self.ChoiceProbabilities_X(transfer, utility_X)
        pY_xy, pY_0y = self.ChoiceProbabilities_X(transfer, utility_Y)

        number_of_observations = (
            2 * jnp.sum(data.matched)
            + jnp.sum(data.unmatched_X)
            + jnp.sum(data.unmatched_Y)
        )

        log_lik_transfer = -jnp.log(jnp.mean((transfer - data.transfer) ** 2)) * (
            transfer.size / 2
        )
        log_lik_matched_X = jnp.nansum(data.matched * jnp.log(pX_xy))
        log_lik_matched_Y = jnp.nansum(data.matched * jnp.log(pY_xy))
        log_lik_unmatched_X = jnp.nansum(data.unmatched_X * jnp.log(pX_x0))
        log_lik_unmatched_Y = jnp.nansum(data.unmatched_Y * jnp.log(pY_0y))

        neg_log_lik = (
            -(
                log_lik_transfer
                + log_lik_matched_X
                + log_lik_matched_Y
                + log_lik_unmatched_X
                + log_lik_unmatched_Y
            )
            / number_of_observations
        )
        return neg_log_lik

    def fit(
        self,
        guess: Array,
        data: ObservedData,
        tol: float = 1e-8,
        maxiter: int = 100,
        verbose: bool | int = True,
    ) -> Array:
        """Estimate parameters of matching model by maximum likelihood (minimize the negative log-likelihood function)

        Args:
            guess (Array): initial parameter guess
            data (ObservedData): observed transfers and numbers of matched and unmatched agents
            tol (float): tolerance of the stopping criterion
            maxiter (int): maximum number of proximal gradient descent iterations
            verbose (bool): if set to True or 1 prints the information at each step of the solver, if set to 2, print also the information of the linesearch

        Returns:
            params (Array): parameter estimates
        """

        result = LBFGS(
            fun=self.neg_log_likelihood,
            tol=tol,
            maxiter=maxiter,
            verbose=verbose,
        ).run(guess, data)
        return result.params

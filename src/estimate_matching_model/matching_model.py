"""
JAX implementation of fixed-point iteration algoritm to solve and one-to-one matching model with transferable utility

Reference:
Esben Scriver Andersen, Note on solving one-to-one matching models with linear transferable utility, 2025 (https://arxiv.org/pdf/2409.05518)
"""

import jax.numpy as jnp

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
        transfer (jnp.ndarray): observed transfers
        matched (jnp.ndarray): number of observed matched agents
        unmatched_X (jnp.ndarray): number of observed unmatched agents of type X
        unmatched_Y (jnp.ndarray): number of observed unmatched agents of type Y
    """
    transfer: jnp.ndarray
    matched: jnp.ndarray
    unmatched_X: jnp.ndarray
    unmatched_Y: jnp.ndarray

@dataclass
class MatchingModel(Pytree, mutable=False):
    """Matching model

    Attributes:
        covariates_X (jnp.ndarray): covariates of utility function of agents of type X
        covariates_Y (jnp.ndarray): covariates of utility function of agents of type Y
        marginal_distribution_X (jnp.ndarray): marginal distribution of agents of type X
        marginal_distribution_Y (jnp.ndarray): marginal distribution of agents of type Y
    """

    covariates_X: jnp.ndarray
    covariates_Y: jnp.ndarray

    marginal_distribution_X: jnp.ndarray
    marginal_distribution_Y: jnp.ndarray

    def ChoiceProbabilities(self, v: jnp.ndarray, axis: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the logit choice probabilities for inside and outside options

        Args:
            v (jnp.ndarray): choice-specific payoffs
            axis (int): axis that describes the choice set

        Returns:
        P_inside (jnp.ndarray):
            choice probabilities of inside options.
        P_outside (jnp.ndarray):
            choice probabilities of outside option.
        """
        # v_max = jnp.max(v, axis=axis, keepdims=True)

        # exponentiated centered payoffs of inside options
        nominator = jnp.exp(v)

        # denominator of choice probabilities
        denominator = 1 + jnp.sum(nominator, axis=axis, keepdims=True)
        return nominator / denominator, 1 / denominator
    
    def Utility(self, covariates: jnp.ndarray, parameters: jnp.ndarray) -> jnp.ndarray:
        """Computes match-specific utilities

        Args:
            covariates (jnp.ndarray): covariates of utility function
            parameters (jnp.ndarray): parameters of utility function

        Returns:
            demand (jnp.ndarray): match-specific utilities
        """
        return jnp.einsum("ijk, k -> ij", covariates, parameters)
    
    def ChoiceProbabilities_X(self, transfer: jnp.ndarray, utility_X: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Computes choice probabilities of agents of type X
        
        Args:
            transfer (jnp.ndarray): match-specific transfers
            utilities (jnp.ndarray): match-specific utilities for agents of type X

        Returns:
            ChoiceProbabilities (jnp.ndarray): match-specific choice probabilities for agents of type X   
        """
        v_X = utility_X + transfer
        return self.ChoiceProbabilities(v_X, axis=1)
    
    def ChoiceProbabilities_Y(self, transfer: jnp.ndarray, utility_Y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Computes choice probabilities of agents of type Y
        
        Args:
            transfer (jnp.ndarray): match-specific transfers
            utilities (jnp.ndarray): match-specific utilities for agents of type Y

        Returns:
            ChoiceProbabilities (jnp.ndarray): match-specific choice probabilities for agents of type Y   
        """
        v_Y = utility_Y - transfer
        return self.ChoiceProbabilities(v_Y, axis=0)

    def Demand_X(self, transfer: jnp.ndarray, utility_X: jnp.ndarray) -> jnp.ndarray:
        """Computes agents of type X's demand for agents of type Y

        Args:
            transfer (jnp.ndarray): match-specific transfers
            utility_X (jnp.ndarray): match-specific utilities

        Returns:
            demand (jnp.ndarray): demand for inside options
        """
        return self.marginal_distribution_X * self.ChoiceProbabilities_X(transfer, utility_X)[0]

    def Demand_Y(self, transfer: jnp.ndarray, utility_Y: jnp.ndarray) -> jnp.ndarray:
        """Computes agents of type Y's demand for agents of type X

        Args:
            transfer (jnp.ndarray): match-specific transfers
            utility_Y (jnp.ndarray): match-specific utilities

        Returns:
            demand (jnp.ndarray): demand for inside options
        """
        return self.marginal_distribution_Y * self.ChoiceProbabilities_Y(transfer, utility_Y)[0]

    def UpdateTransfers(
        self, 
        t_initial: jnp.ndarray,
        utility_X: jnp.ndarray,
        utility_Y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Updates fixed point equation for transfers

        Args:
            t_initial (jnp.ndarray): initial transfers
            utility_X (jnp.ndarray): utility of agents of type X
            utility_Y (jnp.ndarray): utility of agents of type Y

        Returns:
            t_updated (jnp.ndarray): updated transfers
        """
        # Calculate demand for both sides of the market
        demand_X = self.Demand_X(t_initial, utility_X)  # type X's demand for type Y
        demand_Y = self.Demand_Y(t_initial, utility_Y)  # type Y's demand for type X

        # Update transfer
        t_updated = t_initial + 1/2 * jnp.log(demand_Y / demand_X)
        return t_updated
    
    def Utilities_of_agents(self, params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute match-specific utilities for agents of type X and Y
        
        Args:
            params (jnp.ndarray): parameters of agents utility functions

        Returns:
        utility_X (jnp.ndarray): 
            utilities for agents of type X
        utility_Y (jnp.ndarray): 
            utilities for agents of type Y  
        """
        number_of_covariates_X = self.covariates_X.shape[-1]

        parameters_X = params[:number_of_covariates_X]
        parameters_Y = params[number_of_covariates_X:]

        utility_X = self.Utility(self.covariates_X, parameters_X)
        utility_Y = self.Utility(self.covariates_Y, parameters_Y)
        return utility_X, utility_Y

    def solve(
        self,
        utility_X: jnp.ndarray,
        utility_Y: jnp.ndarray,
        fixed_point_solver: SolverTypes = SquaremAcceleration,
        tol: float = 1e-10,
        maxiter: int = 1000,
        verbose: bool = False,
    ) -> jnp.ndarray:
        """Solve for equilibrium transfer

        Args:
            params (jnp.ndarray): structural parameters
            fixed_point_solver (SolverTypes): solver used for solving fixed point equation (FixedPointIteration, AndersonAcceleration, SquaremAcceleration)
            tol (float): stopping tolerance for step length of fixed-point iterations, x_{i+1} - x_{i}
            maxiter (int): maximum number of iterations
            verbose (bool): whether to print information on every iteration or not.

        Returns:
            transfers (jnp.ndarray): equilibrium transfers
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
    
    def neg_log_likelihood(self, params: jnp.ndarray, data: ObservedData) -> jnp.ndarray:
        """Computes the negative log-likelihood function
        
        Args:
            params (jnp.ndarray): parameters
            data (ObservedData): observed data

        Returns:
            neg_log_lik (jnp.ndarray): negative log-likelihood value
        """
        utility_X, utility_Y = self.Utilities_of_agents(params)

        transfer = self.solve(utility_X, utility_Y)

        pX_xy, pX_x0 = self.ChoiceProbabilities_X(transfer, utility_X)
        pY_xy, pY_0y = self.ChoiceProbabilities_X(transfer, utility_Y)

        number_of_observations = 2 * jnp.sum(data.matched) + jnp.sum(data.unmatched_X) + jnp.sum(data.unmatched_Y)

        log_lik_transfer = -jnp.log(jnp.std(transfer - data.transfer))
        log_lik_matched_X = jnp.nansum(data.matched * jnp.log(pX_xy)) / number_of_observations
        log_lik_matched_Y = jnp.nansum(data.matched * jnp.log(pY_xy)) / number_of_observations
        log_lik_unmatched_X = jnp.nansum(data.unmatched_X * jnp.log(pX_x0)) / number_of_observations
        log_lik_unmatched_Y = jnp.nansum(data.unmatched_Y * jnp.log(pY_0y)) / number_of_observations

        neg_log_lik = -(log_lik_transfer + log_lik_matched_X + log_lik_matched_Y + log_lik_unmatched_X + log_lik_unmatched_Y)
        # neg_log_lik = -(log_lik_matched_X + log_lik_matched_Y + log_lik_unmatched_X + log_lik_unmatched_Y)
        return neg_log_lik
    
    def fit(
        self, 
        guess: jnp.ndarray, 
        data: ObservedData,
        tol: float = 1e-8,
        maxiter: int = 100,
        verbose: bool = True,
    ) -> jnp.ndarray:
        """Estimate parameters of matching model by maximum likelihood (minimize the negative log-likelihood function)
        
        Args:
            guess (jnp.ndarray): initial parameter guess
            data (ObservedData): observed data

        Returns:
            params (jnp.ndarray): parameter estimates
        """

        result = LBFGS(
            fun=self.neg_log_likelihood, 
            tol=tol, 
            maxiter=maxiter, 
            verbose=verbose,
        ).run(guess, data)

        print(result)
        return result.params
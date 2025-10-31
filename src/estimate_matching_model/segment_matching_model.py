"""
JAX implementation of a nested fixed-point algoritm to estimate a one-to-one matching model with transferable utility by maximum likelihood.

Through the use og jax.ops.segment_max() and jax.ops.segment_sum(), this implementation allow the choice set to vary over agents.

Reference:
Esben Scriver Andersen, Note on solving one-to-one matching models with linear transferable utility, 2025 (https://arxiv.org/pdf/2409.05518)
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.ops import segment_max, segment_sum

# import simple_pytree (used to store variables)
import equinox as eqx
from estimate_matching_model.dataclass_pytree import Data, ModelParameters, SolverTypes

# import solvers
from jaxopt import LBFGS
from squarem_jaxopt import SquaremAcceleration


class MatchingModel(eqx.Module):
    """Matching model

    Attributes
    ------------
        covariates_X : Array
            covariates of utility function of agents of type X
        covariates_Y : Array
            covariates of utility function of agents of type Y
        marginal_distribution_X : Array
            marginal distribution of agents of type X
        marginal_distribution_Y : Array
            marginal distribution of agents of type Y
        types_idx_X : Array
            index for agents of type X
        types_idx_Y : Array
            index for agents of type Y
        number_of_types_X : int
            number of types of agents of type X
        number_of_types_Y : int
            number of types of agents of type Y
    """
    covariates_X: Array
    covariates_Y: Array
    marginal_distribution_X: Array
    marginal_distribution_Y: Array
    types_idx_X: Array
    types_idx_Y: Array
    number_of_types_X: int
    number_of_types_Y: int

    def __init__(
        self,
        covariates_X,
        covariates_Y,
        types_idx_X,
        types_idx_Y,
        marginal_distribution_X,
        marginal_distribution_Y,
    ):
        self.covariates_X = covariates_X
        self.covariates_Y = covariates_Y
        self.marginal_distribution_X = marginal_distribution_X
        self.marginal_distribution_Y = marginal_distribution_Y
        self.types_idx_X = types_idx_X
        self.types_idx_Y = types_idx_Y
        self.number_of_types_X = int(jnp.max(self.types_idx_X).item()) + 1
        self.number_of_types_Y = int(jnp.max(self.types_idx_Y).item()) + 1

    def ChoiceProbabilities(
        self, v: Array, type_idx: Array, number_of_types: int
    ) -> tuple[Array, Array]:
        """Compute the logit choice probabilities for inside and outside options

        Params
        ----------
            v : Array
                choice-specific payoffs
            type_idx : Array
                index for the type of agents
            number_of_types : int
                number of agent types

        Returns
        ----------
            P_inside : Array
                choice probabilities of inside options
            P_outside : Array
                choice probabilities of outside option
        """
        # Center by subtracting max for numerical stability
        v_max = segment_max(v, type_idx, number_of_types)

        expV_inside = jnp.exp(v - v_max[type_idx])
        expV_outside = jnp.exp(-v_max)

        # denominator of choice probabilities (includes outside option with payoff 0)
        denominator = expV_outside + segment_sum(
            expV_inside, type_idx, num_segments=number_of_types
        )
        return expV_inside / denominator[type_idx], expV_outside / denominator

    def Utility(self, covariates: Array, parameters: Array) -> Array:
        """Computes match-specific utilities

        Params
        ----------
            covariates : Array
                covariates of utility function
            parameters : Array
                parameters of utility function

        Returns
        ----------
            utilities : Array
                match-specific utilities
        """
        return jnp.matmul(covariates, parameters)

    def ChoiceProbabilities_X(
        self, transfer: Array, utility_X: Array, mp: ModelParameters
    ) -> tuple[Array, Array]:
        """Computes choice probabilities of agents of type X

        Params
        ----------
            transfer : Array
                match-specific transfers
            utility_X : Array
                match-specific utilities for agents of type X
            scale_X : Array
                scale parameter of agents of type X

        Returns
        ----------
            P_inside : Array
                choice probabilities of inside options
            P_outside : Array
                choice probabilities of outside option
        """
        v_X = jax.lax.add(utility_X, transfer) / mp.scale_X
        return self.ChoiceProbabilities(
            v_X,
            self.types_idx_X,
            self.number_of_types_X,
        )

    def ChoiceProbabilities_Y(
        self, transfer: Array, utility_Y: Array, mp: ModelParameters
    ) -> tuple[Array, Array]:
        """Computes choice probabilities of agents of type Y

        Params
        ----------
            transfer : Array
                match-specific transfers
            utility_Y : Array
                match-specific utilities for agents of type Y
            mp : ModelParameters
                model parameters

        Returns
        ----------
            P_inside : Array
                choice probabilities of inside options
            P_outside : Array
                choice probabilities of outside option
        """
        v_Y = jax.lax.sub(utility_Y, transfer) / mp.scale_Y
        return self.ChoiceProbabilities(
            v_Y,
            self.types_idx_Y,
            self.number_of_types_Y,
        )

    def Demand_X(self, transfer: Array, utility_X: Array, mp: ModelParameters) -> Array:
        """Computes agents of type X's demand for agents of type Y

        Params
        ----------
            transfer : Array
                match-specific transfers
            utility_X : Array
                match-specific utilities
            mp : ModelParameters
                model parameters

        Returns
        ----------
            demand : Array
                demand for inside options
        """
        return (
            self.marginal_distribution_X[self.types_idx_X]
            * self.ChoiceProbabilities_X(transfer, utility_X, mp)[0]
        )

    def Demand_Y(self, transfer: Array, utility_Y: Array, mp: ModelParameters) -> Array:
        """Computes agents of type Y's demand for agents of type X

        Params
        ----------
            transfer : Array
                match-specific transfers
            utility_Y : Array
                match-specific utilities
            mp : ModelParameters
                model parameters

        Returns
        ----------
            demand : Array
                demand for inside options
        """
        return (
            self.marginal_distribution_Y[self.types_idx_Y]
            * self.ChoiceProbabilities_Y(transfer, utility_Y, mp)[0]
        )

    def UpdateTransfers(
        self,
        t_initial: Array,
        utility_X: Array,
        utility_Y: Array,
        mp: ModelParameters,
    ) -> Array:
        """Updates fixed-point equation for transfers

        Params
        ----------
            t_initial : Array
                initial transfers
            utility_X : Array
                utility of agents of type X
            utility_Y : Array
                utility of agents of type Y
            mp : ModelParameters
                model parameters

        Returns
        ----------
            t_updated : Array
                updated transfers
        """

        # Calculate demand for both sides of the market
        demand_X = self.Demand_X(t_initial, utility_X, mp)
        demand_Y = self.Demand_Y(t_initial, utility_Y, mp)

        # Update transfer
        t_updated = t_initial + mp.adjustment * jnp.log(demand_Y / demand_X)
        return t_updated

    def solve(
        self,
        utility_X: Array,
        utility_Y: Array,
        mp: ModelParameters,
        fixed_point_solver: SolverTypes = SquaremAcceleration,
        tol: float = 1e-10,
        maxiter: int = 50,
        verbose: bool = False,
    ) -> Array:
        """Solve for equilibrium transfer

        Params
        ----------
            utility_X : Array
                utilities of agents of type X
            utility_Y : Array
                utilities of agents of type Y
            mp : ModelParameters
                model parameters
            fixed_point_solver : SolverTypes
                solver used for solving fixed-point equation (FixedPointIteration, AndersonAcceleration, SquaremAcceleration)
            tol : float
                stopping tolerance for step length of fixed-point iterations, x_{i+1} - x_{i}
            maxiter : int
                maximum number of iterations
            verbose : bool
                whether to print information on every iteration or not.

        Returns
        ----------
            transfers : Array
                equilibrium transfers
        """
        # Initial guess for equilibrium transfers
        transfer_init = jnp.zeros(self.covariates_X.shape[:-1])

        # Find equilibrium transfers
        result = fixed_point_solver(
            self.UpdateTransfers,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
        ).run(transfer_init, utility_X, utility_Y, mp)
        return result.params

    def restrict_to_unit_interval(self, unrestricted_nesting: Array) -> Array:
        """Restrict nesting parameter to be in the unit interval

        Params
        ----------
            unrestricted_nesting : Array
                unrestricted nesting parameter

        Returns
        ----------
            restricted_nesting (Array):
                restricted nesting parameter
        """
        exp_nesting = jnp.exp(unrestricted_nesting)
        return exp_nesting / (1 + exp_nesting)

    def extract_parameters(self, params: Array) -> ModelParameters:
        """Extract the scale parameters from params

        Parameters
        ----------
            params : Array
                vector of model parameters

        Returns
        ----------
            mp : ModelParameters
                model parameters
        """
        number_of_covariates_X = self.covariates_X.shape[-1]
        number_of_covariates_Y = self.covariates_Y.shape[-1]
        number_of_covariate = number_of_covariates_X + number_of_covariates_Y

        # extract scale parameters and restrict them to be positive
        scale_X = jnp.exp(params[-2])
        scale_Y = jnp.exp(params[-1])

        # set adjustment factor of fixed-point equation
        adjustment = scale_X * scale_Y / (scale_X + scale_Y)

        return ModelParameters(
            beta_X=params[:number_of_covariates_X],
            beta_Y=params[number_of_covariates_X:number_of_covariate],
            scale_X=scale_X,
            scale_Y=scale_Y,
            adjustment=adjustment,
        )

    def restricted_parameters(self, unrestricted_params: Array):
        """Convert unrestricted parameters to restricted parameters

        Parameters
        ----------
            unrestricted_params : Array
                vector of unrestricted model parameters

        Returns
        ----------
            restricted_params : Array
                vector of restricted model parameters
        """
        mp = self.extract_parameters(unrestricted_params)
        restricted_params = jnp.concatenate([mp.beta_X, mp.beta_Y], axis=0)

        restricted_params = jnp.concatenate(
            [restricted_params, jnp.asarray([mp.scale_X, mp.scale_Y])],
            axis=0,
        )
        return restricted_params

    def Utilities_of_agents(self, mp: ModelParameters) -> tuple[Array, Array]:
        """Compute match-specific utilities for agents of type X and Y

        Params
        ----------
            mp : ModelParameters
                model parameters

        Returns
        ----------
            utility_X : Array
                match-specific utilities for agents of type X
            utility_Y : Array
                match-specific utilities for agents of type Y
        """
        utility_X = self.Utility(self.covariates_X, mp.beta_X)
        utility_Y = self.Utility(self.covariates_Y, mp.beta_Y)
        return utility_X, utility_Y

    def neg_log_likelihood(self, params: Array, data: Data) -> Array:
        """Computes the negative log-likelihood function

        Params
        ----------
            params : Array
                vector of model parameters
            data : Data
                observed transfers and numbers of matched and unmatched agents

        Returns
        ----------
            neg_log_lik : Array
                negative log-likelihood value
        """
        mp = self.extract_parameters(params)

        utility_X, utility_Y = self.Utilities_of_agents(mp)

        transfer = self.solve(utility_X, utility_Y, mp, verbose=False)

        pX_xy, pX_x0 = self.ChoiceProbabilities_X(transfer, utility_X, mp)
        pY_xy, pY_0y = self.ChoiceProbabilities_Y(transfer, utility_Y, mp)

        variance_errors = jnp.var(transfer - data.transfer)

        number_of_observations = (
            2 * jnp.sum(data.matched)
            + jnp.sum(data.unmatched_X)
            + jnp.sum(data.unmatched_Y)
        )

        log_lik_transfer = -jnp.log(variance_errors) * (variance_errors.size / 2)
        log_lik_matched = jnp.nansum(data.matched * (jnp.log(pX_xy) + jnp.log(pY_xy)))
        log_lik_unmatched_X = jnp.nansum(data.unmatched_X * jnp.log(pX_x0))
        log_lik_unmatched_Y = jnp.nansum(data.unmatched_Y * jnp.log(pY_0y))

        neg_log_lik = (
            -(
                log_lik_transfer
                + log_lik_matched
                + log_lik_unmatched_X
                + log_lik_unmatched_Y
            )
            / number_of_observations
        )
        return neg_log_lik

    def fit(
        self,
        guess: Array,
        data: Data,
        tol: float = 1e-8,
        maxiter: int = 100,
        verbose: bool | int = True,
    ) -> Array:
        """Estimate parameters of matching model by maximum likelihood (minimize the negative log-likelihood function)

        Params
        ----------
            guess : Array
                initial guess of model parameters
            data : Data
                observed transfers and numbers of matched and unmatched agents
            tol : float
                stopping tolerance for step length of optimization algorithm
            maxiter : int
                maximum number of iterations
            verbose : bool | int
                whether to print information on every iteration or not.

        Returns
        ----------
            estimated_params : Array
                estimated model parameters
        """
        result = LBFGS(
            fun=self.neg_log_likelihood,
            tol=tol,
            maxiter=maxiter,
            verbose=verbose,
        ).run(guess, data)
        return result.params

    def predict(self, params: Array) -> Data:
        """Predict transfers and number of matched and unmatched agents

        Params
        ----------
            params : Array
                vector of model parameters

        Returns
        ----------
            prediction : Data
                predicted transfers and numbers of matched and unmatched agents
        """
        mp = self.extract_parameters(params)
        print(mp)
        utility_X, utility_Y = self.Utilities_of_agents(mp)

        transfer = self.solve(utility_X, utility_Y, mp, verbose=False)

        pX_xy, pX_x0 = self.ChoiceProbabilities_X(transfer, utility_X, mp)
        pY_xy, pY_0y = self.ChoiceProbabilities_Y(transfer, utility_Y, mp)

        mX_xy = pX_xy * self.marginal_distribution_X[self.types_idx_X]
        mY_xy = pY_xy * self.marginal_distribution_Y[self.types_idx_Y]

        assert jnp.allclose(mX_xy, mY_xy), "demand and supply do not equate"

        return Data(
            transfer=transfer,
            matched=mX_xy,
            unmatched_X=pX_x0 * self.marginal_distribution_X,
            unmatched_Y=pY_0y * self.marginal_distribution_Y,
        )

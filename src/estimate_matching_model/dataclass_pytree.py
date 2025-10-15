from jax import Array
from simple_pytree import Pytree, dataclass

# import solvers
from jaxopt import FixedPointIteration, AndersonAcceleration
from squarem_jaxopt import SquaremAcceleration

SolverTypes = (
    type[SquaremAcceleration] | type[AndersonAcceleration] | type[FixedPointIteration]
)


@dataclass
class ModelParameters(Pytree, mutable=False):
    """Model parameters

    Attributes:
        beta_X (Array): parameters describing the utility function of agents of type X
        beta_Y (Array): parameters describing the utility function of agents of type Y
        scale_X (Array): scale parameter of the taste-shock for agents of type X
        scale_Y (Array): scale parameter of the taste-shock for agents of type Y
        nesting_parameter_X (Array | None): nesting parameter for agents of type X
        nesting_parameter_Y (Array | None): nesting parameter for agents of type Y
        adjustment (Array): adjustment factor for the fixed-point equation
    """

    beta_X: Array
    beta_Y: Array
    scale_X: Array
    scale_Y: Array
    adjustment: Array
    nesting_parameter_X: Array | None = None
    nesting_parameter_Y: Array | None = None


@dataclass
class Data(Pytree, mutable=False):
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

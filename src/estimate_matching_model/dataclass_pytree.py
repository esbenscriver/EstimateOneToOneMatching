from jax import Array
from simple_pytree import Pytree, dataclass

# import solvers
from jaxopt import FixedPointIteration, AndersonAcceleration
from squarem_jaxopt import SquaremAcceleration

SolverTypes = (
    type[SquaremAcceleration] | type[AndersonAcceleration] | type[FixedPointIteration]
)

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
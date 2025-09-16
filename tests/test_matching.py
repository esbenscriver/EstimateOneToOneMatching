import jax.numpy as jnp


def test_temp():
    temp = jnp.zeros((2, 3))
    assert jnp.allclose(temp, 0.0), "not a test"

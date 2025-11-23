import jax.numpy as jnp
def state_to_jax_array(state, dim):
    """Convert OMPL state to JAX array."""
    # return jnp.array([state[i] for i in range(dim)])
    return jnp.array(state[:dim])
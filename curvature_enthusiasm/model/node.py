import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx

from jaxtyping import Array, Float

class NODE(eqx.Module):
    """
        Neural Ordinary Differential Equation (NODE) module.

        This module implements a neural network that serves as the right-hand side
        function for solving ordinary differential equations. It takes state variables
        and time as input to predict the time derivative of the state.

        The network concatenates the state vector with time, processes them through
        an MLP, and scales the output to control the magnitude of predicted derivatives.
        This is commonly used in continuous-time neural networks and physics-informed
        machine learning.

        Attributes:
            out_scale: Scalar multiplier applied to the network output to control
                      the magnitude of predicted derivatives.
            mlp: The underlying multi-layer perceptron that maps concatenated
                 state-time inputs to derivative predictions.
    """

    out_scale: Float[Array, ""]
    mlp: eqx.nn.MLP

    def __init__(self, input_size, output_size, width_size, depth,
                 activation_func=jnn.gelu, out_scale=1e-1,
                 *, key, dtype=None, **kwargs):
        super().__init__(**kwargs)

        self.out_scale = jnp.array(out_scale)

        self.mlp = eqx.nn.MLP(
            in_size=input_size,
            out_size=output_size,
            width_size=width_size,
            depth=depth,
            activation=activation_func,
            dtype=dtype,
            key=key,
        )

    def __call__(
            self,
            x: Float[Array, "... D"],
            t: Float[Array, "... 1"] | Float[Array, "..."]
    ) -> Float[Array, "... D_out"]:
        """
        Forward pass: compute time derivative of state.

        This method implements the right-hand side function f(x, t) for the ODE:
            dx/dt = f(x, t)

        Args:
            x: State vector(s) with shape (..., D) where D is the state dimension.
               Can handle batched inputs with arbitrary leading dimensions.
            t: Time value(s) with shape (..., 1) or (...,). Will be broadcasted
               and concatenated with the state vector.

        Returns:
            Time derivative of the state with shape (..., D_out) where D_out
            is the output dimension specified during initialization.

        Note:
            The function:
            1. Concatenates state x and time t along the last dimension
            2. Passes the concatenated input through the MLP
            3. Scales the output by out_scale to control derivative magnitude

            The scaling is particularly important for stable ODE integration,
            as it prevents the network from predicting overly large derivatives
            that could lead to numerical instability or exploding trajectories.
        """
        if t.ndim == x.ndim - 1:
            t = jnp.expand_dims(t, axis=-1)
        elif t.ndim == x.ndim and t.shape[-1] != 1:
            t = jnp.expand_dims(t, axis=-1)

        xt = jnp.concatenate([x, t], axis=-1)
        yt = self.out_scale * self.mlp(xt)
        return yt
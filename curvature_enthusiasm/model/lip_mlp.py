from typing import Callable

import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from jaxtyping import Float, Array

from ..utils import default_floating_dtype

class LipMLPLayer(eqx.Module):
    weight: Float[Array, "out_features in_features"]
    bias: Float[Array, "out_features"]
    c: Float[Array, "1"]

    def __init__(self, in_features, out_features, key , dtype=None):
        dtype = default_floating_dtype() if dtype is None else dtype
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_features, in_features), dtype=dtype) * jnp.sqrt(2 / in_features)
        self.bias = jnp.zeros(out_features, dtype=dtype)
        self.c = jnp.max(jnp.sum(jnp.abs(self.weight), axis=1), keepdims=True)

    def weight_normalization(self) -> Float[Array, "out_features in_features"]:
        absrowsum = jnp.sum(jnp.abs(self.weight), axis=1)
        scale = jnp.minimum(1.0, jax.nn.softplus(self.c) / absrowsum)
        return self.weight * scale[:, None]

    def __call__(self, x: Float[Array, "in_features"]) -> Float[Array, "out_features"]:
        normalized_weight = self.weight_normalization()
        return jnp.dot(normalized_weight, x) + self.bias


class Lip_MLP(eqx.Module):
    layers: list
    activation_func: Callable
    final_activation: Callable | None
    dtype: jnp.dtype

    def __init__(
        self,
        *,
        in_size: int,
        key,
        width: int | None = 128,
        depth: int | None = 3,
        hidden: list[int] | None = None,
        activation=jnn.gelu,
        out_size: int = 6,
        final_activation: Callable | None = None,
        dtype=None,
    ):
        self.dtype = default_floating_dtype() if dtype is None else dtype

        # Build hidden sizes
        if hidden is None:
            width = 128 if width is None else width
            depth = 3 if depth is None else depth
            hidden = [width] * depth
        else:
            # make a copy to avoid mutating caller's list
            hidden = list(hidden)

        self.activation_func = activation
        self.final_activation = final_activation

        # Full layer sizes
        sizes = [in_size, *hidden, out_size]

        # One PRNG key per linear layer
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [
            LipMLPLayer(sizes[i], sizes[i + 1], keys[i], self.dtype)
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x: Float[Array, "in_size"]) -> Float[Array, "out_size"]:
        for layer in self.layers[:-1]:
            x = self.activation_func(layer(x))

        x = self.layers[-1](x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

    # Recommended Lipschitz regularizer (product of per-layer bounds)
    def get_lipschitz_loss(self) -> Float[Array, ""]:
        # recommended objective: product of per-layer bounds
        return jnp.prod(jax.nn.softplus(jnp.array([layer.c for layer in self.layers])))
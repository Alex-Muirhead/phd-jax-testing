import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


class ConvexCell(eqx.Module):
    normals: Float[Array, "ncells nfaces ndim"]
    offsets: Float[Array, "ncells nfaces"]

    @eqx.filter_jit
    def contains(self, points: Float[Array, "npoints ndim"], epsilon: float = 0.0):
        if epsilon == 0.0:
            epsilon = np.finfo(points.dtype).resolution
        point_offsets = jnp.einsum("...k,...jk->...j", points, self.normals)
        return jnp.all(point_offsets <= self.offsets + epsilon, axis=-1)


class LinearRay(eqx.Module):
    origin: Float[Array, "nrays ndim"]
    tangent: Float[Array, "nrays ndim"]

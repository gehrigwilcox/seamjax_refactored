"""
Typed, immutable containers for JAX: `@pytree` turns a class into a frozen dataclass that jit / vmap / scan / tree_map see
as a pytree (every field is data). Use `x.replace(a=...)` to change fields and `x[i]` to index every field at once
(e.g. `enemies[room]` -> the enemies of one room, `state[b]` -> one game out of a batch).
"""
import dataclasses
import jax


def pytree(cls):
    cls = dataclasses.dataclass(frozen=True)(cls)
    cls.replace = lambda self, **kw: dataclasses.replace(self, **kw)
    cls.__getitem__ = lambda self, i: jax.tree_util.tree_map(lambda a: a[i], self)
    cls.fields = classmethod(lambda c: tuple(f.name for f in dataclasses.fields(c)))
    return jax.tree_util.register_dataclass(cls)


def where(pred, a, b):
    """field-wise jnp.where(pred, a, b) for two pytrees of the same type; `pred` broadcasts from the left"""
    import jax.numpy as jnp
    def sel(x, y):
        p = jnp.reshape(pred, jnp.shape(pred)+(1,)*(jnp.ndim(x)-jnp.ndim(pred)))
        return jnp.where(p, x, y)
    return jax.tree_util.tree_map(sel, a, b)

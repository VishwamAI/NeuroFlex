import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import optax
from typing import Any, Callable, Dict
import dataclasses

@struct.dataclass
class ExtendedTrainState:
    params: Dict[str, Any]
    tx: Dict[str, optax.GradientTransformation]
    opt_state: Dict[str, optax.OptState]
    batch_stats: dict
    apply_fn: Callable

    @classmethod
    def create(cls, *, apply_fn, params, tx, batch_stats):
        opt_state = {k: v.init(params[k]) for k, v in tx.items()}
        return cls(
            params=params,
            tx=tx,
            opt_state=opt_state,
            batch_stats=batch_stats,
            apply_fn=apply_fn,
        )

    def apply_gradients(self, *, grads):
        updates = {}
        new_opt_state = {}
        for k, v in self.tx.items():
            updates[k], new_opt_state[k] = v.update(grads[k], self.opt_state[k], self.params[k])
        new_params = jax.tree_map(
            lambda p, u: optax.apply_updates(p, u),
            self.params,
            updates
        )
        return self.replace(params=new_params, opt_state=new_opt_state)

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def update_batch_stats(self, new_batch_stats):
        return self.replace(batch_stats=new_batch_stats)

    def tree_flatten(self):
        return (self.params, self.opt_state, self.batch_stats), (self.tx, self.apply_fn)

    @staticmethod
    def tree_unflatten(aux_data, children):
        return ExtendedTrainState(params=children[0], tx=aux_data[0], opt_state=children[1], batch_stats=children[2], apply_fn=aux_data[1])

# ExtendedTrainState is now automatically registered as a PyTree node
# due to the use of @struct.dataclass and the implementation of
# tree_flatten and tree_unflatten methods.

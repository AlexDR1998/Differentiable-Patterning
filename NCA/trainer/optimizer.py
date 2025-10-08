import jax
import optax
import jax.tree_util as jtu

"""
    Collection of helper functions for defining optimizers for training NCA models.
"""




def build_muon_dnums(params):
    return jtu.tree_map(
        lambda x: optax.contrib.MuonDimensionNumbers(
            reduction_axis=(1, 2, 3),  # in_channels, H, W
            output_axis=(0,)           # out_channels
        ) if (isinstance(x, jax.Array) and x.ndim == 4) else None,
        params
    )

def muon_optimizer(schedule):
    optimiser = optax.contrib.muon(
        schedule,
        muon_weight_dimension_numbers=build_muon_dnums
    )
    return optimiser
    # return optax.chain(
    #     optimiser,
    #     optax.scale_by_param_block_norm(),
    # )

def sam_optimizer(base_optimizer, rho=0.05, sync_period=2):
    """Wraps an existing optimizer with SAM (Sharpness-Aware Minimization)."""
    # return optax.chain(
        # optax.sam(rho=rho, base_optimizer=base_optimizer),
        # optax.scale_by_param_block_norm(),
    # )
    adv_opt = optax.chain(
        optax.contrib.normalize(),
        optax.adam(rho),
    )
    opt = optax.contrib.sam(
        optimizer=base_optimizer,
        adv_optimizer=adv_opt,
        sync_period=sync_period,
        opaque_mode=False
    )
    return opt


def reduce_on_plateau(optimizer,factor=0.8,patience=10,cooldown=5,accumulate_steps=1,rtol=1e-4,min_lr=1e-6):
    """Wraps an existing optimizer with ReduceLROnPlateau schedule."""
    scheduler = optax.contrib.reduce_on_plateau(
        factor=factor,
        patience=patience,
        cooldown=cooldown,
        accumulation_size=accumulate_steps,
        min_scale=min_lr,
        rtol=rtol,
    )
    opt = optax.chain(
        optimizer,
        scheduler
    )
    return opt



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




def build_optimizer(cfg):
    """
        Takes a hydra config and constructs the appropriate optimizer with learning rate schedule and any additional features (block norm, SAM, etc.)
    """
    # schedule = optax.exponential_decay(1e-3, transition_steps=cfg.run.iterations, decay_rate=0.99)
    init_lr = 1e-6      # starting learning rate
    
    warmup_fn = optax.linear_schedule(
        init_value=init_lr,
        end_value=cfg.optimiser.learn_rate,
        transition_steps=cfg.optimiser.warmup_steps,
    )

    decay_fn = optax.exponential_decay(
        init_value=cfg.optimiser.learn_rate,
        transition_steps=cfg.run.iterations,
        decay_rate=cfg.optimiser.decay_rate,
    )

    schedule = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[cfg.optimiser.warmup_steps],
    )
    if cfg.optimiser.type == "nadam":
        optimizer = optax.nadam(schedule)
        opt_name = "nadam"
    elif cfg.optimiser.type == "muon":
        optimizer = muon_optimizer(schedule)
        opt_name = "muon"
    elif cfg.optimiser.type == "adamw":
        optimizer = optax.adamw(schedule)
        opt_name = "adamw"
    else:
        raise ValueError(f"Unsupported optimizer type: {cfg.optimiser.type}")
    if cfg.optimiser.blocknorm:
        optimizer = optax.chain(optax.scale_by_param_block_norm(), optimizer)
        opt_name += "_blocknorm"
    if cfg.optimiser.sam:
        optimizer = sam_optimizer(optimizer, rho=cfg.optimiser.sam_rho, sync_period=cfg.optimiser.sam_sync_period)  
        opt_name += "_sam"
    return optimizer, opt_name

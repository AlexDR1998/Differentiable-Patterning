import jax
import optax
import jax.tree_util as jtu

"""
    Collection of helper functions for defining optimizers for training NCA models.
"""




def _cfg_get(cfg, key, default=None):
    """Read an optional config value without requiring all configs to define it."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    try:
        return getattr(cfg, key)
    except (AttributeError, KeyError):
        return default


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


def _schedule_config(cfg):
    schedule_cfg = _cfg_get(cfg.optimiser, "schedule", None)
    if isinstance(schedule_cfg, str):
        return schedule_cfg, None
    return _cfg_get(schedule_cfg, "type", "exponential"), schedule_cfg


def build_learning_rate_schedule(cfg):
    """Build the configured Optax learning-rate schedule and its name.

    All schedules share the existing linear warmup. Schedule-specific time is
    counted after warmup, except for the legacy exponential schedule whose
    transition length remains ``run.iterations`` for backward compatibility.
    """
    peak_lr = float(cfg.optimiser.learn_rate)
    warmup_steps = int(cfg.optimiser.warmup_steps)
    total_steps = int(cfg.run.iterations)
    if peak_lr <= 0:
        raise ValueError("optimiser.learn_rate must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            "optimiser.warmup_steps must be non-negative and smaller than run.iterations"
        )

    schedule_type, schedule_cfg = _schedule_config(cfg)
    schedule_type = str(schedule_type).lower()
    decay_steps = total_steps - warmup_steps

    if schedule_type == "exponential":
        decay_rate = float(
            _cfg_get(schedule_cfg, "decay_rate", cfg.optimiser.decay_rate)
        )
        if decay_rate <= 0:
            raise ValueError("optimiser decay_rate must be positive")
        post_warmup_schedule = optax.exponential_decay(
            init_value=peak_lr,
            transition_steps=total_steps,
            decay_rate=decay_rate,
        )
        schedule_name = f"exp{decay_rate:g}"
    elif schedule_type == "constant":
        post_warmup_schedule = optax.constant_schedule(peak_lr)
        schedule_name = "const"
    elif schedule_type == "cosine":
        final_factor = float(_cfg_get(schedule_cfg, "final_factor", 0.1))
        if not 0 <= final_factor <= 1:
            raise ValueError("optimiser.schedule.final_factor must be in [0, 1]")
        post_warmup_schedule = optax.cosine_decay_schedule(
            init_value=peak_lr,
            decay_steps=decay_steps,
            alpha=final_factor,
        )
        schedule_name = f"cos{final_factor:g}"
    elif schedule_type == "late_step":
        transition_fraction = float(
            _cfg_get(schedule_cfg, "transition_fraction", 0.75)
        )
        final_factor = float(_cfg_get(schedule_cfg, "final_factor", 0.2))
        if not 0 < transition_fraction < 1:
            raise ValueError(
                "optimiser.schedule.transition_fraction must be strictly between 0 and 1"
            )
        if not 0 <= final_factor <= 1:
            raise ValueError("optimiser.schedule.final_factor must be in [0, 1]")
        transition_step = max(
            1,
            min(decay_steps - 1, round(transition_fraction * decay_steps)),
        )
        post_warmup_schedule = optax.piecewise_constant_schedule(
            init_value=peak_lr,
            boundaries_and_scales={transition_step: final_factor},
        )
        schedule_name = f"step{transition_fraction:g}x{final_factor:g}"
    else:
        raise ValueError(
            "Unsupported optimiser.schedule.type "
            f"{schedule_type!r}; expected exponential, constant, cosine, or late_step"
        )

    if warmup_steps == 0:
        return post_warmup_schedule, schedule_name

    warmup_init_lr = float(
        _cfg_get(schedule_cfg, "warmup_init_lr", 1e-6)
    )
    if warmup_init_lr < 0:
        raise ValueError("optimiser.schedule.warmup_init_lr cannot be negative")
    warmup_schedule = optax.linear_schedule(
        init_value=warmup_init_lr,
        end_value=peak_lr,
        transition_steps=warmup_steps,
    )
    return (
        optax.join_schedules(
            schedules=[warmup_schedule, post_warmup_schedule],
            boundaries=[warmup_steps],
        ),
        schedule_name,
    )




def build_optimizer(cfg, return_schedule=False):
    """
        Takes a hydra config and constructs the appropriate optimizer with learning rate schedule and any additional features (block norm, SAM, etc.)
    """
    schedule, schedule_name = build_learning_rate_schedule(cfg)
    if cfg.optimiser.type == "nadam":
        base_optimizer = optax.nadam(schedule)
        opt_name = f"nadam_sched{schedule_name}"
    elif cfg.optimiser.type == "muon":
        base_optimizer = muon_optimizer(schedule)
        opt_name = f"muon_sched{schedule_name}"
    elif cfg.optimiser.type == "adamw":
        base_optimizer = optax.adamw(schedule)
        opt_name = f"adamw_sched{schedule_name}"
    else:
        raise ValueError(f"Unsupported optimizer type: {cfg.optimiser.type}")

    preprocessors = []

    gradient_clip_norm = _cfg_get(cfg.optimiser, "gradient_clip_norm", None)
    if gradient_clip_norm is not None:
        preprocessors.append(optax.clip_by_global_norm(gradient_clip_norm))
        opt_name += f"_clip{gradient_clip_norm:g}"

    if cfg.optimiser.blocknorm:
        preprocessors.append(optax.scale_by_param_block_norm())
        opt_name += "_blocknorm"

    optimizer = optax.chain(*preprocessors, base_optimizer)

    if cfg.optimiser.sam:
        optimizer = sam_optimizer(optimizer, rho=cfg.optimiser.sam_rho, sync_period=cfg.optimiser.sam_sync_period)  
        opt_name += "_sam"

    apply_if_finite = _cfg_get(cfg.optimiser, "apply_if_finite", False)
    if apply_if_finite:
        max_consecutive_errors = _cfg_get(cfg.optimiser, "max_consecutive_errors", 8)
        optimizer = optax.apply_if_finite(
            optimizer,
            max_consecutive_errors=max_consecutive_errors,
        )
        opt_name += f"_finite{max_consecutive_errors}"

    if return_schedule:
        return optimizer, opt_name, schedule
    return optimizer, opt_name

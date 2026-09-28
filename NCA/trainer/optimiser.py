import jax
import optax
import jax.tree_util as jtu

"""
    Collection of helper functions for defining optimisers for training NCA models.
"""




def build_muon_dnums(params):
    return jtu.tree_map(
        lambda x: optax.contrib.MuonDimensionNumbers(
            reduction_axis=(1, 2, 3),  # in_channels, H, W
            output_axis=(0,)           # out_channels
        ) if (isinstance(x, jax.Array) and x.ndim == 4) else None,
        params
    )

def muon_optimiser(schedule):
    optimiser = optax.contrib.muon(
        schedule,
        muon_weight_dimension_numbers=build_muon_dnums
    )
    return optimiser
    # return optax.chain(
    #     optimiser,
    #     optax.scale_by_param_block_norm(),
    # )

def sam_optimiser(base_optimiser, rho=0.05, sync_period=2):
    """Wraps an existing optimiser with SAM (Sharpness-Aware Minimization)."""
    # return optax.chain(
        # optax.sam(rho=rho, base_optimiser=base_optimiser),
        # optax.scale_by_param_block_norm(),
    # )
    adv_opt = optax.chain(
        optax.contrib.normalize(),
        optax.adam(rho),
    )
    opt = optax.contrib.sam(
        optimizer=base_optimiser,  # optax keyword
        adv_optimizer=adv_opt,  # optax keyword
        sync_period=sync_period,
        opaque_mode=False
    )
    return opt


def build_learning_rate_schedule(optimiser_config, total_steps):
    """Build the configured Optax learning-rate schedule and its name.

    All schedules share the existing linear warmup. Schedule-specific time is
    counted after warmup, except for the legacy exponential schedule whose
    transition length remains ``run.iterations`` for backward compatibility.
    """
    peak_lr = float(optimiser_config.learn_rate)
    warmup_steps = int(optimiser_config.warmup_steps)
    total_steps = int(total_steps)
    if peak_lr <= 0:
        raise ValueError("optimiser.learn_rate must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            "optimiser.warmup_steps must be non-negative and smaller than run.iterations"
        )

    schedule_cfg = optimiser_config.schedule
    schedule_type = schedule_cfg.type.lower()
    decay_steps = total_steps - warmup_steps

    if schedule_type == "exponential":
        decay_rate = float(
            optimiser_config.decay_rate
            if schedule_cfg.decay_rate is None
            else schedule_cfg.decay_rate
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
        final_factor = float(schedule_cfg.final_factor)
        if not 0 <= final_factor <= 1:
            raise ValueError("optimiser.schedule.final_factor must be in [0, 1]")
        post_warmup_schedule = optax.cosine_decay_schedule(
            init_value=peak_lr,
            decay_steps=decay_steps,
            alpha=final_factor,
        )
        schedule_name = f"cos{final_factor:g}"
    elif schedule_type == "late_step":
        transition_fraction = float(schedule_cfg.transition_fraction)
        final_factor = float(schedule_cfg.final_factor)
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

    warmup_init_lr = float(schedule_cfg.warmup_init_lr)
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




def build_optimiser(optimiser_config, total_steps, return_schedule=False):
    """
        Construct an optimiser from its focused typed configuration.
    """
    schedule, schedule_name = build_learning_rate_schedule(optimiser_config, total_steps)
    if optimiser_config.type == "nadam":
        base_optimiser = optax.nadam(schedule)
        opt_name = f"nadam_sched{schedule_name}"
    elif optimiser_config.type == "muon":
        base_optimiser = muon_optimiser(schedule)
        opt_name = f"muon_sched{schedule_name}"
    elif optimiser_config.type == "adamw":
        base_optimiser = optax.adamw(schedule)
        opt_name = f"adamw_sched{schedule_name}"
    else:
        raise ValueError(f"Unsupported optimiser type: {optimiser_config.type}")

    preprocessors = []

    gradient_clip_norm = optimiser_config.gradient_clip_norm
    if gradient_clip_norm is not None:
        preprocessors.append(optax.clip_by_global_norm(gradient_clip_norm))
        opt_name += f"_clip{gradient_clip_norm:g}"

    if optimiser_config.blocknorm:
        preprocessors.append(optax.scale_by_param_block_norm())
        opt_name += "_blocknorm"

    optimiser = optax.chain(*preprocessors, base_optimiser)

    if optimiser_config.sam:
        optimiser = sam_optimiser(optimiser, rho=optimiser_config.sam_rho, sync_period=optimiser_config.sam_sync_period)
        opt_name += "_sam"

    if optimiser_config.apply_if_finite:
        max_consecutive_errors = optimiser_config.max_consecutive_errors
        optimiser = optax.apply_if_finite(
            optimiser,
            max_consecutive_errors=max_consecutive_errors,
        )
        opt_name += f"_finite{max_consecutive_errors}"

    if return_schedule:
        return optimiser, opt_name, schedule
    return optimiser, opt_name

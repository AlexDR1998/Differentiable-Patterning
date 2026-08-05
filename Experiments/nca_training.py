"""Lifecycle boundary shared by active NCA experiment entrypoints."""

from NCA.trainer.context import TrainerContext
from NCA.trainer.trainer import build_trainer


def run_training(config, *, model, data, context: TrainerContext, key,
                 timesteps=None, loss_overrides=None):
    """Train and, when configured, publish the resulting model bundle."""

    trainer = build_trainer(config, model, data, context)
    result = trainer.train(
        key=key,
        timesteps=timesteps,
        loss_overrides=loss_overrides,
    )
    model_store = config.model_store
    if model_store.enabled and result.checkpoint_path is not None:
        from NCA.registry import publish_model_bundle

        if not model_store.root:
            raise ValueError(
                "model_store.root must be set when model bundling is enabled"
            )
        collection = model_store.collection or config.logging.wandb.project
        bundle = publish_model_bundle(
            store_root=model_store.root,
            collection=collection,
            run_name=context.run_name,
            checkpoint_path=result.checkpoint_path,
            cfg=config,
            training_result=result,
            model_factory=model_store.model_factory,
            evaluation_input=context.evaluation_input,
        )
        bundle.verify()
        result.checkpoint_path.unlink()
        print(f"Published model bundle: {bundle.path}")
    return result


__all__ = ["run_training"]

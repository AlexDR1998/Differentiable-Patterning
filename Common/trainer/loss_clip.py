import sys
# sys.path.append("..")

from functools import partial
from io import BytesIO
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax
import requests
from flax.training import orbax_utils
from flax.traverse_util import flatten_dict
from PIL import Image
from transformers import AutoTokenizer

from clip_jax import CLIPModel
# from clip_jax.data import image_to_logits
from clip_jax.utils import load_config



def build_clip_model():
    model_name = "boris/cappa-large-patch16-256-jax"
    local_dir = "cappa-large-patch16-256-jax"
    # !huggingface-cli download {model_name} --local-dir {local_dir}
    config_name = f"{local_dir}/config.json"
    config = load_config(config_name)
    config["vision_config"]["position_embedding_shape"] = (16,16)
    model = CLIPModel(**config)
    model_path = str(Path(local_dir).resolve())
    # model_path
    # restore checkpoint
    rng = jax.random.PRNGKey(0)
    logical_shape = jax.eval_shape(lambda rng: model.init_weights(rng), rng)["params"]
    params = jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), logical_shape)


    ckpt = {"params": params}
    restore_args = orbax_utils.restore_args_from_target(ckpt)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_options = orbax.checkpoint.CheckpointManagerOptions()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(model_path, orbax_checkpointer, orbax_options)
    step = checkpoint_manager.latest_step()
    ckpt = checkpoint_manager.restore(step, ckpt, restore_kwargs={"restore_args": restore_args, "transforms": {}})
    params = ckpt["params"]
    return model
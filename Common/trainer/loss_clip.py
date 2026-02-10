from calendar import c
import sys
# sys.path.append("..")

from functools import partial
from io import BytesIO
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import orbax
import requests
from flax.training import orbax_utils
from flax.traverse_util import flatten_dict
from PIL import Image
from einops import rearrange,einsum
# from transformers import AutoTokenizer
from Common.trainer.experiment_channel_grouping import duplicate_x_channels_9ch,split_and_pad_by_experiment_groups_12ch,pad_to_multiple_of_3_channels
from clip_jax import CLIPModel
# from clip_jax.data import image_to_logits
from clip_jax.utils import load_config


class CLIPVisionExtractor:
    """Lightweight wrapper for CLIP vision-only feature extraction."""
    
    def __init__(self, model, params, target_size=256):
        self.model = model
        self.params = params  # Keep all params as model may need full structure
        self.target_size = target_size  # Expected input size (256 for cappa-large-patch16-256)
    
    def extract_features(self, images, normalize=True):
        """
        Extract visual features from images.
        
        Args:
            images: Input images in shape (B, H, W, C)
            normalize: Whether to L2 normalize the embeddings
            
        Returns:
            image_embeds: Feature embeddings (B, embed_dim)
        """
        # Resize images to target size if needed
        B, H, W, C = images.shape
        if H != self.target_size or W != self.target_size:
            images = jax.image.resize(
                images, 
                (B, self.target_size, self.target_size, C), 
                method='bilinear'
            )
        
        # Call get_image_features - calling directly on model, not via apply
        # The model needs to bind params first
        bound_model = self.model.bind({"params": self.params})
        image_features = bound_model.get_image_features(
            images,
            attention_mask=None,
            position_ids=None,
            deterministic=True,
        )
        
        # Extract embeddings - get pooled_output from vision_model_output
        # If not available, use CLS token from last_hidden_state
        vision_output = image_features.get("vision_model_output", {})
        embeds = vision_output.get("pooled_output")
        
        if embeds is None:
            # Fall back to CLS token (first token) from last_hidden_state
            last_hidden = vision_output.get("last_hidden_state")
            if last_hidden is not None:
                embeds = last_hidden[:, 0, :]  # Take CLS token
        
        if embeds is None:
            raise ValueError(f"Could not extract embeddings from vision_model_output")
        
        if normalize:
            embeds = embeds / jnp.linalg.norm(embeds, axis=-1, keepdims=True)
        
        return embeds
    
    def __call__(self, images, normalize=True):
        """Convenience method for feature extraction."""
        return self.extract_features(images, normalize=normalize)


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
    return model, params


def build_clip_vision_extractor():
    """Build a vision-only feature extractor from CLIP."""
    model, params = build_clip_model()
    return CLIPVisionExtractor(model, params)

# _vision_extractor = build_clip_vision_extractor()

@eqx.filter_jit
def clip_loss_3ch(x,y,key,where,aux):
    """ 
    
    Parameters:
    x: predicted images (N C H W)
    y: target images (N C H W)
    where: mask of shape (N C H W) indicating where to compute loss

    Returns:
    loss: array N
    """

    metric = {
        "l2": lambda x,y: jnp.sum((x - y) ** 2, axis=-1),
        "l1": lambda x,y: jnp.sum(jnp.abs(x - y), axis=-1),
    }[aux["clip_metric"]]
    # vision_extractor = _vision_extractor

    if where is not None:
        x = x*where.astype(x.dtype)
        y = y*where.astype(y.dtype)
    x = rearrange(x, "n c h w -> n h w c")[:,:,:,:3]
    y = rearrange(y, "n c h w -> n h w c")[:,:,:,:3]
    x_embeds = aux["vision_extractor"](x, normalize=aux["normalize"])
    y_embeds = aux["vision_extractor"](y, normalize=aux["normalize"])
    loss = metric(x_embeds,y_embeds)

    # loss = jnp.mean(jnp.sum((x_embeds - y_embeds) ** 2, axis=-1))
    return loss



@eqx.filter_jit
def clip_loss_hyperspectral(x,y,key,where,aux):

    if where is not None:
        x = x*where.astype(x.dtype)
        y = y*where.astype(y.dtype)
    
    x = pad_to_multiple_of_3_channels(x)
    y = pad_to_multiple_of_3_channels(y)
    bc = x.shape[1] // 3
    x = rearrange(x,"n (c vc) w h -> (n c) w h vc", vc=3,c=bc)
    y = rearrange(y,"n (c vc) w h -> (n c) w h vc", vc=3,c=bc)
    x_embeds = aux["vision_extractor"](x, normalize=aux["normalize"])
    y_embeds = aux["vision_extractor"](y, normalize=aux["normalize"])
    metric = {
        "l2": lambda x,y: jnp.sum((x - y) ** 2, axis=-1),
        "l1": lambda x,y: jnp.sum(jnp.abs(x - y), axis=-1),
    }[aux["clip_metric"]]
    loss = metric(x_embeds,y_embeds)
    loss = rearrange(loss,"(n c) -> n c", c=bc)
    loss = jnp.mean(loss, axis=-1) # Average over c dimension
    return loss


@eqx.filter_jit
def clip_loss_colony(x,y,key,where,aux):
    if where is not None:
        x = x*where.astype(x.dtype)
        where_y = duplicate_x_channels_9ch(where)
        y = y*where_y.astype(y.dtype)

    x = duplicate_x_channels_9ch(x)
    x = split_and_pad_by_experiment_groups_12ch(x)
    y = split_and_pad_by_experiment_groups_12ch(y)		
    bc = x.shape[1] // 3
    x = rearrange(x,"n (c vc) w h -> (n c) w h vc", vc=3,c=bc)
    y = rearrange(y,"n (c vc) w h -> (n c) w h vc", vc=3,c=bc)
    x_embeds = aux["vision_extractor"](x, normalize=aux["normalize"])
    y_embeds = aux["vision_extractor"](y, normalize=aux["normalize"])
    metric = {
        "l2": lambda x,y: jnp.sum((x - y) ** 2, axis=-1),
        "l1": lambda x,y: jnp.sum(jnp.abs(x - y), axis=-1),
    }[aux["clip_metric"]]
    loss = metric(x_embeds,y_embeds)
    loss = rearrange(loss,"(n c) -> n c", c=bc)
    loss_weighting = jnp.array([0.5,1.0,0.5,1.0,1.0,1.0]) # Should there be an extra 1.0 here?
    # loss_weighting = jnp.array([0.5,1.0,0.5,1.0,1.0]) # Should there be an extra 1.0 here?
    loss = einsum(loss, loss_weighting,"n c, c -> n c",)
    loss = jnp.mean(loss, axis=-1) # Average over c dimension
    return loss


def clip_loss_colony_and_l2(x,y,key,where,aux):
	clip_loss = clip_loss_colony(x,y,key,where,aux)
	x_full = duplicate_x_channels_9ch(x)
	_l2 = (x_full-y)**2
	weighting = jnp.array([0.5,0.5,0.5,1.0,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.0]) # Account for duplicate channels
	_l2 = einsum(_l2,weighting,"n c x y , c -> n c x y")
	where_full = duplicate_x_channels_9ch(where).astype(where.dtype)
	l2_loss = jnp.nan_to_num(jnp.mean(_l2,axis=[-1,-2,-3],where=where_full))
	return clip_loss + l2_loss
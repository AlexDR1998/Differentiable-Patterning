import wandb
import numpy as np
from jaxtyping import Float, Array
from einops import rearrange
import io
from PIL import Image
wandb.login(key="c969e9166d4abf8c10db353deaa242e386db8b99")
class Train_log(object):
    def __init__(
        self,
        data,
        wandb_config=None,
    ):
        self.run = wandb.init(
            **wandb_config
        )
        self.wandb_config = wandb_config
        self.log_data_at_init(data)

    def log_data_at_init(self,data):    
        """
        Log data at the start of training. Defined as a separate function to allow for overwriting in subclasses
        """
        outputs = np.array(data)
        self.log_image("True sequence RGB", rearrange(outputs, "Batch Time C x y ->(Batch x) (Time y) C")[:,:,:3], step=None)

    def log_scalar(self, tag, value, step=None):
        wandb.log({tag: value}, step=step)

    def log_scalars(self, scalars_dict, step=None):
        wandb.log(scalars_dict, step=step)

    def log_image_single(
        self, 
        tag, 
        image: Float[Array,"Width Height Channels"], 
        step=None
    ):
        # image can be a numpy array or a local image file; wandb.Image handles both.
        try:
            image = np.array(image)
            assert len(image.shape) == 3, "Image must be 3D"
            
            # Convert to PIL Image
            if image.dtype != np.uint8:
                image = np.clip(image * 255 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            
            # Use in-memory buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            wandb_image = wandb.Image(Image.open(buffer))
            wandb.log({tag: wandb_image}, step=step)
            
        except Exception as e:
            print(f"Warning: Failed to process single image {tag}: {e}")
    
    def log_image_batch(
        self, 
        tag, 
        images: Float[Array,"Batch Width Height Channels"],
        step=None
    ):
        image = np.array(images)
        assert len(image.shape) == 4, "Image batch must be 4D"
        # Convert to a list of wandb.Image objects
        wandb_images = []
        for img in image:
            try:
                # Convert to PIL Image
                if img.dtype != np.uint8:
                    img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
                
                pil_image = Image.fromarray(img)
                
                # Use in-memory buffer
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')
                buffer.seek(0)
                
                wandb_images.append(wandb.Image(Image.open(buffer)))
                
            except Exception as e:
                print(f"Warning: Failed to process image: {e}")
                continue
        
        if wandb_images:
            wandb.log({tag: wandb_images}, step=step)
    
    def log_video(self,tag,video:Float[Array,"T C X Y"],step=None):
        """
            Expects a 4D tensor of shape (T, C, X, Y) where C is 1 or 3
            Values should be floats in [0,1]
        """
        assert len(video.shape) == 4, "Video must be 4D"
        assert video.shape[1] in [1, 3], "Video must have 1 or 3 channels"
        
        video = np.array(video)
        # Convert to uint8
        video = np.clip(video * 255, 0, 255).astype(np.uint8)
    
        wandb_video = wandb.Video(video, fps=10,format="mp4")
        wandb.log({tag: wandb_video}, step=None)

    def log_image(self, tag, images, step=None):
        # Accepts either [Batch, Width, Height, Channels] or [Width, Height, Channels]
        # If images is a list, convert to numpy array
        images = np.array(images)
        if images.shape[-1]==2:
            images = np.concatenate([images, np.zeros_like(images[..., :1])], axis=-1)  # Add a zero 3rd channel if only 2 channels are present
        if len(images.shape) == 4:
            # If images is a batch, log as a batch
            self.log_image_batch(tag, images, step)
        elif len(images.shape) == 3:
            # If images is a single image, log as a single image
            self.log_image_single(tag, images, step)
        else:
            raise ValueError("Image must be 3D or 4D (batch)")

    def log_histogram(self, tag, values, step=None):
        # values = np.array(values)
        # values = values.ravel()
        # if values.size < 2 or np.max(values) == np.min(values):
        #     print(f"Warning: Histogram {tag} has no variation.")
        # else:
        try:
            wandb.log({tag: wandb.Histogram(values)}, step=step)
        except Exception as e:
            print(f"Warning: Failed to log histogram {tag}: {e}")
    def log_text(self, tag, text, step=None):
        wandb.log({tag: text}, step=step)

    def log(self, data_dict, step=None):
        wandb.log(data_dict, step=step)

    def finish(self):
        wandb.finish()

    def tb_training_end_log(self,model,x,t,*args):
        self.finish()
    
    def log_model_parameters(self,model,i):
        raise NotImplementedError
    
    def log_model_outputs(self,x,i):
        raise NotImplementedError
    
    def normalise_images(self,x):
        """
        Normalises the images to [0,1] range for tensorboard logging
        """
        x = x - np.min(x)
        x = x / np.max(x)
        return x

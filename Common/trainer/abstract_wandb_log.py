import wandb
import numpy as np
from jaxtyping import Float, Array
from einops import rearrange
class Train_log(object):
    def __init__(
        self,
        data,
        config,
        project="default_project",
        entity=None,
        tags=None,
        notes=None,
        dir=None,
    ):
        self.run = wandb.init(
            config=config,
            project=project,
            entity=entity,
            tags=tags,
            notes=notes,
            dir=dir,
        )
        
        outputs = np.array(data)    
        self.log_image("True sequence RGB", rearrange(outputs, "Batch Time C x y ->(Batch x) (Time y) C")[:,:,:3], step=None)

    def log_scalar(self, tag, value, step=None):
        wandb.log({tag: value}, step=step)

    def log_scalars(self, scalars_dict, step=None):
        wandb.log(scalars_dict, step=step)

    def log_image_single(self, tag, image, step=None):
        # image can be a numpy array or a local image file; wandb.Image handles both.
        image = np.array(image)
        assert len(image.shape) == 3, "Image must be 3D"
        print(image.shape)
        wandb.log({tag: wandb.Image(image)}, step=step)
    
    def log_image_batch(self, tag, images, step=None):
        image = np.array(images)
        assert len(image.shape) == 4, "Image batch must be 4D"
        # Convert to a list of wandb.Image objects
        wandb_images = [wandb.Image(img) for img in image]
        # Log the images as a batch
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
    
        print("Video shape: ",video.shape)
        wandb_video = wandb.Video(video, fps=10,format="mp4")
        wandb.log({tag: wandb_video}, step=None)

    def log_image(self, tag, images, step=None):
        images = np.array(images)
        if len(images.shape) == 4:
            # If images is a batch, log as a batch
            self.log_image_batch(tag, images, step)
        elif len(images.shape) == 3:
            # If images is a single image, log as a single image
            self.log_image_single(tag, images, step)
        else:
            raise ValueError("Image must be 3D or 4D (batch)")

    def log_histogram(self, tag, values, step=None):
        wandb.log({tag: wandb.Histogram(values)}, step=step)

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

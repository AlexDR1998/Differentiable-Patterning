from einops import rearrange
import numpy as np
from Common.trainer.abstract_wandb_log import Train_log

class Impulse_Train_log(Train_log):
    """
        Class for logging training behaviour of NCA_Impulse_Trainer classes 
        """
    
    def log_training(self,aux,i,log_interval):
        """Log training behaviour

        Args:
            aux : auxiliary data from training step
            i : training step
            log_interval : interval for logging
        """
        
        self.log({"Train/total_loss":aux['total_loss']},step=i)
        self.log({"Train/mean_loss":aux['mean_loss']},step=i)
        self.log({"Train/regulariser":aux['dx_reg']},step=i)
        self.log({"Train/losses":aux['loss_batches']},step=i)
        self.log_histogram("Train/impulse", np.ravel(aux['dx']),step=i)
        self.log({"Train/log_mean_loss":np.log(aux['mean_loss']+1e-8)},step=i)
        self.log({"Train/location":aux['dx_location']},step=i)
        if i % log_interval == 0:
            self.log_image(
                tag = "Train/output",
                # images = rearrange(aux['final_states'][:,:27],"POOL (C1 C2 C3) W H -> POOL (C1 W) (C2 H) C3",C3=3,C1=3,C2=3),
                images = rearrange(aux['final_states'][:1,:3],"POOL C W H -> POOL W H C"),
                # images
                step=i
            )
            self.log_image(
                tag = "Train/output_hidden",
                images = rearrange(aux['final_states'][:1,3:30,::2,::2],"POOL (C1 C2 C3) W H -> POOL (C1 W) (C2 H) C3",C3=3,C1=3,C2=3),
                step=i
            )


    def log_final_trajectory(self,T):
        print("Logging final trajectory")
        print("Trajectory shape: "+str(T.shape))

        self.log_video(
            tag = "Evaluation/trajectory",
            video = T[0,:,:3,:,:],
            step=None)
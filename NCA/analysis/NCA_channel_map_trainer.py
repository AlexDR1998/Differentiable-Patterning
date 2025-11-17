import jax
import jax.numpy as jnp
import equinox as eqx
from Common.model.abstract_model import AbstractModel # Inherit model 
from NCA.analysis.NCA_channel_map import NCA_channel_map_linear
from Common.model.boundary import model_boundary
from .NCA_channel_extractor import NCA_channel_extractor
from NCA.analysis.tensorboard_log import CM_Train_log
from jaxtyping import Float, Array
from einops import rearrange,repeat
from tqdm import tqdm
import time
import jax.random as jr
PVC_PATH = "/mnt/ceph/ar-dp/"

class NCA_channel_map_trainer(object):
    def __init__(self,
                 NCA_model,
                 data,
                 boundary_mask,
                 MEASURED_CHANNELS,
                 TARGET_CHANNELS,
                 BATCHES:int,
                 CM_MODEL = NCA_channel_map_linear,
                 GATED=True,
                 key=jr.PRNGKey(int(time.time()))
    ):
        
        self.GATED = GATED
        self.MEASURED_CHANNELS = MEASURED_CHANNELS
        self.TARGET_CHANNELS = TARGET_CHANNELS
        

        boundary_func = model_boundary(boundary_mask[0])
        data = data*rearrange(boundary_mask,"B () X Y -> B () () X Y")

        print("Channel order: Foxa2, Sox17, TbxT, Lmbr, Cer, Lefty, Nodal, Lef1")
        print(f"Total data shape: {data.shape}")
        self.data = data

        self.CE = NCA_channel_extractor(
            NCA_model,
            boundary_func,
            GATED=GATED
        )
        # Define arguments for whichever model class from NCA_channel_map is used
        cm_args = {
            "key":key,
            "MEASURED_CHANNELS":self.MEASURED_CHANNELS,
            "TARGET_CHANNELS":self.TARGET_CHANNELS,
            "LATENT":32
        }
        self.model = CM_MODEL(**cm_args)
        
    
    def train(self,
              STEPS_BETWEEN_IMAGES:int,
              ITERS:int,
              optimiser,
              FILENAME:str,
              LOG_EVERY=100,
              wandb_config={"project":"micropattern channel map",
                            "group":"Development",
                            "tags":["NCA","channel map","micropattern"]},
              key=jr.PRNGKey(int(time.time())),

              ):
        # The step that actually does the training
        #--------------------------------------
        @eqx.filter_jit
        def make_step(model,x,y,opt_state):
            @eqx.filter_value_and_grad(has_aux=True)
            def compute_loss(model_diff,model_static,X,Y_true):
                model = eqx.combine(model_diff,model_static)
                Y = model(X)
                loss = jnp.mean((Y_true - Y)**2)
                return loss,Y
            model_diff,model_static = eqx.partition(model,filter_spec=eqx.is_inexact_array)
            (loss,Y),grad = compute_loss(model_diff,model_static,x,y)
            updates, opt_state = optimiser.update(grad,opt_state,model)
            model = eqx.apply_updates(model,updates)
            return model,loss,opt_state,Y
        #--------------------------------------


        # Initialise the model and optimiser
        Y_true = self.data[:,:,self.TARGET_CHANNELS]
        self.LOGGER = CM_Train_log(
            data=Y_true,
            wandb_config=wandb_config)

        model = self.model
        model_diff,_ = eqx.partition(model,filter_spec=eqx.is_inexact_array)
        opt_state = optimiser.init(model_diff)
        
        X,_ = self.CE.generate_data(
            STEPS_BETWEEN_IMAGES,
            self.data,
            MEASURED_CHANNELS=self.MEASURED_CHANNELS,
            TARGET_CHANNELS=self.TARGET_CHANNELS,
            key=key)
        
        self.LOGGER.log_data_at_init(X,label="Measured channels processed")

        # Do the training loop
        pbar = tqdm(range(ITERS))
        for i in pbar:
            key = jr.fold_in(key,i)
            model,loss,opt_state,y_pred = make_step(
                model,
                x=X,
                y=Y_true,
                opt_state=opt_state)
            pbar.set_postfix(loss=loss)
            self.LOGGER.tb_training_loop_log_sequence(loss,model,y_pred,i,write_images=True,LOG_EVERY=LOG_EVERY)
        model.save(FILENAME)
        self.model = model
        self.LOGGER.finish()
        jnp.save(PVC_PATH+"output/"+FILENAME+"_prediction.npy",y_pred)
        jnp.save(PVC_PATH+"output/"+FILENAME+"_true.npy",Y_true)
        
import jax
import jax.numpy as jnp
import equinox as eqx
from Common.model.abstract_model import AbstractModel # Inherit model 
from Common.utils import adhesion_mask_convex_hull_circle
from Common.model.boundary import model_boundary
from .NCA_channel_extractor import NCA_channel_extractor
from NCA.analysis.tensorboard_log import CM_Train_log
from jaxtyping import Float, Array
from einops import rearrange,repeat
from tqdm import tqdm
import time
import jax.random as jr

class NCA_channel_map(AbstractModel):
    layers: list
    hyperparameters: dict
    def __init__(self,key,FULL_CHANNELS,TARGET_CHANNELS):
        self.hyperparameters = {
            "FULL_CHANNELS":FULL_CHANNELS,
            "TARGET_CHANNELS":TARGET_CHANNELS
        }
        self.layers = [
            eqx.nn.Linear(in_features=len(FULL_CHANNELS),
                          out_features=len(TARGET_CHANNELS),
                          key=key),
        ]

    def __call__(self,X:Float[Array,"B T c X Y"]):
        X_flat = rearrange(X,"B T C_INPUT X Y -> (B T X Y) C_INPUT")
        vfunc = eqx.filter_vmap(self.layers[0])
        Y_flat = vfunc(X_flat)
        Y = rearrange(Y_flat,"(B T X Y) C_TARGET -> B T C_TARGET X Y",B=X.shape[0],T=X.shape[1],X=X.shape[3],Y=X.shape[4])
        return Y
    

class NCA_channel_map_trainer(object):
    def __init__(self,
                 NCA_model,
                 data,
                 FULL_CHANNELS,
                 TARGET_CHANNELS,
                 BATCHES:int,
                 GATED=True):
        
        self.GATED = GATED
        self.FULL_CHANNELS = FULL_CHANNELS
        self.TARGET_CHANNELS = TARGET_CHANNELS
        
        
        #data = jnp.load("output/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_T_true.npy")
        boundary_mask = adhesion_mask_convex_hull_circle(rearrange(data[0],"C X Y -> X Y C"))[0]
        boundary_mask = rearrange(boundary_mask,"X Y -> 1 X Y")
        print("Boundary mask shape: ",boundary_mask.shape)
        boundary_func = model_boundary(boundary_mask)


        boundary_mask = repeat(boundary_mask,"1 X Y -> B 1 X Y",B=BATCHES)
        data = repeat(data,"T C X Y -> B T C X Y", B=BATCHES)

        data = data*rearrange(boundary_mask,"B () X Y -> B () () X Y")

        print("Channel order: Foxa2, Sox17, TbxT, Lmbr, Cer, Lefty, Nodal, Lef1")
        print(f"Total data shape: {data.shape}")
        self.data = data

        self.CE = NCA_channel_extractor(
            NCA_model,
            boundary_func,
            GATED=GATED
        )
        

    def train(self,
              STEPS_BETWEEN_IMAGES:int,
              ITERS:int,
              optimiser,
              FILENAME:str,
              wandb_config={"project":"micropattern channel map",
                            "group":"Development",
                            "tags":["NCA","channel map","micropattern"]},
              key=jr.PRNGKey(int(time.time())),
              ):
        """Train the NCA channel map model"""
        @eqx.filter_jit
        def make_step(model,x,y,opt_state):
            @eqx.filter_value_and_grad(has_aux=False)
            def compute_loss(model_diff,model_static,X,Y_true):
                model = eqx.combine(model_diff,model_static)
                Y = model(X)
                loss = jnp.mean((Y_true - Y)**2)
                return loss
            model_diff,model_static = eqx.partition(model,filter_spec=eqx.is_inexact_array)
            loss,grad = compute_loss(model_diff,model_static,x,y)
            updates, opt_state = optimiser.update(grad,opt_state,model)
            model = eqx.apply_updates(model,updates)
            return model,loss,opt_state
        #--------------------------------------


        self.LOGGER = CM_Train_log(data=None,wandb_config=wandb_config)

        model = NCA_channel_map(key,self.FULL_CHANNELS,self.TARGET_CHANNELS)
        model_diff,_ = eqx.partition(model,filter_spec=eqx.is_inexact_array)
        opt_state = optimiser.init(model_diff)
        pbar = tqdm(range(ITERS))
        data = self.CE.generate_data(STEPS_BETWEEN_IMAGES,self.data,channels=self.FULL_CHANNELS,key=key)
        for i in pbar:
            key = jr.fold_in(key,i)
            model,loss,opt_state = make_step(
                model,
                x=data,
                y=self.data[:,:,self.TARGET_CHANNELS],
                opt_state=opt_state)
            pbar.set_postfix(loss=loss)
        model.save("")
        #return model
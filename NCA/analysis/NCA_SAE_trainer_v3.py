import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jr
import optax
import equinox as eqx
import Common.trainer.loss as loss
from NCA.trainer.data_augmenter_nca import DataAugmenter
from Common.utils import key_pytree_gen
from tqdm import tqdm
import time
import datetime
from NCA.analysis.tensorboard_log import SAE_Train_log_v3
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.analysis.optimiser import unitary_decoder_transform

from pprint import pprint

class NCA_SAE_Trainer(NCA_Trainer):
    """
    NCA_SAE_Trainer is a class for training a Sparse Autoencoder (SAE) on the features extracted from a NCA model.
    It inherits from the NCA_Trainer class and provides methods for training the SAE, logging the training process,
    and generating activations.

    Train the SAE in the same setup as the original NCA was trained. The only change is:
        - The NCA is already trained
        - Gradients are taken wrt the SAE parameters
    
        
    """

    def __init__(self,
                 NCA_model,
                 SAE,
                 data,
                 model_filename=None,
                 DATA_AUGMENTER = DataAugmenter,
                 BOUNDARY_MASK = None, 
                 BOUNDARY_MODE = "soft", # "soft" or "hard"
                 SHARDING = None, 
                 GRAD_LOSS = True,
                 OBS_CHANNELS = None,
                 LOSS_TIME_CHANNEL_MASK = None,
                 MODEL_DIRECTORY="models/",
                 LOG_DIRECTORY="logs/"):
        super().__init__(NCA_model=NCA_model,
                         data=data,
                         model_filename=model_filename,
                         DATA_AUGMENTER=DATA_AUGMENTER,
                         BOUNDARY_MASK=BOUNDARY_MASK,
                         BOUNDARY_MODE=BOUNDARY_MODE,
                         SHARDING=SHARDING,
                         GRAD_LOSS=GRAD_LOSS,
                         OBS_CHANNELS=OBS_CHANNELS,
                         LOSS_TIME_CHANNEL_MASK=LOSS_TIME_CHANNEL_MASK,
                         MODEL_DIRECTORY=MODEL_DIRECTORY,
                         LOG_DIRECTORY=LOG_DIRECTORY)
        self.SAE = SAE

    
    def setup_logging(self, BACKEND, wandb_args):
        if self.model_filename is None:
            self.model_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.IS_LOGGING = False
        else:
            self.IS_LOGGING = True
            config = {"NCA":self.NCA_model.get_config(),
                      "SAE":self.SAE.get_config(),
                      "TRAINING":self.TRAIN_CONFIG}
            wandb_args["config"] = config
            self.LOGGER = SAE_Train_log_v3(
                data=None,
                wandb_config=wandb_args,
            )

    def latent_sparsity(self,latents):
        _sparsity = lambda x: jnp.mean(jnp.abs(x))
        S = jtu.tree_map(_sparsity, latents)
        S = jnp.array(S)
        S = jnp.mean(S)
        return S
        #return jnp.mean(jnp.abs(latents))


    def train(self,
                t,
                iters,
                optimiser=None,
                STATE_REGULARISER=1.0,
                BOUNDARY_REGULARISER=1.0,
                SPARSITY_STRENGTH=1.0,
                WARMUP=64,
                LOSS_SAMPLING = 64,
                LOG_EVERY=40,
                CLEAR_CACHE_EVERY=100,
                WRITE_IMAGES=True,
                LOSS_FUNC_STR = "euclidean",
                LOOP_AUTODIFF = "checkpointed",
                wandb_args={"project":"NCA",
                            "group":"group_1",
                            "tags":["training"]},
                key=jr.PRNGKey(int(time.time()))):
        self.TRAIN_CONFIG = {
            "t":t,
            "iters":iters,
            "optimiser":optimiser,
            "STATE_REGULARISER":STATE_REGULARISER,
            "BOUNDARY_REGULARISER":BOUNDARY_REGULARISER,
            "WARMUP":WARMUP,
            "LOSS_SAMPLING":LOSS_SAMPLING,
            "LOG_EVERY":LOG_EVERY,
            "CLEAR_CACHE_EVERY":CLEAR_CACHE_EVERY,
            "WRITE_IMAGES":WRITE_IMAGES,
            "LOSS_FUNC_STR":LOSS_FUNC_STR,
            "LOOP_AUTODIFF":LOOP_AUTODIFF,
		}
        self.setup_logging(BACKEND="wandb", wandb_args=wandb_args)
        if LOSS_FUNC_STR=="l2":
            self._loss_func = loss.l2
        elif LOSS_FUNC_STR=="l1":
            self._loss_func = loss.l1
        elif LOSS_FUNC_STR=="vgg":
            self._loss_func = loss.vgg
        elif LOSS_FUNC_STR=="euclidean":
            self._loss_func = loss.euclidean
        elif LOSS_FUNC_STR=="spectral":
            self._loss_func = loss.spectral
        elif LOSS_FUNC_STR=="spectral_full":
            self._loss_func = loss.spectral_weighted
        elif LOSS_FUNC_STR=="rand_euclidean":
            self._loss_func = lambda x,y,dummy_key:loss.random_sampled_euclidean(x,y,key=key)

        @eqx.filter_jit
        def make_step(sae,nca,x,y,t,opt_state,key):


            @eqx.filter_value_and_grad(has_aux=True)
            def compute_loss(sae_diff,sae_static,nca,x,y,t,key):
                _sae = eqx.combine(sae_diff,sae_static)
                
                def nca_call_with_sae(x,boundary_callback,key):
                    x,latents,_,_ = nca.call_with_SAE(x,
                                                SAE=_sae,
                                                latent_edit={"mode":"none",
                                                            "positions":None,
                                                            "values":1.0},
                                                key=key,
                                                boundary_callback=boundary_callback)
                    return (x,latents)
                
                v_nca = jax.vmap(nca_call_with_sae,
                                in_axes=(0,None,0),
                                out_axes=(0,0))
                _vv_nca = lambda x,callback,key_array:jax.tree_util.tree_map(v_nca,x,callback,key_array)
                
                def vv_nca(x,callback,key_array):
                    output = _vv_nca(x,callback,key_array)
                    x,latents = map(list, zip(*output))
                    return x,latents
                
                v_intermediate_reg = lambda x:jnp.array(jax.tree_util.tree_map(self.intermediate_reg,x))  # noqa: E731
                _loss_func = lambda x,y,key:self.loss_func(x,y,key,SAMPLES=LOSS_SAMPLING)  # noqa: E731
                v_loss_func = lambda x,y,key_array:jnp.array(jax.tree_util.tree_map(_loss_func,x,y,key_array))


                def nca_step(carry,j):
                    key,x = carry
                    key = jr.fold_in(key,j)
                    key_array = key_pytree_gen(key,(len(x),x[0].shape[0]))
                    x,latents = vv_nca(x,self.BOUNDARY_CALLBACK,key_array)
                    reg_log=v_intermediate_reg(x)
                    boundary_reg_log=self.boundary_regulariser(x)
                    loss_sparsity = self.latent_sparsity(latents)
                    stack = (loss_sparsity,reg_log,boundary_reg_log)
                    return (key,x),stack
                

                (key,x),(loss_sparsity,reg_log,boundary_reg_log) = eqx.internal.scan(
                    nca_step,
                    (key,x),
                    xs=jnp.arange(t),
                    kind=LOOP_AUTODIFF
                )
                loss_key = key_pytree_gen(key, (len(x),))
                losses = v_loss_func(x, y, loss_key)
                mean_recon_loss = jnp.mean(losses)
                mean_loss_sparsity = jnp.mean(loss_sparsity)
                reg_log = jnp.mean(reg_log)
                boundary_reg_log = jnp.mean(boundary_reg_log)
                loss_total = mean_recon_loss + STATE_REGULARISER*reg_log + BOUNDARY_REGULARISER*boundary_reg_log + SPARSITY_STRENGTH*mean_loss_sparsity
                return loss_total, (x,losses,mean_recon_loss,mean_loss_sparsity,reg_log,boundary_reg_log)

            sae_diff,sae_static = eqx.partition(sae, eqx.is_inexact_array)
            loss_x,grads = compute_loss(sae_diff,sae_static,nca,x,y,t,key)
            updates,opt_state = self.OPTIMISER.update(grads, opt_state, sae_diff)
            sae = eqx.apply_updates(sae,updates)
            (loss_total, (x,losses,mean_recon_loss,mean_loss_sparsity,reg_log,boundary_reg_log)) = loss_x
            loss_dict = {"mean loss":loss_total,
                         "mean loss recon":mean_recon_loss,
                         "mean loss sparsity":mean_loss_sparsity,
                         "batch loss recon":losses,
                         "intermediate reg":reg_log,
                         "boundary reg":boundary_reg_log}
            return sae,nca,x,y,t,opt_state,key,loss_dict
        
        nca = self.NCA_model
        sae = self.SAE
        sae_diff,_ = eqx.partition(sae, eqx.is_inexact_array)
        if optimiser is None:
            schedule = optax.exponential_decay(1e-3, transition_steps=iters, decay_rate=0.99)
            self.OPTIMISER = optax.chain(
            optax.nadam(schedule),
            unitary_decoder_transform(norm=1.0,eps=1e-8,axis=1)
        )
            
        else:
            self.OPTIMISER = optimiser
        opt_state = self.OPTIMISER.init(sae_diff)
        x,y = self.DATA_AUGMENTER.data_load(key)
        best_loss = 100000000
        loss_thresh = 1e16
        model_saved = False
        loss_diff = 0
        #prev_loss = 0
        mean_loss = 0
        loss_diff_thresh = 1e-2
        error = 0
        error_at = 0

        pbar = tqdm(range(iters))
        for i in pbar:
            #prev_loss = mean_loss
            if i%CLEAR_CACHE_EVERY==0:
                #print(f"Clearing cache at step {i}")
                jax.clear_caches()
            key = jr.fold_in(key,i)

            
            sae,nca,x_new,y_new,t,opt_state,key,loss_dict = make_step(sae,nca,x,y,t,opt_state,key)
            mean_loss = loss_dict["mean loss"]
            loss_diff = mean_loss - best_loss
            pbar.set_postfix({'loss': mean_loss,'best loss': best_loss,'loss diff':loss_diff,'sparsity':loss_dict["mean loss sparsity"]})

            if self.IS_LOGGING:
                self.LOGGER.tb_training_loop_log_sequence(
                    loss_dict=loss_dict,
                    X_sae=x_new,
                    step=i,
                    sae=sae,
                    write_images=WRITE_IMAGES,
                    LOG_EVERY=LOG_EVERY)
            if jnp.isnan(mean_loss):
                error = 1
                error_at=i
                break
            elif any(list(map(lambda x: jnp.any(jnp.isnan(x)), x))):
                error = 2
                error_at=i
                break
            elif mean_loss>loss_thresh:
                error = 3
                error_at=i
                break
            if error==0:
                if loss_diff<loss_diff_thresh or i<WARMUP:
                    x,y = self.DATA_AUGMENTER.data_callback(x_new, y_new, i, key)
                
                
                # Save model whenever mean_loss beats the previous best loss
                if i>WARMUP:
                    if mean_loss < best_loss:
                        model_saved=True
                        self.SAE = sae
                        self.SAE.save(self.MODEL_PATH,overwrite=True)
                        best_loss = mean_loss

        if error==0:
            print("Training completed successfully")
        elif error==1:
            print("|-|-|-|-|-|-  Loss reached NaN at step "+str(error_at)+" -|-|-|-|-|-|")
        elif error==2:
            print("|-|-|-|-|-|-  X reached NaN at step "+str(error_at)+" -|-|-|-|-|-|")
        elif error==3:
            print( "|-|-|-|-|-|-  Loss exceded "+str(loss_thresh)+" at step "+str(error_at)+", optimisation probably diverging  -|-|-|-|-|-|")
        if error!=0 and model_saved==False:
            print("|-|-|-|-|-|-  Training did not converge, model was not saved  -|-|-|-|-|-|")
        elif self.IS_LOGGING and model_saved:
            x,y = self.DATA_AUGMENTER.split_x_y(1)
            x,y = self.DATA_AUGMENTER.data_callback(x,y,0,key)
            #try:
            self.LOGGER.tb_training_end_log(self.NCA_model,x,t*x[0].shape[0],self.BOUNDARY_CALLBACK)
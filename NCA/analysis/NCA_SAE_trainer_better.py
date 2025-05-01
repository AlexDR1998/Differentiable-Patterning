#from NCA.analysis.NCA_feature_extractor import NCA_Feature_Extractor_Texture, NCA_Feature_Extractor_Emoji
from NCA.analysis.NCA_SAE_class import SparseAutoencoder
from NCA.analysis.tensorboard_log import SAE_Train_better_log as SAE_Train_log
from Common.utils import key_pytree_gen
import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import datetime
import time
from tqdm import tqdm
import optax
from Common.trainer.loss import l2, l1, cosine, spectral

class NCA_SAE_Trainer(object):

    def __init__(self,
                 SAE: SparseAutoencoder,
                 filename = None,
                 model_directory="",
                 log_directory=""
                 ):
        
        self.SAE = SAE
        if filename is None:
            self.model_filename = model_directory+"SAE_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_directory = log_directory+"SAE_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/train"
        else:
            self.model_filename = model_directory+filename
            self.log_directory = log_directory+filename+"/train"


    def loss_reconstruction(self,X_base,X_sae):
        """
            Computes the loss between the base and sae activations
        """

        return self._loss(X_base[:,:self.OBS_CHANNELS],X_sae[:,:self.OBS_CHANNELS])
    
    def loss_reconstruction_normalised(self,X_base,X_sae,X_init):
        """
            Computes the loss between the base and sae activations
        """

        return self._loss(X_base[:,:self.OBS_CHANNELS],X_sae[:,:self.OBS_CHANNELS]) / (self._loss(X_base[:,:self.OBS_CHANNELS],X_init[:,:self.OBS_CHANNELS])+1e-8)
        
    def loss_sparsity(self,latents):
        
        return jnp.mean(jnp.abs(latents))

    def reset_x0(self,key): # DO NOT JIT - CAUSES OOM!?
        key_batch,key_time = jr.split(key,2)
        inds_batch = jr.choice(key_batch, self.FULL_TRAJECTORY.shape[0], shape=(self.MINIBATCH_SIZE,), replace=False)
        #inds_time = jr.choice(key_time, self.FULL_TRAJECTORY.shape[1]-self.PATH_LENGTH, shape=(self.MINIBATCH_SIZE,), replace=False)
        self.inds_time += jr.bernoulli(key_time, self.XO_TIME_UPDATE_PROBABILITY, shape=(self.MINIBATCH_SIZE,))
        self.inds_time = self.inds_time%(self.FULL_TRAJECTORY.shape[1]-self.PATH_LENGTH)
        #self.inds_time = self.inds_time.at[-1].set(0) # Always have one start from the first time step as that is hardest to learn
        x0 = self.FULL_TRAJECTORY[inds_batch,self.inds_time]
        
        return x0

    def train(self,
              NCA,
              X0,
              ITERS,
              SPARSITY,
              optimiser,
              MINIBATCH_SIZE=2,
              PATH_LENGTH=8,
              PATH_LOSS_SAMPLING="uniform",
              NCA_TIMESTEPS=256,
              LOG_EVERY=512,
              RESAMPLE_X0_EVERY=8,
              PROPAGATE_SAE_PATHS=1,
              LOSS="l2",
              LOSS_CHANNEL_MODE = "obs",
              NORMALISE_MODE="none", # "none", "stepwise", "pathwise"
              LOOP_AUTODIFF="checkpoint",
              CLEAR_CACHE_EVERY = 256,
              BOUNDARY_CALLBACK = lambda x:x,
              wandb_config={"project":"NCA_SAE","name":"SAE_Train_better"},
              key=jr.PRNGKey(int(time.time()))):
        
        """
            Trains the SAE to approximate internal activations of NCA, when that NCA is run from the given X0
        """
        if LOSS_CHANNEL_MODE == "obs":
            self.OBS_CHANNELS = 4
        elif LOSS_CHANNEL_MODE == "all":
            self.OBS_CHANNELS = NCA.N_CHANNELS
        if LOSS == "l2":
            self._loss = lambda x,y: jnp.mean(l2(x,y)) # extra mean for batch dimension here
        elif LOSS == "l1":
            self._loss = lambda x,y: jnp.mean(l1(x,y))
        elif LOSS == "cosine":
            self._loss = lambda x,y: jnp.mean(cosine(x,y))
        elif LOSS == "spectral":
            self._loss = lambda x,y: jnp.mean(spectral(x,y))        

        self.NORMALISE_MODE = NORMALISE_MODE
        self.PATH_LOSS_SAMPLING = PATH_LOSS_SAMPLING
        @eqx.filter_jit
        def make_step(SAE,NCA,X,T,opt_state,key):
            
            @eqx.filter_value_and_grad(has_aux=True)
            def compute_loss(SAE_diff,SAE_static,NCA,X,key):
                SAE = eqx.combine(SAE_diff,SAE_static)
                v_nca = jax.vmap(
                    NCA,
                    in_axes=(0,None,0), # No vmap over boundary callback for now
                    out_axes=(0)) # vmap over batches
                v_nca_SAE = jax.vmap(
                    lambda x,boundary,key:NCA.call_with_SAE(
                        x=x,
                        SAE=SAE,
                        boundary_callback=boundary,
                        key=key),
                    in_axes=(0,None,0), # No vmap over boundary callback for now
                    out_axes=(0,0,0,0)) # vmap over batches
                
                def nca_step(carry,j): # function of type a,b -> a
                    #Unpack carry
                    key,x_base,x_sae,x_init = carry
                    if self.NORMALISE_MODE == "stepwise":
                        x_init = x_base
                    
                    #Process PRNG keys
                    key = jr.fold_in(key,j)
                    key_base = jr.fold_in(key,j+1)
                    key_sae = jr.fold_in(key,j+2)
                    keys_base = jr.split(key_base,x_base.shape[0])
                    keys_sae = jr.split(key_sae,x_sae.shape[0])

                    #Do NCA steps
                    x_base = v_nca(x_base,self.BOUNDARY_CALLBACK,keys_base)
                    x_sae,latents,_,_ = v_nca_SAE(x_sae,self.BOUNDARY_CALLBACK,keys_sae)

                    #Compute loss and add to stack
                    if self.NORMALISE_MODE == "none":
                        loss_recon = self.loss_reconstruction(x_base,x_sae)
                    else:
                        loss_recon = self.loss_reconstruction_normalised(x_base,x_sae,x_init)

                    loss_sparsity = self.loss_sparsity(latents)
                    loss = loss_recon+SPARSITY*loss_sparsity
                    
                    carry = (key,x_base,x_sae,x_init)
                    stack = (loss_recon,loss_sparsity,loss)
                    return carry,stack
                
                (key,x_base,x_sae,x_init),(loss_recon,loss_sparsity,loss) = eqx.internal.scan(nca_step,(key,X,X,X),xs=jnp.arange(T),kind=LOOP_AUTODIFF)
                
                if self.PATH_LOSS_SAMPLING == "uniform":
                    mean_loss = jnp.mean(loss)
                    mean_loss_recon = jnp.mean(loss_recon)
                    mean_loss_sparsity = jnp.mean(loss_sparsity)
                elif self.PATH_LOSS_SAMPLING == "endpoint":
                    mean_loss = loss[-1]
                    mean_loss_recon = loss_recon[-1]
                    mean_loss_sparsity = loss_sparsity[-1]
                elif self.PATH_LOSS_SAMPLING == "geometric_forward":
                    mean_loss = jnp.average(loss,weights=jnp.arange(T,0,-1))
                    mean_loss_recon = jnp.average(loss_recon,weights=jnp.arange(T,0,-1))
                    mean_loss_sparsity = jnp.average(loss_sparsity,weights=jnp.arange(T,0,-1))
                elif self.PATH_LOSS_SAMPLING == "geometric_backward":
                    mean_loss = jnp.average(loss,weights=jnp.arange(1,T+1))
                    mean_loss_recon = jnp.average(loss_recon,weights=jnp.arange(1,T+1))
                    mean_loss_sparsity = jnp.average(loss_sparsity,weights=jnp.arange(1,T+1))

                return mean_loss,(mean_loss_recon,mean_loss_sparsity,x_sae)

            SAE_diff,SAE_static = SAE.partition()
            loss_x,grads = compute_loss(SAE_diff,SAE_static,NCA,X,key)
            updates,opt_state = self.OPTIMISER.update(grads, opt_state, SAE_diff)
            SAE = eqx.apply_updates(SAE,updates)
            return SAE,opt_state,loss_x
        
        #--------------------------------------
        training_config = {
            "iters": ITERS,
            "optimiser": optimiser,
            "MINIBATCH_SIZE": MINIBATCH_SIZE,
            "LOG_EVERY": LOG_EVERY,
            "SPARSITY": SPARSITY,
            "NCA_TIMESTEPS": NCA_TIMESTEPS,
            "PATH_LENGTH": PATH_LENGTH,
            "PROPAGATE_SAE_PATHS": PROPAGATE_SAE_PATHS,
            "NORMALISE_MODE": NORMALISE_MODE,
            "RESAMPLE_X0_EVERY": RESAMPLE_X0_EVERY,
            "CLEAR_CACHE_EVERY": CLEAR_CACHE_EVERY,
            "NCA":NCA.get_config(),
            "BOUNDARY_CALLBACK": BOUNDARY_CALLBACK,
            "LOOP_AUTODIFF": LOOP_AUTODIFF,
            "PATH_LOSS_SAMPLING": PATH_LOSS_SAMPLING,
            "LOSS": LOSS,
            "LOSS_CHANNEL_MODE": LOSS_CHANNEL_MODE,
        }
        _config = {
            "MODEL": self.SAE.config,
            "TRAINING": training_config,
        }
        wandb_config["config"] = _config

        self.BOUNDARY_CALLBACK = BOUNDARY_CALLBACK
        FULL_TRAJECTORY = [NCA.run(iters=NCA_TIMESTEPS,x=x0,key=key) for x0 in X0]
        FULL_TRAJECTORY = jnp.array(FULL_TRAJECTORY)
        self.FULL_TRAJECTORY = FULL_TRAJECTORY
        self.LOGGER = SAE_Train_log(data=FULL_TRAJECTORY,wandb_config=wandb_config)
        if MINIBATCH_SIZE is None:
            MINIBATCH_SIZE = FULL_TRAJECTORY.shape[0]
        if PATH_LENGTH is None:
            PATH_LENGTH = FULL_TRAJECTORY.shape[1]

        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.PATH_LENGTH = PATH_LENGTH
        #--------------------------------------

        if optimiser is None:
            schedule = optax.exponential_decay(1e-3, transition_steps=ITERS, decay_rate=0.99)
            self.OPTIMISER = optax.nadam(schedule)
            
        else:
            self.OPTIMISER = optimiser
        
        SAE = self.SAE
        SAE_diff,_ = SAE.partition()
        opt_state = self.OPTIMISER.init(SAE_diff)
        pbar = tqdm(range(ITERS))

        
        
        self.XO_TIME_UPDATE_PROBABILITY= 0.5
        self.inds_time = jnp.zeros((MINIBATCH_SIZE,),dtype=int)
        x0 = self.reset_x0(key)
        x_sae = x0
        best_loss = None
        for i in pbar:
                
            key = jr.fold_in(key,i)
            if i % CLEAR_CACHE_EVERY == 0 and i>0:
                jax.clear_caches()
            if i % RESAMPLE_X0_EVERY == 0 and i>0:
                x0 = self.reset_x0(key)
            else: # Only propagate the sae paths if we are not resampling - so they don't propagate forever
                if PROPAGATE_SAE_PATHS>0:
                    x0 = x0.at[jnp.arange(PROPAGATE_SAE_PATHS)].set(x_sae[jnp.arange(PROPAGATE_SAE_PATHS)])
            
            SAE,opt_state,loss_x = make_step(SAE,NCA,x0,PATH_LENGTH,opt_state,key)
            mean_loss,(mean_loss_recon,mean_loss_sparsity,x_sae) = loss_x
            if i==0:
                best_loss = mean_loss
            else:
                if mean_loss < best_loss:
                    best_loss = mean_loss
            pbar.set_postfix({'best_loss':best_loss,'loss': mean_loss,'reconstruction loss': mean_loss_recon,'sparsity loss':mean_loss_sparsity})
            self.LOGGER.tb_training_loop_log_sequence(L=[mean_loss,mean_loss_recon,mean_loss_sparsity],
                                                      X_sae=x_sae,
                                                      sae=SAE,
                                                      step=i,
                                                      write_images=True,
                                                      LOG_EVERY=LOG_EVERY)
            if jnp.isnan(mean_loss):
                print("NaN loss, stopping training")
                break
        X0 = FULL_TRAJECTORY[:,0]
        self.LOGGER.log_end(SAE,X0,NCA,NCA_TIMESTEPS,key)
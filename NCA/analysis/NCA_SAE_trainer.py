from NCA.analysis.NCA_feature_extractor import NCA_Feature_Extractor_Texture, NCA_Feature_Extractor_Emoji
from NCA.analysis.NCA_SAE_class import SparseAutoencoder
from NCA.analysis.tensorboard_log import SAE_Train_log
import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import datetime
import time
from tqdm import tqdm
import optax


class NCA_SAE_Trainer(object):

    def __init__(self,
                 Feature_Extractor: NCA_Feature_Extractor_Emoji,
                 SAE: SparseAutoencoder,
                 filename = None,
                 model_directory="",
                 log_directory=""):
        
        self.FE = Feature_Extractor
        self.SAE = SAE
        if filename is None:
            self.model_filename = model_directory+"SAE_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_directory = log_directory+"SAE_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/train"
        else:
            self.model_filename = model_directory+filename
            self.log_directory = log_directory+filename+"/train"
        
        
        #self.log_directory = log_directory+self.model_filename+"/train"
        
                
    def generate_activations(self,FE_params):
        X,activations = self.FE.extract_features(**FE_params)
        activations = self.FE.flatten_activations(activations)
        activations = activations[self.SAE.TARGET_LAYER]
        return activations

    

    def train(self,
              iters,
              Sparsity,
              optimiser,
              LOSS="l2",
              MINIBATCH_SIZE=4096,
              REGENERATE_EVERY=512,
              LOG_EVERY=512,
              FE_params={"t0":0,"t1":128,"BATCH_SIZE":1,"SIZE":32,"key":jr.PRNGKey(int(time.time()))},
              wandb_config={"project":"NCA_SAE","name":"SAE_Train"},
              key=jr.PRNGKey(int(time.time()))):
        @eqx.filter_jit
        def make_step(SAE,x,opt_state):

            @eqx.filter_value_and_grad(has_aux=True)
            def compute_loss(SAE_diff,SAE_static,X):
                SAE = eqx.combine(SAE_diff,SAE_static)
                vEnc = jax.vmap(SAE.encode,in_axes=0,out_axes=0)
                vDec = jax.vmap(SAE.decode,in_axes=0,out_axes=0)
                latent = vEnc(X)
                x_reconstructed = vDec(latent)
                loss_reconstruction = self._loss(X,x_reconstructed)#/jnp.mean(X**2,axis=1))
                loss_sparsity = jnp.mean(jnp.abs(latent))

                return loss_reconstruction+Sparsity*loss_sparsity,(loss_reconstruction,loss_sparsity)
            
            SAE_diff,SAE_static = SAE.partition()
            (loss,(loss_recon,loss_sparse)),grad = compute_loss(SAE_diff,SAE_static,x)
            updates,opt_state = optimiser.update(grad, opt_state, SAE)
            SAE = eqx.apply_updates(SAE,updates)
            return SAE,opt_state,loss,loss_recon,loss_sparse
    
        #--------------------------------------
        if LOSS == "l2":
            self._loss = lambda x,y: jnp.mean((x-y)**2)
        elif LOSS == "l1":
            self._loss = lambda x,y: jnp.mean(jnp.abs(x-y))
        elif LOSS == "cosine":
            self._loss = lambda x,y: 1-jnp.dot(x,y)/(jnp.linalg.norm(x)*jnp.linalg.norm(y))
        
        training_config = {
            "iters": iters,
            "optimiser": optimiser,
            "MINIBATCH_SIZE": MINIBATCH_SIZE,
            "REGENERATE_EVERY": REGENERATE_EVERY,
            "LOG_EVERY": LOG_EVERY,
            "FE_params": FE_params,
            "SPARSITY": Sparsity,

        }

        _config = {
            "MODEL": self.SAE.config,
            "TRAINING": training_config,
        }
        wandb_config["config"] = _config
        self.LOGGER = SAE_Train_log(data=(self.FE,FE_params),wandb_config=wandb_config)

        SAE = self.SAE
        SAE_diff,_ = SAE.partition()
        opt_state = optimiser.init(SAE_diff)


        pbar = tqdm(range(iters))
        activations = self.generate_activations(FE_params)


        for i in pbar:
            if i % REGENERATE_EVERY == 0 and i>0:
                FE_params["key"] = key
                activations = self.generate_activations(FE_params)
            key = jr.fold_in(key,i)
            inds = jr.choice(key,activations.shape[0],(MINIBATCH_SIZE,),replace=False)
            X = activations[inds]
            SAE,opt_state,loss,loss_recon,loss_sparse = make_step(SAE,X,opt_state)

            pbar.set_postfix({'loss': loss,'reconstruction loss': loss_recon,'sparsity loss':loss_sparse})
            self.LOGGER.tb_training_loop_log_sequence((loss,loss_recon,loss_sparse),
                                                      SAE,
                                                      i,
                                                      self.FE,
                                                      FE_params,
                                                      write_images=True,
                                                      LOG_EVERY=LOG_EVERY)
        
        #self.LOGGER.finish()
        self.SAE = SAE
        self.SAE.save(self.model_filename)
        X0 = self.FE.initial_condition(BATCH_SIZE=FE_params["BATCH_SIZE"])
        NCA = self.FE.NCA_models[0]
        NCA_TIMESTEPS = FE_params["t1"]
        key = jr.fold_in(key,1)
        self.LOGGER.log_end(SAE,X0,NCA,NCA_TIMESTEPS,key)
        

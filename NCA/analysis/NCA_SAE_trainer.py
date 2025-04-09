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
                 Sparsity: float,
                 filename = None,
                 model_directory="",
                 log_directory=""):
        
        self.FE = Feature_Extractor
        self.SAE = SAE
        self.Sparsity = Sparsity
        if filename is None:
            self.model_filename = model_directory+"SAE_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_directory = log_directory+"SAE_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/train"
        else:
            self.model_filename = model_directory+filename
            self.log_directory = log_directory+filename+"/train"
        
        
        #self.log_directory = log_directory+self.model_filename+"/train"
        self.LOGGER = SAE_Train_log(self.log_directory)
            
        
    
    def generate_activations(self,FE_params):
        X,activations = self.FE.extract_features(**FE_params)
        activations = self.FE.flatten_activations(activations)
        activations = activations[self.SAE.TARGET_LAYER]
        return activations



    def train(self,
              iters,
              optimiser,
              MINIBATCH_SIZE=4096,
              REGENERATE_EVERY=512,
              LOG_EVERY=512,
              FE_params={"t0":0,"t1":64,"BATCH_SIZE":1,"SIZE":32,"key":jr.PRNGKey(int(time.time()))},
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
                loss_reconstruction = jnp.mean((X-x_reconstructed)**2)#/jnp.mean(X**2,axis=1))
                loss_sparsity = jnp.mean(jnp.abs(latent))

                return loss_reconstruction+self.Sparsity*loss_sparsity,(loss_reconstruction,loss_sparsity)
            
            SAE_diff,SAE_static = SAE.partition()
            (loss,(loss_recon,loss_sparse)),grad = compute_loss(SAE_diff,SAE_static,x)
            updates,opt_state = optimiser.update(grad, opt_state, SAE)
            SAE = eqx.apply_updates(SAE,updates)
            return SAE,opt_state,loss,loss_recon,loss_sparse
    
        #--------------------------------------

        

        SAE = self.SAE
        SAE_diff,_ = SAE.partition()
        opt_state = optimiser.init(SAE_diff)


        pbar = tqdm(range(iters))
        activations = self.generate_activations(FE_params)


        for i in pbar:
            if i % REGENERATE_EVERY == 0 and i>0:
                FE_params["key"] = key
                #print("Regenerating activations")
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

        self.SAE = SAE
        self.SAE.save(self.model_filename)
        return SAE

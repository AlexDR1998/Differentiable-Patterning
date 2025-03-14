from NCA.analysis.NCA_feature_extractor import NCA_Feature_Extractor_Texture
from NCA.analysis.NCA_SAE_class import SparseAutoencoder
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
                 Feature_Extractor: NCA_Feature_Extractor_Texture,
                 SAE: SparseAutoencoder,
                 Sparsity: float,
                 filename = None,
                 file_path=""):
        self.FE = Feature_Extractor
        self.SAE = SAE
        self.Sparsity = Sparsity
        if filename is None:
            self.filename = file_path+"SAE_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.filename = file_path+filename
        
    
    def generate_activations(self,FE_params):
        X,activations = self.FE.extract_features(**FE_params)
        activations = self.FE.flatten_activations(activations)
        activations = activations[self.SAE.TARGET_LAYER]
        return activations

    def train(self,
              iters,
              optimiser,
              MINIBATCH_SIZE=4096,
              REGENERATE_EVERY=4096,
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

        loss_logs = [[],[],[]]
        for i in pbar:
            if i % REGENERATE_EVERY == 0:
                FE_params["key"] = key
                activations = self.generate_activations(FE_params)
            key = jr.fold_in(key,i)
            inds = jr.choice(key,activations.shape[0],(MINIBATCH_SIZE,),replace=False)
            X = activations[inds]
            SAE,opt_state,loss,loss_recon,loss_sparse = make_step(SAE,X,opt_state)
            pbar.set_postfix({'loss': loss,'reconstruction loss': loss_recon,'sparsity loss':loss_sparse})
            loss_logs[0].append(loss)
            loss_logs[1].append(loss_recon)
            loss_logs[2].append(loss_sparse)
        self.SAE = SAE
        self.SAE.save(self.filename)
        return SAE,loss_logs

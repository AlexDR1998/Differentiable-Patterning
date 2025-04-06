import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU") 
from Common.trainer.abstract_tensorboard_log import Train_log
import jax.numpy as jnp
import jax.random as jr
import jax
import equinox as eqx
from einops import rearrange


class SAE_Train_log(object):
    def __init__(self,log_dir):
        """
			Initialises the tensorboard logging of training.
			Writes some initial information. Very similar to setup_tb_log_single, but designed for sequence modelling

		"""

        self.LOG_DIR = log_dir
        train_summary_writer = tf.summary.create_file_writer(self.LOG_DIR)
        self.train_summary_writer = train_summary_writer
        print(f"Logging to {self.LOG_DIR}")

    def tb_training_loop_log_sequence(self,L,sae,step,FE,FE_params,write_images=True,LOG_EVERY=10):
        loss,loss_recon,loss_sparse = L
        
        with self.train_summary_writer.as_default():
 
            tf.summary.scalar("Total Loss",loss,step=step)
            tf.summary.scalar("Reconstruction Loss",loss_recon,step=step)
            tf.summary.scalar("Sparsity Loss",loss_sparse,step=step)


        if step%LOG_EVERY==0:
            self.log_model_parameters(sae,step)
            if write_images:
                self.log_model_outputs(sae,FE,FE_params,step)


    def log_model_parameters(self,sae,step):

        enc = sae.encoder.weight
        dec = sae.decoder.weight
        enc = rearrange(enc, "i j -> () i j ()")
        dec = rearrange(dec, "i j -> () i j ()")
        with self.train_summary_writer.as_default():
            #tf.summary.histogram('Encoder weights',enc,step=step)
            #tf.summary.histogram('Decoder weights',dec,step=step)
            tf.summary.image('Encoder weights',enc,step=step)
            tf.summary.image('Decoder weights',dec,step=step)

    def log_model_outputs(self,sae,FE,FE_params,step):
        latent_edit = {"mode":"None",
                        "positions":[],
                        "values":[]}
        
        X_FINALS = []
        key = FE_params["key"]
        key = jr.fold_in(key,step)
        @eqx.filter_jit
        def loop_body(X,nca,sae,latent_edit,key):
            X,_,_,_ = nca.call_with_SAE(X,
                                        SAE=sae,
                                        latent_edit=latent_edit,
                                        key=key)
            return X
        
        for nca in FE.NCA_models:
            X = FE.initial_condition(FE_params["SIZE"],1,key)[0]
            for i in range(FE_params["t1"]):
                key = jr.fold_in(key,i)
                X = loop_body(X,nca,sae,latent_edit,key)
                
                
                
            X_FINALS.append(X)
        X_FINALS = jnp.array(X_FINALS)
        X_FINALS = jnp.clip(X_FINALS,0,1)[:,:3]
        X_FINALS = rearrange(X_FINALS,"(m1 m2) c h w -> () (m1 h) (m2 w) c",m1=3)
        with self.train_summary_writer.as_default():
            tf.summary.image("NCA output with SAE replacement",X_FINALS,step=step)
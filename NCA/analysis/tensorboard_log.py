from Common.trainer.abstract_wandb_log import Train_log
import jax.numpy as jnp
import jax.random as jr
import jax
import equinox as eqx
from einops import rearrange


class SAE_Train_log(Train_log):
        
    def log_data_at_init(self, data):
        FE,FE_params = data
        X,Activations = FE.extract_features(**FE_params)
        print("X shape",X.shape)
        for key,value in Activations.items():
            print(f"{key} shape: {value.shape}")

        X = rearrange(X, "T B C H W -> B (T H) W C")[...,:3]
        self.log_image("NCA output without SAE",X,step=None)


    def tb_training_loop_log_sequence(self,L,sae,step,FE,FE_params,write_images=True,LOG_EVERY=10):
        loss,loss_recon,loss_sparse = L
        self.log_scalars({"Total Loss":loss,
                          "Reconstruction Loss":loss_recon,
                          "Sparsity Loss":loss_sparse},step=step)

        if step%LOG_EVERY==0:
            self.log_model_parameters(sae,step)
            if write_images:
                self.log_model_outputs(sae,FE,FE_params,step)


    def log_model_parameters(self,sae,step):

        enc = sae.encoder.weight
        dec = sae.decoder.weight
        enc = rearrange(enc, "i j -> i j ()")
        dec = rearrange(dec, "i j -> i j ()")
        
        self.log_image("Encoder weights",enc,step=step)
        self.log_image("Decoder weights",dec,step=step)
        self.log_histogram("Encoder weights",enc,step=step)
        self.log_histogram("Decoder weights",dec,step=step)

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
        #X_FINALS = rearrange(X_FINALS,"(m1 m2) c h w -> () (m1 h) (m2 w) c",m1=3)
        X_FINALS = rearrange(X_FINALS, "b c h w -> b h w c")
        #with self.train_summary_writer.as_default():
        #    tf.summary.image("NCA output with SAE replacement",X_FINALS,step=step)
        self.log_image("NCA output with SAE replacement",X_FINALS,step=step)
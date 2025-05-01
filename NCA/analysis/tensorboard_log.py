from Common.trainer.abstract_wandb_log import Train_log
from NCA.trainer.tensorboard_log import NCA_Train_log
from Common.utils import squarish
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
            X = FE.initial_condition(BATCH_SIZE=FE_params["BATCH_SIZE"])[0]
            for i in range(FE_params["t1"]):
                key = jr.fold_in(key,i)
                X = loop_body(X,nca,sae,latent_edit,key)
                
            X_FINALS.append(X)
        X_FINALS = jnp.array(X_FINALS)
        X_FINALS = jnp.clip(X_FINALS,0,1)[:,:3]
        X_FINALS = rearrange(X_FINALS, "b c h w -> b h w c")
        self.log_image("NCA output with SAE replacement",X_FINALS,step=step)


class SAE_Train_better_log(SAE_Train_log):
        
    def log_data_at_init(self, data):
         
        X = rearrange(data, "T B C H W -> B (T H) W C")[...,:3]
        self.log_image("NCA output without SAE",X,step=None)
    
    def tb_training_loop_log_sequence(self,L,X_sae,sae,step,write_images=True,LOG_EVERY=10):
        loss,loss_recon,loss_sparse = L
        self.log_scalars({"Total Loss":loss,
                          "Reconstruction Loss":loss_recon,
                          "Sparsity Loss":loss_sparse,
                          "Total Log Loss":jnp.log(loss),
                          "Reconstruction Log Loss":jnp.log(loss_recon),
                          "Sparsity Log Loss": jnp.log(loss_sparse)},
                          step=step)

        if step%LOG_EVERY==0:
            self.log_model_parameters(sae,step)
            if write_images:
                self.log_model_outputs(X_sae,step)



    def log_model_outputs(self, x, i):
        x = jnp.clip(x,0,1)[:,:3]
        x = rearrange(x, "b c h w -> b h w c")
        self.log_image("NCA output with SAE replacement",x,step=i)

    def log_end(self,SAE,X0,NCA,NCA_TIMESTEPS,key):
        def _run_with_sae(NCA,SAE,X,NCA_TIMESTEPS,key):
            T = []
            for i in range(NCA_TIMESTEPS):
                key = jr.fold_in(key,i)
                X,_,_,_ = NCA.call_with_SAE(X,SAE=SAE,key=key)
                T.append(X[:3])
            T = jnp.array(T)
            return T
        FULL_TRAJECTORY = [_run_with_sae(NCA,SAE,x0,NCA_TIMESTEPS,key) for x0 in X0]
        FULL_TRAJECTORY = jnp.array(FULL_TRAJECTORY)
        b1,b2 = squarish(FULL_TRAJECTORY.shape[0])
        FULL_TRAJECTORY = rearrange(FULL_TRAJECTORY, "(b1 b2) t c h w -> t c (b1 h) (b2 w)",b1=b1,b2=b2)
        self.log_video("Full trajectory with SAE replacement",FULL_TRAJECTORY,step=None)
        self.finish()

class SAE_Train_log_v3(NCA_Train_log):




    def tb_training_loop_log_sequence(self,loss_dict,X_sae,sae,step,write_images=True,LOG_EVERY=10):
        
        self.log_scalars({"Total Loss":loss_dict["mean loss"],
                          "Reconstruction Loss":loss_dict["mean loss recon"],
                          "Sparsity Loss":loss_dict["mean loss sparsity"],
                          "Intermediate regulariser":loss_dict["intermediate reg"],
                          "Boundary regulariser":loss_dict["boundary reg"],
                          },
                          step=step)

        if step%LOG_EVERY==0:
            self.log_model_parameters(sae,step)
            if write_images:
                self.log_model_outputs(X_sae,step)


    def log_data_at_init(self, data):
        pass

    def log_model_parameters(self, SAE, i):
        self.log_histogram("Encoder weights",SAE.encoder.weight,step=i)
        self.log_histogram("Decoder weights",SAE.decoder.weight,step=i)
        self.log_histogram("Encoder pre-bias",SAE.bias_params[0],step=i)
        self.log_histogram("Encoder post-bias",SAE.bias_params[1],step=i)
        self.log_histogram("Decoder bias",SAE.bias_params[2],step=i)
        

    def tb_training_end_log(self,
                            SAE,
                            NCA,
                            X0,
                            NCA_TIMESTEPS,
                            key,
                            boundary_callback):
        def _run_with_sae(NCA,SAE,X,NCA_TIMESTEPS,key):
            T = []
            for i in range(NCA_TIMESTEPS):
                key = jr.fold_in(key,i)
                X,_,_,_ = NCA.call_with_SAE(X,SAE=SAE,key=key,boundary_callback=boundary_callback)
                T.append(X[:3])
            T = jnp.array(T)
            return T
        FULL_TRAJECTORY = [_run_with_sae(NCA,SAE,x0,NCA_TIMESTEPS,key) for x0 in X0]
        FULL_TRAJECTORY = jnp.array(FULL_TRAJECTORY)
        b1,b2 = squarish(FULL_TRAJECTORY.shape[0])
        FULL_TRAJECTORY = rearrange(FULL_TRAJECTORY, "(b1 b2) t c h w -> t c (b1 h) (b2 w)",b1=b1,b2=b2)
        self.log_video("Full trajectory with SAE replacement",FULL_TRAJECTORY,step=None)
        self.finish()
            
    
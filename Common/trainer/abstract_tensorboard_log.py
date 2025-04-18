import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU") # Force tensorflow not to use GPU, as it's only logging data
import numpy as np






class Train_log(object):


    def __init__(self,log_dir,data,RGB_mode="RGB"):
        """
			Initialises the tensorboard logging of training.
			Writes some initial information. Very similar to setup_tb_log_single, but designed for sequence modelling

		"""

        self.LOG_DIR = log_dir
        self.RGB_mode = RGB_mode


        train_summary_writer = tf.summary.create_file_writer(self.LOG_DIR)

        #--- Log the target image and initial condtions
        with train_summary_writer.as_default():
            for i in range(len(data[0])):
                outputs = [im[i,:3] for im in data]
                tf.summary.image('True sequence RGB',np.einsum("ncxy->nxyc",outputs),step=i,max_outputs=len(outputs))
                #for b in range(len(data)):
                #    if self.RGB_mode=="RGB":
                        #tf.summary.image('True sequence RGB',np.einsum("ncxy->nxyc",data[0,:,:3,...]),step=0,max_outputs=data.shape[0])
                        #tf.summary.image('True sequence RGB',np.einsum("ncxy->nxyc",data[b][:,:3,...]),step=b,max_outputs=data[b].shape[0])
                    
                #        tf.summary.image('True sequence RGB, batch '+str(b),np.einsum("ncxy->nxyc",data[b][i][np.newaxis,:3,...]),step=i,max_outputs=data[b].shape[0])
                #elif self.RGB_mode=="RGBA":
                #    #tf.summary.image('True sequence RGBA',np.einsum("ncxy->nxyc",data[0,:,:4,...]),step=0,max_outputs=data.shape[0])
                #    tf.summary.image('True sequence RGBA',np.einsum("ncxy->nxyc",data[b][:,:4,...]),step=b,max_outputs=data[b].shape[0])
            
        self.train_summary_writer = train_summary_writer

    # def tb_training_loop_log_sequence(self,losses,x,i,model,write_images=True,LOG_EVERY=10):


    #     BATCHES = losses.shape[0]
    #     N = losses.shape[1]
    #     with self.train_summary_writer.as_default():
    #         tf.summary.histogram("Loss",losses,step=i)
    #         tf.summary.scalar("Average Loss",np.mean(losses),step=i)
    #         for n in range(N):
    #             tf.summary.histogram("Loss of each batch, timestep "+str(n),losses[:,n],step=i)
    #             tf.summary.scalar("Loss of averaged over each batch, timestep "+str(n),np.mean(losses[:,n]),step=i)
    #         for b in range(BATCHES):
    #             tf.summary.histogram("Loss of each timestep, batch "+str(b),losses[b],step=i)
    #             tf.summary.scalar("Loss of averaged over each timestep,  batch "+str(b),np.mean(losses[b]),step=i)

    #     if i%LOG_EVERY==0:
    #         self.log_model_parameters(model,i)
    #         if write_images:
    #             self.log_model_outputs(x,i)
    def log_image(self,tag,image,step):
        """
        Logs an image to tensorboard
        """
        image = np.array(image)
        with self.train_summary_writer.as_default():
            tf.summary.image(tag,image,step=step,max_outputs=image.shape[0])

    def log_scalar(self,tag,value,step):
        """
        Logs a scalar to tensorboard
        """
        with self.train_summary_writer.as_default():
            tf.summary.scalar(tag,value,step=step)
    
    def log_histogram(self,tag,value,step):
        """
        Logs a histogram to tensorboard
        """
        with self.train_summary_writer.as_default():
            tf.summary.histogram(tag,value,step=step)

    def normalise_images(self,x):
        """
        Normalises the images to [0,1] range for tensorboard logging
        """
        x = x - np.min(x)
        x = x / np.max(x)
        return x

    def tb_training_end_log(self,model,x,t,*args):
        raise NotImplementedError
    
    def log_model_parameters(self,model,i):
        raise NotImplementedError
    
    def log_model_outputs(self,x,i):
        raise NotImplementedError
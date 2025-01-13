import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU") # Force tensorflow not to use GPU, as it's only logging data
import jax.numpy as np
from einops import rearrange


class boundary_train_log(object):
    def __init__(self,log_dir):
        self.log_dir = log_dir
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)

    def tb_training_loop_log(self,loss,loss_dict,boundary,i,LOG_EVERY=10):
        with self.train_summary_writer.as_default():
            #tf.summary.histogram("Loss",losses,step=i)
            tf.summary.scalar("Average Loss",loss,step=i)
            tf.summary.scalar("Celltype min",loss_dict["celltype_min"],step=i)
            tf.summary.scalar("Celltype max",loss_dict["celltype_max"],step=i)
            tf.summary.scalar("Micropattern size",loss_dict["micropattern_size"],step=i)

        if i%LOG_EVERY==0:
            self.log_model_parameters(i)


    def log_model_parameters(self,boundary,i):
        mask = boundary.get_mask()
        with self.train_summary_writer.as_default():
            tf.summary.image("Mask",rearrange(mask,"c x y -> () x y c"),step=i)

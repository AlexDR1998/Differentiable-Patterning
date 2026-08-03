from Common.trainer.abstract_data_augmenter_array import DataAugmenterAbstract
from Common.mnist_reader import MnistDataloader
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import einops
import time
import numpy as np
import pandas as pd
from os.path  import join
from scipy import ndimage

class DataAugmenter(DataAugmenterAbstract):
    # Data format is (((B,N,N_agents,2),(B,N,N_agents,2)),(B,N,CHANNELS,width,height))
    #   ((agent_position,agent_velocity),pheremone_lattice)

    def __init__(self,int_list,BATCHES,lattice_size,N_agents,channels=4):
        
        # Get data file paths
        points_path = "../Data/mnist_point_cloud/train.csv"
        input_path = '../Data/mnist_images'
        training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
        training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

        # Load images
        mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
        (x_train, y_train), (_, _) = mnist_dataloader.load_data()
        x_train = np.array(x_train)
        x = x_train.reshape((60000,-1))
        data_image = np.vstack((y_train,x.T)).T
        images = pd.DataFrame(data_image)
        
        # Load point clouds
        points = pd.read_csv(points_path)
        
        
        # Randomly subsample "BATCHES" samples from each label type
        R = np.random.randint(0,1000000)
        images_sampled = images.groupby(0).sample(BATCHES,random_state=R)
        points_sampled = points.groupby("label").sample(BATCHES,random_state=R)

        # Select only labels that are in int_list
        images_sampled = images_sampled[images_sampled[0].isin(int_list)]
        points_sampled = points_sampled[points_sampled["label"].isin(int_list)]
        
        
        
        images_sampled = images_sampled.iloc[:,1:].to_numpy()
        print(images_sampled.shape)
        images_sampled = images_sampled.reshape((len(int_list),BATCHES,28,28))[:,:,::-1]
        images_sampled = np.einsum("nbxy->bnxy",images_sampled)
        images_sampled = images_sampled.astype(float)/255.0
        #images_resized = []
        #for i in range(images_sampled.shape[0]):
            
        #images_resized = jnp.array(images_resized)   

        
       
        #df_sampled = points.groupby("label").sample(BATCHES)
        #df_sampled = df_sampled[df_sampled["label"].isin(int_list)]
        
    
        key=jr.PRNGKey(int(time.time()))
        self.key = key
        number_of_points = points_sampled.filter(regex="y").gt(-1).sum(axis=1)
        min_number = number_of_points.min()
        
        #print("Minimum number of points: "+str(min_number))

        weights = points_sampled.filter(regex="v")
        weights = weights[weights>0]
        weights = weights.fillna(0)
        weights = weights.div(weights.sum(axis=1),axis=0)

        def sample_by_min(df,weights,i,N):
            #print(weights.iloc[i].to_numpy().shape)
            indexes = np.random.choice(np.arange(351),N,replace=False,p=weights.iloc[i])
            X = df.iloc[i].filter(regex="x")
            Y = df.iloc[i].filter(regex="y")
            X = X.iloc[indexes].to_numpy()
            Y = Y.iloc[indexes].to_numpy()
            return jnp.array([X,Y])

        def sample_more(df,weights,i,N):
            #print(weights.iloc[i].to_numpy().shape)
            indexes = np.random.choice(np.arange(351),N,replace=True,p=weights.iloc[i])
            X = df.iloc[i].filter(regex="x")
            Y = df.iloc[i].filter(regex="y")
            X = X.iloc[indexes].to_numpy()
            Y = Y.iloc[indexes].to_numpy()
            X = X + np.random.normal(loc=0,scale=0.3,size=X.shape)
            Y = Y + np.random.normal(loc=0,scale=0.3,size=Y.shape)
            return np.array([X,Y])
        
        pos = []
        ims = []
        for i in range(BATCHES):
            data_batch = []
            ims_batch = []
            for n in range(len(int_list)):
                data_batch.append(sample_more(points_sampled,weights,i+n*BATCHES,N_agents))
                ims_batch.append(ndimage.zoom(images_sampled[i,n],(lattice_size/28.0)))
                #data_batch.append(sample_by_min(df_sampled,weights,i+n*BATCHES,min_number))
            pos.append(data_batch)
            ims.append(ims_batch)
        ims = jnp.array(ims)
        pos = jnp.array(pos)
        print(ims.shape)
        print(pos.shape)
        #print("Agent positions: "+str(pos.shape))
        pos = pos* (lattice_size/28.0)
        vel = jr.uniform(key=key,shape=pos.shape,minval=-1,maxval=1)
        ph = jnp.zeros((BATCHES,len(int_list),channels,lattice_size,lattice_size))
        ph = ph.at[:,:,0].set(ims)

        self.OBS_CHANNELS = channels
        self.LATTICE_SIZE = lattice_size
        #ph = jnp.pad(ph,((0,0),(0,0),(0,hidden_channels),(0,0),(0,0)))
        
        data_true = ((pos,vel),ph)
        print(ph.shape)
        print(pos.shape)
        self.data_true = data_true
        self.data_saved = data_true

    def data_init(self):
        data = self.return_saved_data()
        #data = self.duplicate_batches(data, 4)
        self.save_data(data)
        
        return None
    def initialize_pool(self):
        return self.split_x_y()


    def callback_training(self,data,i):
        # called in the ABM iteration loop - so that parts of X are always fixed
        # Tuned such that it decays to identity transform after M epochs
        # NOTE this acts on un vmapped model
        ((a_p,a_v),ph) = data
        
        if i<100:
            _a = jnp.linspace(0,1,self.LATTICE_SIZE)
            alpha = (100-i)/100.0
            ph_x_grad = einops.repeat(_a,"h -> h w", w=self.LATTICE_SIZE)
            ph_y_grad = einops.repeat(_a,"w -> h w", h=self.LATTICE_SIZE)
            
            ph = ph.at[-1].set(alpha*ph_x_grad + (1-alpha)*ph[-1])
            ph = ph.at[-2].set(alpha*ph_y_grad + (1-alpha)*ph[-2])
            

        return ((a_p,a_v),ph)
    
    def duplicate_batches(self,data,B):
        ((a_p,a_v),ph) = data
        a_p = super.duplicate_batches(a_p,B)
        a_v = super.duplicate_batches(a_v,B)
        ph = super.duplicate_batches(ph,B)
        return ((a_p,a_v),ph)
    
    def split_x_y(self, N_steps=1):
        ((a_p,a_v),ph) = self.data_saved
        X = ((a_p[:,:-N_steps],a_v[:,:-N_steps]),ph[:,:-N_steps])
        Y = ((a_p[:,N_steps:],a_v[:,N_steps:]),ph[:,N_steps:])
        return X,Y
    @eqx.filter_jit
    def advance_pool(self, x, y, i,key):
        self.key = jr.fold_in(self.key,i)
        key1,key2 = jr.split(self.key)
        ((x_p,x_v),x_ph) = x
        #((y_p,y_v),y_ph) = y

        x_p = x_p.at[:,1:].set(x_p[:,:-1]) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
        x_v = x_v.at[:,1:].set(x_v[:,:-1])
        x_ph = x_ph.at[:,1:].set(x_ph[:,:-1])
        ((x_p_true,x_v_true),x_ph_true),_ =self.split_x_y(1)
        x_p = x_p.at[:,0].set(x_p_true[:,0])
        x_v = x_v.at[:,0].set(jr.uniform(key=key1,shape=x_v[:,0].shape,minval=-1,maxval=1))
        #x_ph = x_ph.at[:,0].set(jr.uniform(key=key2,shape=x_ph[:,0].shape,minval=0,maxval=1))
        x_ph = x_ph.at[:,0].set(x_ph_true[:,0])
        return ((x_p,x_v),x_ph),y
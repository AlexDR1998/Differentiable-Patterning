import numpy as np
from Common.trainer.loss import l2,euclidean,vgg,spectral,random_sampled_euclidean,spectral_weighted
import jax
import optax


"""
    Functions that take in the Eddie job array index and return a tuple of training hyperparameters
"""

def index_to_learnrate_parameters(index):
	"""
		Takes job array index from 1-84 and constructs 4-tuple of learn rate, optimiser, training mode and gradient normalisation

	"""
	learn_rates = np.logspace(1,-5,7)
	learn_rates_name = ["1e1","1e0","1e-1","1e-2","1e-3","1e-4","1e-5"]
	optimisers = ["Adagrad",
				  "Adam",
				  "Nadam"]
	training_modes = ["differential","full"]
	grad_norm = [True,False]
	L1 = len(learn_rates)
	L2 = len(optimisers)
	L3 = len(training_modes)
	L4 = len(grad_norm)

	indices = np.unravel_index(index,(L1,L2,L3,L4))
	lr= learn_rates[indices[0]]
	lr_name = learn_rates_name[indices[0]]
	opt = optimisers[indices[1]]
	mode = training_modes[indices[2]]
	grad = grad_norm[indices[3]]
	return lr,lr_name,opt,mode,grad


def index_to_Nadam_parameters(index):
	"""
		Takes index from 1-N and constructs 3-tuple of learn rate, NCA to PDE step ratio and grad_norm

	"""
	learn_rates = np.logspace(1,-5,7)
	learn_rates_name = ["1e1","1e0","1e-1","1e-2","1e-3","1e-4","1e-5"]
	ratios = [1,2,4,8]
	grad_norm = [True,False]
	L1 = len(learn_rates)
	L2 = len(ratios)
	L3 = len(grad_norm)
	indices = np.unravel_index(index,(L1,L2,L3))
	lr = learn_rates[indices[0]]
	lr_name = learn_rates_name[indices[0]]
	ratio = ratios[indices[1]]
	grad = grad_norm[indices[2]]
	return lr,lr_name,ratio,grad



def index_to_mitosis_parameters(index):
	"""
		Takes index from 1-N and constructs 2-tuple of loss function and time sampling rate 
	"""
	loss_funcs = [loss_spectral,
				  loss_bhattacharyya_modified,
				  loss_hellinger_modified,
				  None]
	loss_func_name = ["spectral","bhattachryya","hellinger","euclidean"]
	sampling_rates = [1,2,4,6,8,12,16,24,32]
	instance = [0,1,2,3]
	L1 = len(loss_funcs)
	L2 = len(sampling_rates)
	L3 = len(instance)
	indices = np.unravel_index(index,(L1,L2,L3))
	loss = loss_funcs[indices[0]]
	loss_name = loss_func_name[indices[0]]
	sampling = sampling_rates[indices[1]]
	i = instance[indices[2]]
	return loss,loss_name,sampling,i

def index_to_generalise_test(index):
    loss_funcs = [None,
                  loss_bhattacharyya_modified,
                  loss_bhattacharyya_euclidean,
				  loss_spectral,
				  loss_hellinger_modified,
				  loss_kl_divergence]
    loss_func_names = ["euclidean",
                       "bhattacharyya",
                       "bhattacharyya_euclidean",
					   "spectral",
					   "hellinger",
					   "kl_divergence"]
    sampling_rates = [1,2,4,8,16]
    tasks = ["heat",
            "mitosis",
            "coral",
			"gol"]
    #sampling_rates = [16,32,48,64]
    #tasks=["emoji"]
    L1 = len(loss_funcs)
    L2 = len(sampling_rates)
    L3 = len(tasks)
    indices = np.unravel_index(index,(L1,L2,L3))
    loss = loss_funcs[indices[0]]
    loss_name = loss_func_names[indices[0]]
    sampling = sampling_rates[indices[1]]
    task = tasks[indices[2]]
    return loss,loss_name,sampling,task

def index_to_generalise_test_2():
	#Rerunning some of index_tp_generalise_test, as fft loss was bugged
	loss = loss_spectral
	loss_name = "spectral"
	sampling = 8
	task = "coral"
	return loss,loss_name,sampling,task
def index_to_model_exploration_parameters(index):
    loss_funcs = [None,loss_spectral]
    loss_func_names = ["euclidean","spectral"]
    tasks = ["heat","mitosis","coral","emoji"]
    layers = [2,3]
    kernels = ["ID_LAP","ID_LAP_AV","ID_DIFF_LAP","ID_DIFF"]
    activations=["linear","relu","swish","tanh"]
    L1 = len(loss_funcs)
    L2 = len(tasks)
    L3 = len(layers)
    L4 = len(kernels)
    L5 = len(activations)
    indices = np.unravel_index(index, (L1,L2,L3,L4,L5))
    LOSS = loss_funcs[indices[0]]
    LOSS_STR = loss_func_names[indices[0]]
    TASK = tasks[indices[1]]
    LAYER = layers[indices[2]]
    KERNEL= kernels[indices[3]]
    ACTIVATION= activations[indices[4]]
    return LOSS,LOSS_STR,TASK,LAYER,KERNEL,ACTIVATION
    

def index_to_activations_and_kernels(index):
	
	
	
	
	
	kernels = [["ID","LAP"],
			   ["ID","LAP","AV"],
			   ["ID","DIFF","LAP"],
			   ["ID","DIFF"]]
	activations=["linear","relu","swish","tanh","leaky_relu","gelu"]
	
	L1 = len(kernels)
	L2 = len(activations)
	indices = np.unravel_index(index, (L1,L2))
	
	KERNEL= kernels[indices[0]]
	ACTIVATION= activations[indices[1]]
	return KERNEL,ACTIVATION

def index_to_emoji_symmetry_parameters(index):
	kernels = ["ID_LAP_AV","ID_DIFF_LAP"]
	tasks=["normal","rotated_data","rotation_task","translation_task"]
	L1 = len(kernels)
	L2 = len(tasks)
	indices = np.unravel_index(index,(L1,L2))
	KERNEL = kernels[indices[0]]
	TASK = tasks[indices[1]]
	return KERNEL,TASK
	
def index_to_channel_sample(index):
	channels = [4,6,8,10,12,14,16,32]
	samplings = [1,2,4,8,16,32,64]
	L1 = len(channels)
	L2 = len(samplings)
	indices = np.unravel_index(index,(L1,L2))
	N_CHANNELS = channels[indices[0]]
	SAMPLING = samplings[indices[1]]
	return N_CHANNELS,SAMPLING


def index_to_channel(index):
	channels = [4,6,8,10,12,14,16,32]
	SAMPLING=64
	N_CHANNELS = channels[index]
	return N_CHANNELS,SAMPLING
	

def index_to_sample(index):
	N_CHANNELS = 16
	samplings = [1,2,4,8,16,32,128]
	SAMPLING=samplings[index]
	return N_CHANNELS,SAMPLING


def index_to_data_nca_type(index):
	indices = np.unravel_index(index,(4,4))
	data_index = indices[0]
	nca_type_index = indices[1]
	return data_index,nca_type_index

def index_to_data_nca_type_multi_species(index):
	indices = np.unravel_index(index,(5,2))
	data_index = indices[0]
	nca_type_index = indices[1]
	return data_index,nca_type_index


def index_to_kaNCA_hyperparameters(index):
	indices = np.unravel_index(index,(2,3,8,8,2))
	LEARN_RATE = [1e-4,1e-3][indices[0]]
	
	OPTIMISER = [optax.nadam,optax.nadamw,optax.lamb][indices[1]]
	BASIS_RESOLUTION = [2,3,4,5,8,11,16,25][indices[2]]
	BASIS_WIDTH = [0.1,0.5,1,2,4,8,12,16][indices[3]]
	INIT_SCALE = [0.01,0.1][indices[4]]
	LEARN_RATE_TEXT = ["1e-4","1e-3"][indices[0]]
	OPTIMISER_TEXT = ["nadam","nadamw","lamb"][indices[1]]
	BASIS_WIDTH_TEXT = ["1e-1","5e-1","1","2","4","8","12","16"][indices[3]]
	INIT_SCALE_TEXT = ["1e-2","1e-1"][indices[4]]
	return LEARN_RATE,OPTIMISER,BASIS_RESOLUTION,BASIS_WIDTH,INIT_SCALE,LEARN_RATE_TEXT,OPTIMISER_TEXT,BASIS_WIDTH_TEXT,INIT_SCALE_TEXT


def index_to_pde_hyperparameters(index):
	indices = np.unravel_index(index,(2,4,1,3,2,4,2))
	
	INNER_ACTIVATIONS = [jax.nn.relu,jax.nn.tanh][indices[0]]
	OUTER_ACTIVATIONS = [jax.nn.tanh,jax.nn.sigmoid,jax.nn.relu,lambda x:x][indices[1]]
	#INIT_SCALES = [0.01,0.1][indices[2]]
	#STABILITY_FACTOR = [0.01,0.1][indices[3]]
	INIT_SCALES = 0.5
	#INIT_SCALE_TEXT = "1e-1"
	STABILITY_FACTOR = 0.5
	#STABILITY_FACTOR_TEXT = "1e-1"
	OPTIMISER = [optax.nadam,optax.nadamw][0]#[indices[2]]
	LEARN_RATES = [1e-4,1e-3][indices[2]]
	TRAJECTORY_LENGTH = [8,32,64][indices[3]]
	USE_BIAS = [True,False][indices[4]]
	INNER_TEXT = ["relu","tanh"][indices[0]]
	OUTER_TEXT = ["tanh","sigmoid","relu","identity"][indices[1]]
	OPTIMISER_TEXT = ["nadam","nadamw"][0]#[indices[2]]
	LEARN_RATE_TEXT = ["1e4","1e3"][indices[2]]
	EQUATION_INDEX = indices[5]
	ZERO_INIT = indices[6]
	return [
		INNER_ACTIVATIONS,
		OUTER_ACTIVATIONS,
		INIT_SCALES,
		STABILITY_FACTOR,
		OPTIMISER,
		LEARN_RATES,
		TRAJECTORY_LENGTH,
		USE_BIAS,
		INNER_TEXT,
		OUTER_TEXT,
		OPTIMISER_TEXT,
		LEARN_RATE_TEXT,
		EQUATION_INDEX,
		ZERO_INIT]



def index_to_pde_advection_hyperparameters(index):
	indices = np.unravel_index(index,(2,2,2,2,3,4))
	INTERNAL_ACTIVATIONS = [jax.nn.relu,jax.nn.tanh][indices[0]]
	ADVECTION_OUTER_ACTIVATIONS = [jax.nn.relu,jax.nn.tanh][indices[1]]
	OPTIMISER = [optax.nadam,optax.nadamw][indices[2]]
	LEARN_RATE = [1e-4,1e-3][indices[3]]
	TRAJECTORY_LENGTH = [8,32,64][indices[4]]
	EQUATION_INDEX = indices[5]

	INTERNAL_TEXT = ["relu","tanh"][indices[0]]
	OUTER_TEXT = ["relu","tanh"][indices[1]]

	OPTIMISER_TEXT = ["nadam","nadamw"][indices[2]]
	LEARN_RATE_TEXT = ["1e-4","1e-3"][indices[3]]
	params = {"INTERNAL_ACTIVATIONS":INTERNAL_ACTIVATIONS,
		   	  "ADVECTION_OUTER_ACTIVATIONS":ADVECTION_OUTER_ACTIVATIONS,
			  "OPTIMISER":OPTIMISER,
			  "LEARN_RATE":LEARN_RATE,
			  "TRAJECTORY_LENGTH":TRAJECTORY_LENGTH,
			  "INTERNAL_TEXT":INTERNAL_TEXT,
			  "OUTER_TEXT":OUTER_TEXT,
			  "OPTIMISER_TEXT":OPTIMISER_TEXT,
			  "LEARN_RATE_TEXT":LEARN_RATE_TEXT,
			  "EQUATION_INDEX":EQUATION_INDEX}
	return params



def index_to_pde_gray_scott_hyperparameters(index):
	indices = np.unravel_index(index,(2,2,2,2))
	INTERNAL_ACTIVATIONS = ["linear","tanh"][1]
	LOSS_FUNCTION = [euclidean,spectral_weighted][1]

	REACTION_RATIO = [1,0.1,0.01][1]
	ADVECTION_RATIO = [1,0.1,0][0]
	REACTION_ZERO_INIT = [True,False][1]
	ADVECTION_ZERO_INIT = [True,False][1]
	DIFFUSION_ZERO_INIT = [True,False][1]
	REACTION_INIT = ["orthogonal","permuted"][indices[0]]
	DIFFUSION_INIT = ["orthogonal","diagonal"][indices[1]]
	#TRAJECTORY_LENGTH =  [4,4,16,16,16,32,32,32,64,64,64,64][indices[3]]
	#LOSS_TIME_SAMPLING = [1,2,1, 4, 8, 1, 4, 8, 1 ,8, 16,32][indices[3]]
	TRAJECTORY_LENGTH = 16
	LOSS_TIME_SAMPLING = 1
	N_LAYERS = 3
	ORDER = 2
	OPTIMISER = [optax.nadam,optax.nadamw][indices[2]]
	OPTIMISER_PRE_PROCESS = [optax.identity(),optax.scale_by_param_block_norm(),optax.adaptive_grad_clip(1.0)][indices[3]]
	TIME_RESOLUTION = [51,101,201][1]

	INTERNAL_ACTIVATIONS_TEXT = ["none","tanh"][1]
	LOSS_FUNCTION_TEXT = ["euclidean_","spectral_weighted_"][1]
	REACTION_RATIO_TEXT = ["1","1e-1","1e-2"][1]
	ADVECTION_RATIO_TEXT = ["1","1e-1","0"][0]
	REACTION_ZERO_INIT_TEXT = ["_zero_init",""][1]
	ADVECTION_ZERO_INIT_TEXT = ["_zero_init",""][1]
	DIFFUSION_ZERO_INIT_TEXT = ["_zero_init",""][1]
	
	#OPTIMISER_PRE_PROCESS_TEXT = ["none","scale_by_param_block_norm","adaptive_grad_clip"][indices[3]]
	OPTIMISER_TEXT = ["nadam","nadamw"][indices[2]]
	OPTIMISER_PRE_PROCESS_TEXT = ["","_scale_by_param_block_norm","_adaptive_grad_clip"][indices[3]]
	params = {
		"LOSS_FUNCTION":LOSS_FUNCTION,
		"ORDER":ORDER,
		"OPTIMISER":OPTIMISER,
		"TIME_RESOLUTION":TIME_RESOLUTION,
		"OPTIMISER_PRE_PROCESS":OPTIMISER_PRE_PROCESS,
		"OPTIMISER_TEXT":LOSS_FUNCTION_TEXT+OPTIMISER_TEXT+OPTIMISER_PRE_PROCESS_TEXT,
		"LOSS_TIME_SAMPLING":LOSS_TIME_SAMPLING,
		"INTERNAL_ACTIVATIONS":INTERNAL_ACTIVATIONS,
		"REACTION_RATIO":REACTION_RATIO,
		"ADVECTION_RATIO":ADVECTION_RATIO,
		"REACTION_INIT":REACTION_INIT,
		"DIFFUSION_INIT":DIFFUSION_INIT,
		"INTERNAL_ACTIVATIONS_TEXT":INTERNAL_ACTIVATIONS_TEXT,
		"REACTION_RATIO_TEXT":REACTION_RATIO_TEXT,
		"ADVECTION_RATIO_TEXT":ADVECTION_RATIO_TEXT,
		"REACTION_ZERO_INIT":REACTION_ZERO_INIT,
		"ADVECTION_ZERO_INIT":ADVECTION_ZERO_INIT,
		"DIFFUSION_ZERO_INIT":DIFFUSION_ZERO_INIT,
		"REACTION_ZERO_INIT_TEXT":REACTION_ZERO_INIT_TEXT,
		"ADVECTION_ZERO_INIT_TEXT":ADVECTION_ZERO_INIT_TEXT,
		"DIFFUSION_ZERO_INIT_TEXT":DIFFUSION_ZERO_INIT_TEXT,
		"N_LAYERS":N_LAYERS,
		"TRAJECTORY_LENGTH":TRAJECTORY_LENGTH,
		}
	return params




def index_to_pde_gray_scott_rda(index):
	indices = np.unravel_index(index,(8))
	params = {
		"LOSS_FUNCTION":[euclidean,spectral_weighted][0],
		"OPTIMISER":[optax.nadam,optax.nadamw][1],
		"OPTIMISER_PRE_PROCESS":[optax.identity(),optax.scale_by_param_block_norm()][1],
		"TEXT_LABEL":["_euclidean","_spectral_weighted"][0]+["_nadam","_nadamw"][1]+["","_scale_by_param_block_norm"][1],
		"INTERNAL_ACTIVATION":["tanh","relu6"][0],
		
		"TERMS":[["reaction_pure","diffusion_linear"],
		   		 ["reaction_pure","diffusion"],
				 ["reaction_pure","advection","diffusion_linear"],
				 ["reaction_pure","advection","diffusion"]][0],
		"REACTION_INIT":"orthogonal",
		"DIFFUSION_INIT":"orthogonal",
		"PDE_STR":["cahn_hilliard","gray_scott","keller_segel"][2],
		"PDE_SOLVER":["euler","heun"][1],
		"PDE_SOLVER_ADAPTIVE":[False,True][1],
		"N_LAYERS":[1,2,3,4][2],
		"ORDER":1,
		"NOISE_FRACTION":[0,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5][indices[0]],
		"NOISE_STRING":["0","001","005","010","015","020","025","030","035","040","045","050"][indices[0]],
		"TIME_RESOLUTION":[2,4,6,8,10,12,14,16][-1],
		"CHANNELS":16,
		"TRAJECTORY_FULL":[False,True][0],
		"TRAJECTORY_TYPE":["end","full"][0],
		"TRAJECTORY_LENGTH":[1,2,3,4,5,6,7,8][-1],
		"LOSS_TIME_SAMPLING":1
	}
	return params


def index_to_pde_texture_hyperparameters(index):
	indices = np.unravel_index(index,(4,3,2,2,2,2))
	filename = ["honeycombed/honeycombed_0078.jpg",
			    "banded/banded_0109.jpg",
				"dotted/dotted_0116.jpg",
				"interlaced/interlaced_0172.jpg"][indices[0]]
	filename_short = ["honeycombed",
			    	  "banded",
					  "dotted",
					  "interlaced"][indices[0]]
	

	#n_layers = [1,2][indices[2]]
	OPTIMISER_PRE_PROCESS = [optax.identity(),optax.scale_by_param_block_norm(),optax.adaptive_grad_clip(1.0)][indices[1]]
	OPTIMISER_PRE_PROCESS_TEXT = ["","_scale_by_param_block_norm","_adaptive_grad_clip"][indices[1]]
	REACTION_INIT = ["orthogonal","permuted"][indices[2]]
	DIFFUSION_INIT = ["orthogonal","diagonal"][indices[3]]
	ADVECTION_INIT = ["orthogonal","permuted"][indices[4]]
	n_layers = [2,3][indices[5]]

	return {"FILENAME":filename,
		 	"FILENAME_SHORT":filename_short,
			"N_LAYERS":n_layers,
			"REACTION_INIT":REACTION_INIT,
			"DIFFUSION_INIT":DIFFUSION_INIT,
			"ADVECTION_INIT":ADVECTION_INIT,
			"OPTIMISER_PRE_PROCESS":OPTIMISER_PRE_PROCESS,
			"OPTIMISER_PRE_PROCESS_TEXT":OPTIMISER_PRE_PROCESS_TEXT}
def index_to_kaNCA_pde_parameters(index):
	indices = np.unravel_index(index,(4,4))
	EQUATION_INDEX = indices[0]
	
	TIME_SAMPLING = [8,16,32,64][indices[1]]

	return EQUATION_INDEX,TIME_SAMPLING
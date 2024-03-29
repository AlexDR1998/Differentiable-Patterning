import jax
import equinox as eqx
import optax
import time
from ABM.model.neural_slime_angle import NeuralSlimeAngle
from ABM.model.neural_chemotaxis import NeuralChemotaxis
from ABM.model.neural_slime import NeuralSlime
from ABM.trainer.slime_trainer import SlimeTrainer
from ABM.ABM_visualiser import my_animate_agents
from tqdm import tqdm
from Common.trainer.loss import vgg
from Common.utils import load_textures
from Common.utils import key_array_gen
import matplotlib.pyplot as plt


#data = load_textures(["grid/grid_0057.jpg"],downsample=4,crop_square=True)
#print(data.shape)
#imsize = data.shape[-1]


timesteps = 128
resolution = 64
warmup=10
init_val = 1e-3
peak_val = 5e-3

cooldown = 60
iters=5*warmup+5*cooldown

N_agents = 2000
nslime = NeuralChemotaxis(N_agents, resolution, 16,dt=0.1,decay_rate=0.98,PERIODIC=True,gaussian_blur=1)


schedule = optax.exponential_decay(1e-2, transition_steps=iters, decay_rate=0.99)
# schedule = optax.sgdr_schedule([{"init_value":init_val, "peak_value":peak_val, "decay_steps":cooldown, "warmup_steps":warmup, "end_value":init_val},
#                                 {"init_value":init_val, "peak_value":peak_val, "decay_steps":cooldown, "warmup_steps":warmup, "end_value":init_val/2.0},
#                                 {"init_value":init_val/2.0, "peak_value":peak_val/2.0, "decay_steps":cooldown, "warmup_steps":warmup, "end_value":init_val/4.0},
#                                 {"init_value":init_val/4.0, "peak_value":peak_val/4.0, "decay_steps":cooldown, "warmup_steps":warmup, "end_value":init_val/8.0},
#                                {"init_value":init_val/8.0, "peak_value":peak_val/8.0, "decay_steps":cooldown, "warmup_steps":warmup, "end_value":init_val/16.0},])
#optimiser = optax.adamw(schedule)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.adam(schedule))
#optimiser = optax.chain(optax.scale(),
#                        optax.adam(schedule))


trainer = SlimeTrainer(nslime,[0,1],BATCHES=1,N_agents=N_agents,model_filename="ant_sinkhorn_test_19",alpha=0.0)
trainer.train(timesteps,iters,WARMUP=warmup,optimiser=optimiser)
nslime = trainer.nslime



# #v_nslime = jax.vmap(nslime,in_axes=(0),out_axes=(0),axis_name="A")
# v_nslime = lambda states: jax.tree_util.tree_map(nslime,states)

# v_init = lambda key_tree: jax.tree_util.tree_map(lambda key:nslime.init_state(key,zero_pheremone=True),key_tree)





# def run_nslime(state,nslime):
# 	#print(key)
# 	#trajectory = []
# 	#agent_trajectory = []
	
# 	for i in tqdm(range(timesteps)):
# 		#agents,pheremone_lattice = state
# 		#trajectory.append(pheremone_lattice)
# 		#agent_trajectory.append(agents)
# 		state = nslime(state)
# 	return state


# v_init = jax.vmap(lambda key:nslime.init_state(key,zero_pheremone=True),in_axes=(0),out_axes=(0,0)) 
# vv_init = jax.vmap(v_init,in_axes=(0),out_axes=(0,0))

# v_run = jax.vmap(run_nslime,in_axes=(0,None),out_axes=(0,0))
# vv_run = jax.vmap(v_run,in_axes=(0,None),out_axes=(0,0))
# key=jax.random.PRNGKey(int(time.time()))
# keys = key_array_gen(key,(3,5,))

# state = vv_init(keys)
# state = vv_run(state,nslime)


# print(jax.tree_util.tree_structure(state))
# print(state[0][0].shape)
# print(state[0][1].shape)
# print(state[1].shape)
# #print(keys)
# states = v_init(keys)
# print(jax.tree_util.tree_structure(states))
# print(states[0][0][0].shape)
# print(states[0][0][1].shape)
# print(states[0][1].shape)
# for i in tqdm(range(timesteps)):
#     states = v_nslime(states)





# trajectory = []
# agent_trajectory = []
# #nslime = NeuralSlime(1000, imsize, 12,dt=0.5,decay_rate=1.0)
# agents,pheremone_lattice = nslime.init_state()
# state = nslime.init_state(zero_pheremone=True)

# for i in tqdm(range(timesteps)):
# 	agents,pheremone_lattice = state
# 	trajectory.append(pheremone_lattice)
# 	agent_trajectory.append(agents)
# 	state = nslime(state)
# print(trajectory[0].shape)

# a = my_animate_agents(trajectory,agent_trajectory)
# plt.show()








# N=8
# X = jax.random.uniform(jax.random.PRNGKey(0), shape=(2,N,3,128,128))
# Y = jax.random.uniform(jax.random.PRNGKey(1), shape=(2,N,3,128,128))

# v_alexnet = jax.vmap(vgg,in_axes=(0,0,None),out_axes=0,axis_name="Batch")
# print(v_alexnet(X, Y, jax.random.PRNGKey(2)))
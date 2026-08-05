import math

import jax
import jax.numpy as jnp
import numpy as np

from NCA.NCA_visualiser import plot_to_image
from NCA.trainer.tensorboard_log import NCA_Train_log


def uses_fast_kan_diagnostics(model):
	return (
		hasattr(model, "get_edge_norms")
		and hasattr(model, "evaluate_edge_functions")
	)


def _kan_layer(model, layer_index):
	layer = model.layers[layer_index]
	return getattr(layer, "layer", layer)


def _kan_layer_width(layer):
	if hasattr(layer, "_width"):
		return float(jax.device_get(layer._width()))
	if getattr(layer, "log_rbf_width", None) is not None:
		return float(np.exp(np.array(jax.device_get(layer.log_rbf_width))))
	return float(getattr(layer, "rbf_width", np.nan))


def _normalise_label(label, max_length=32):
	label = str(label)
	if len(label) <= max_length:
		return label
	return label[: max_length - 3] + "..."


def _flatten_channel_first_samples(x):
	x = np.asarray(jax.device_get(x))
	return np.moveaxis(x, 0, -1).reshape(-1, x.shape[0])


def _subsample_rows(x, max_samples):
	if x.shape[0] <= max_samples:
		return x
	indices = np.linspace(0, x.shape[0] - 1, max_samples, dtype=np.int32)
	return x[indices]


def _fraction_abs_below(x, eps):
	x = np.asarray(x)
	if x.size == 0:
		return 0.0
	return float(np.mean(np.abs(x) < eps))


class kaNCA_Train_log(NCA_Train_log):
	def _log_legacy_kan_parameters(self,nca,i):
		#Log weights and biasses of model every 10 training epochs
		weights = nca.get_weights()
		if len(weights) >= 2:
			self.log_histogram('Input layer weights',weights[0],step=i)
			self.log_histogram('Output layer weights',weights[1],step=i)
			return
		for idx, w in enumerate(weights):
			self.log_histogram(f"Train/KAN/weight_{idx}", np.squeeze(w), step=i)

	def _log_fast_kan_weight_histograms(self,nca,i):
		for idx, w in enumerate(nca.get_weights()):
			self.log_histogram(f"Train/KAN/weight_{idx}", np.squeeze(w), step=i)

	def _edge_norm_summary(self,edge_norms):
		edge_norms = np.asarray(jax.device_get(edge_norms))
		max_norm = float(np.max(edge_norms)) if edge_norms.size else 0.0
		active_fraction = 0.0
		if max_norm > 0.0:
			active_fraction = float(np.mean(edge_norms > 0.1 * max_norm))
		return {
			"max": max_norm,
			"mean": float(np.mean(edge_norms)) if edge_norms.size else 0.0,
			"median": float(np.median(edge_norms)) if edge_norms.size else 0.0,
			"active_fraction": active_fraction,
		}

	def _plot_edge_norms(self,edge_norms,layer_index):
		import matplotlib.pyplot as plt

		edge_norms = np.asarray(jax.device_get(edge_norms))
		fig, ax = plt.subplots(figsize=(6, 5))
		image = ax.imshow(edge_norms.T, aspect="auto", origin="lower")
		ax.set_xlabel("Input edge")
		ax.set_ylabel("Output edge")
		ax.set_title(f"KAN layer {layer_index} edge norms")
		fig.colorbar(image, ax=ax, label="Edge norm")
		fig.tight_layout()
		return plot_to_image(fig)

	def _plot_top_edge_functions(self,nca,layer_index,k=12,xs=None):
		import matplotlib.pyplot as plt

		layer = _kan_layer(nca, layer_index)
		if xs is None:
			xs = jnp.linspace(layer.grid_min-2, layer.grid_max+2, 200)
		xs_np = np.asarray(jax.device_get(xs))
		edge_values = np.asarray(jax.device_get(nca.evaluate_edge_functions(xs)[layer_index]))
		if hasattr(nca, "get_top_edges"):
			top_edges = nca.get_top_edges(k=k, layer_index=layer_index)
		else:
			edge_norms = np.asarray(jax.device_get(nca.get_edge_norms()[layer_index]))
			flat_order = np.argsort(edge_norms.ravel())[::-1][:k]
			input_indices, output_indices = np.unravel_index(flat_order, edge_norms.shape)
			top_edges = [
				{
					"rank": rank,
					"input_index": int(input_index),
					"output_index": int(output_index),
				}
				for rank, (input_index, output_index) in enumerate(
					zip(input_indices, output_indices),
					start=1,
				)
			]
		fig, ax = plt.subplots(figsize=(8, 4))
		for edge in top_edges:
			input_index = edge["input_index"]
			output_index = edge["output_index"]
			label_input = edge.get("input_name", f"in {input_index}")
			label_output = edge.get("output_name", f"out {output_index}")
			label = (
				f"{edge['rank']}: {_normalise_label(label_input)}"
				f" -> {_normalise_label(label_output)}"
			)
			ax.plot(
				xs_np,
				edge_values[input_index, output_index],
				label=label,
				alpha=0.8,
			)
		ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
		ax.set_xlabel("Input value")
		ax.set_ylabel("Edge contribution")
		ax.set_title(f"KAN layer {layer_index} top {len(top_edges)} edge functions")
		if top_edges:
			ax.legend(fontsize="x-small", ncols=2)
		fig.tight_layout()
		return plot_to_image(fig)

	def _rollout_states_from_log_dict(self,log_dict,max_states=4):
		if "x_latent" not in log_dict:
			return None
		x_latent = log_dict["x_latent"]
		if isinstance(x_latent, (list, tuple)):
			batches = [jnp.asarray(batch) for batch in x_latent if batch is not None]
			if not batches:
				return None
			states = jnp.concatenate(batches, axis=0)
		else:
			states = jnp.asarray(x_latent)
			if states.ndim == 3:
				states = states[None]
		if states.ndim != 4:
			return None
		return states[:max_states]

	def _layer_labels(self,nca,layer_index,input_size,output_size):
		if layer_index == 0 and hasattr(nca, "get_feature_names"):
			input_labels = nca.get_feature_names()
		else:
			input_labels = [f"hidden_{idx}" for idx in range(input_size)]
		if layer_index == 0:
			output_labels = [f"hidden_{idx}" for idx in range(output_size)]
		else:
			output_labels = [f"channel_{idx}" for idx in range(output_size)]
		return input_labels, output_labels

	def _top_edges_from_rollout_score(self,score,relative_score,input_labels,output_labels,k):
		flat_score = score.ravel()
		k = min(k, flat_score.shape[0])
		order = np.argsort(flat_score)[::-1][:k]
		input_indices, output_indices = np.unravel_index(order, score.shape)
		top_edges = []
		for rank, (flat_index, input_index, output_index) in enumerate(
			zip(order,input_indices,output_indices),
			start=1,
		):
			top_edges.append(
				{
					"rank": rank,
					"input_index": int(input_index),
					"output_index": int(output_index),
					"score": float(flat_score[flat_index]),
					"relative_score": float(relative_score[input_index, output_index]),
					"input_name": input_labels[input_index],
					"output_name": output_labels[output_index],
				}
			)
		return top_edges

	def _edge_contribution_variance(self,layer,layer_inputs,chunk_size=1024):
		total = layer_inputs.shape[0]
		edge_sum = None
		edge_sum_sq = None
		for start in range(0, total, chunk_size):
			chunk = jnp.asarray(layer_inputs[start:start + chunk_size])
			edge_values = np.asarray(
				jax.device_get(layer.edge_contributions_from_inputs(chunk))
			)
			chunk_sum = np.sum(edge_values, axis=-1)
			chunk_sum_sq = np.sum(edge_values**2, axis=-1)
			if edge_sum is None:
				edge_sum = chunk_sum
				edge_sum_sq = chunk_sum_sq
			else:
				edge_sum = edge_sum + chunk_sum
				edge_sum_sq = edge_sum_sq + chunk_sum_sq
		edge_mean = edge_sum / max(total, 1)
		edge_var = edge_sum_sq / max(total, 1) - edge_mean**2
		return np.maximum(edge_var, 0.0)

	def _collect_fast_kan_rollout_stats(self,nca,log_dict,max_samples=8192,max_states=4,k=12):
		if not hasattr(nca, "get_kan_layer_inputs_outputs"):
			return None
		states = self._rollout_states_from_log_dict(log_dict,max_states=max_states)
		if states is None:
			return None
		per_state_io = [nca.get_kan_layer_inputs_outputs(state) for state in states]
		layer_stats = []
		for layer_index, layer in enumerate(nca.layers):
			layer_inputs = []
			layer_outputs = []
			for state_io in per_state_io:
				layer_input, layer_output = state_io[layer_index]
				layer_inputs.append(_flatten_channel_first_samples(layer_input))
				layer_outputs.append(_flatten_channel_first_samples(layer_output))
			layer_inputs = _subsample_rows(np.concatenate(layer_inputs, axis=0), max_samples)
			layer_outputs = _subsample_rows(np.concatenate(layer_outputs, axis=0), max_samples)
			kan_layer = _kan_layer(nca, layer_index)
			spline_inputs = np.asarray(
				jax.device_get(
					kan_layer.spline_inputs_from_inputs(jnp.asarray(layer_inputs))
				)
			)
			edge_var = self._edge_contribution_variance(
				kan_layer,
				layer_inputs,
			)
			edge_std = np.sqrt(edge_var)
			input_std = np.std(layer_inputs, axis=0)
			spline_input_std = np.std(spline_inputs, axis=0)
			output_std = np.std(layer_outputs, axis=0)
			output_var = np.var(layer_outputs, axis=0)
			relative_score = edge_var / (output_var[None, :] + 1e-8)
			input_labels, output_labels = self._layer_labels(
				nca,
				layer_index,
				layer_inputs.shape[1],
				layer_outputs.shape[1],
			)
			layer_stats.append(
				{
					"layer_index": layer_index,
					"input_samples": layer_inputs,
					"spline_input_samples": spline_inputs,
					"output_samples": layer_outputs,
					"edge_var": edge_var,
					"edge_std": edge_std,
					"relative_score": relative_score,
					"input_std": input_std,
					"spline_input_std": spline_input_std,
					"output_std": output_std,
					"input_labels": input_labels,
					"output_labels": output_labels,
					"top_edges": self._top_edges_from_rollout_score(
						edge_var,
						relative_score,
						input_labels,
						output_labels,
						k,
					),
				}
			)
		return layer_stats

	def _plot_rollout_edge_variance(self,stats):
		import matplotlib.pyplot as plt

		edge_var = stats["edge_var"]
		layer_index = stats["layer_index"]
		fig, ax = plt.subplots(figsize=(6, 5))
		image = ax.imshow(edge_var.T, aspect="auto", origin="lower")
		ax.set_xlabel("Input feature")
		ax.set_ylabel("Output feature")
		ax.set_title(f"KAN layer {layer_index} rollout edge variance")
		fig.colorbar(image, ax=ax, label="Var(edge contribution)")
		fig.tight_layout()
		return plot_to_image(fig)

	def _plot_rollout_sorted_feature_std(self,stats):
		import matplotlib.pyplot as plt

		layer_index = stats["layer_index"]
		input_std = np.sort(stats["input_std"])[::-1]
		spline_input_std = np.sort(stats["spline_input_std"])[::-1]
		output_std = np.sort(stats["output_std"])[::-1]
		fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=False)
		axes[0].plot(input_std)
		axes[0].set_ylabel("raw in std")
		axes[0].set_title(f"KAN layer {layer_index} sorted feature std")
		axes[1].plot(spline_input_std)
		axes[1].set_ylabel("spline in std")
		axes[2].plot(output_std)
		axes[2].set_ylabel("out std")
		axes[2].set_xlabel("Feature rank")
		for ax in axes:
			ax.set_yscale("symlog", linthresh=1e-4)
			ax.grid(alpha=0.2)
		fig.tight_layout()
		return plot_to_image(fig)

	def _plot_rollout_pre_post_layernorm_histograms(self,stats):
		import matplotlib.pyplot as plt

		layer_index = stats["layer_index"]
		raw_inputs = stats["input_samples"].ravel()
		spline_inputs = stats["spline_input_samples"].ravel()
		outputs = stats["output_samples"].ravel()
		fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=False)
		axes[0].hist(raw_inputs, bins=80, color="tab:blue", alpha=0.75)
		axes[0].set_title(f"KAN layer {layer_index} raw input distribution")
		axes[1].hist(spline_inputs, bins=80, color="tab:orange", alpha=0.75)
		axes[1].set_title("Spline/RBF input distribution after layernorm")
		axes[2].hist(outputs, bins=80, color="tab:green", alpha=0.75)
		axes[2].set_title("Layer output distribution")
		for ax in axes:
			ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
			ax.set_ylabel("count")
		axes[2].set_xlabel("value")
		fig.tight_layout()
		return plot_to_image(fig)

	def _plot_rollout_feature_activity(self,stats,max_features=64):
		import matplotlib.pyplot as plt

		layer_index = stats["layer_index"]
		input_std = stats["input_std"][:max_features]
		output_std = stats["output_std"][:max_features]
		fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=False)
		axes[0].bar(np.arange(input_std.shape[0]), input_std)
		axes[0].set_ylabel("Input std")
		axes[0].set_title(f"KAN layer {layer_index} rollout feature activity")
		axes[1].bar(np.arange(output_std.shape[0]), output_std)
		axes[1].set_ylabel("Output std")
		axes[1].set_xlabel("Feature index")
		fig.tight_layout()
		return plot_to_image(fig)

	def _plot_rollout_top_edge_table(self,stats):
		import matplotlib.pyplot as plt

		top_edges = stats["top_edges"]
		rows = [
			[
				edge["rank"],
				_normalise_label(edge["input_name"], 24),
				_normalise_label(edge["output_name"], 18),
				f"{edge['score']:.2e}",
				f"{edge['relative_score']:.2e}",
			]
			for edge in top_edges
		]
		fig, ax = plt.subplots(figsize=(9, max(2.5, 0.35 * max(len(rows), 1) + 1)))
		ax.axis("off")
		table = ax.table(
			cellText=rows,
			colLabels=["rank", "input", "output", "var", "rel var"],
			loc="center",
			cellLoc="left",
		)
		table.auto_set_font_size(False)
		table.set_fontsize(8)
		table.scale(1, 1.2)
		ax.set_title(f"KAN layer {stats['layer_index']} top rollout edges")
		fig.tight_layout()
		return plot_to_image(fig)

	def _plot_top_rollout_edge_functions(self,nca,stats,xs=None):
		import matplotlib.pyplot as plt

		layer_index = stats["layer_index"]
		layer = _kan_layer(nca, layer_index)
		top_edges = stats["top_edges"]
		if xs is None:
			xs = jnp.linspace(layer.grid_min-2, layer.grid_max+2, 200)
		xs_np = np.asarray(jax.device_get(xs))
		edge_values = np.asarray(jax.device_get(nca.evaluate_edge_functions(xs)[layer_index]))
		n_edges = len(top_edges)
		n_cols = min(3, max(n_edges, 1))
		n_rows = max(1, math.ceil(max(n_edges, 1) / n_cols))
		fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.8 * n_rows))
		axes = np.atleast_1d(axes).ravel()
		for ax_index, ax in enumerate(axes):
			if ax_index >= n_edges:
				ax.axis("off")
				continue
			edge = top_edges[ax_index]
			input_index = edge["input_index"]
			output_index = edge["output_index"]
			visited = stats["input_samples"][:, input_index]
			lo, hi = np.percentile(visited, [5, 95])
			ax.axvspan(lo, hi, color="tab:blue", alpha=0.12)
			ax.plot(
				xs_np,
				edge_values[input_index, output_index],
				color="tab:blue",
				linewidth=1.5,
			)
			ax_hist = ax.twinx()
			ax_hist.hist(visited, bins=30, density=True, color="grey", alpha=0.22)
			ax_hist.set_yticks([])
			ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
			ax.set_title(
				f"{edge['rank']}: {_normalise_label(edge['input_name'], 18)}"
				f" -> {_normalise_label(edge['output_name'], 14)}\n"
				f"var={edge['score']:.2e}, rel={edge['relative_score']:.2e}",
				fontsize=8,
			)
			ax.set_xlabel("Input value")
			ax.set_ylabel("Edge contribution")
		fig.suptitle(f"KAN layer {layer_index} top rollout-varying edge functions")
		fig.tight_layout()
		return plot_to_image(fig)

	def log_fast_kan_rollout_diagnostics(self,nca,log_dict,i,k=12,max_samples=8192):
		stats_by_layer = self._collect_fast_kan_rollout_stats(
			nca,
			log_dict,
			max_samples=max_samples,
			k=k,
		)
		if stats_by_layer is None:
			return False
		for stats in stats_by_layer:
			layer_index = stats["layer_index"]
			edge_var = stats["edge_var"]
			self.log_scalar(
				f"Train/KAN/layer_{layer_index}/rollout_edge_var_max",
				float(np.max(edge_var)) if edge_var.size else 0.0,
				step=i,
			)
			self.log_scalar(
				f"Train/KAN/layer_{layer_index}/rollout_edge_var_mean",
				float(np.mean(edge_var)) if edge_var.size else 0.0,
				step=i,
			)
			self.log_scalar(
				f"Train/KAN/layer_{layer_index}/rollout_input_std_mean",
				float(np.mean(stats["input_std"])) if stats["input_std"].size else 0.0,
				step=i,
			)
			self.log_scalar(
				f"Train/KAN/layer_{layer_index}/rollout_output_std_mean",
				float(np.mean(stats["output_std"])) if stats["output_std"].size else 0.0,
				step=i,
			)
			self.log_scalar(
				f"Train/KAN/layer_{layer_index}/rollout_spline_input_std_mean",
				float(np.mean(stats["spline_input_std"])) if stats["spline_input_std"].size else 0.0,
				step=i,
			)
			for eps in [1e-3, 1e-2, 1e-1]:
				eps_tag = f"{eps:g}".replace(".", "p").replace("-", "m")
				self.log_scalar(
					f"Train/KAN/layer_{layer_index}/frac_abs_raw_input_lt_{eps_tag}",
					_fraction_abs_below(stats["input_samples"], eps),
					step=i,
				)
				self.log_scalar(
					f"Train/KAN/layer_{layer_index}/frac_abs_spline_input_lt_{eps_tag}",
					_fraction_abs_below(stats["spline_input_samples"], eps),
					step=i,
				)
				self.log_scalar(
					f"Train/KAN/layer_{layer_index}/frac_abs_output_lt_{eps_tag}",
					_fraction_abs_below(stats["output_samples"], eps),
					step=i,
				)
			self.log_image(
				f"Train/KAN/layer_{layer_index}_rollout_edge_variance",
				self._plot_rollout_edge_variance(stats),
				step=i,
			)
			self.log_image(
				f"Train/KAN/layer_{layer_index}_rollout_sorted_feature_std",
				self._plot_rollout_sorted_feature_std(stats),
				step=i,
			)
			self.log_image(
				f"Train/KAN/layer_{layer_index}_rollout_pre_post_layernorm_histograms",
				self._plot_rollout_pre_post_layernorm_histograms(stats),
				step=i,
			)
			self.log_image(
				f"Train/KAN/layer_{layer_index}_rollout_feature_activity",
				self._plot_rollout_feature_activity(stats),
				step=i,
			)
			self.log_image(
				f"Train/KAN/layer_{layer_index}_rollout_top_edge_table",
				self._plot_rollout_top_edge_table(stats),
				step=i,
			)
			self.log_image(
				f"Train/KAN/layer_{layer_index}_rollout_top_edge_functions",
				self._plot_top_rollout_edge_functions(nca,stats),
				step=i,
			)
		return True

	def log_fast_kan_diagnostics(self,nca,i,k=12,log_static_top_edges=True):
		self._log_fast_kan_weight_histograms(nca,i)
		for layer_index, edge_norms in enumerate(nca.get_edge_norms()):
			summary = self._edge_norm_summary(edge_norms)
			for name, value in summary.items():
				self.log_scalar(
					f"Train/KAN/layer_{layer_index}/edge_norm_{name}",
					value,
					step=i,
				)
			self.log_scalar(
				f"Train/KAN/layer_{layer_index}/rbf_width",
				_kan_layer_width(_kan_layer(nca, layer_index)),
				step=i,
			)
			self.log_image(
				f"Train/KAN/layer_{layer_index}_edge_norms",
				self._plot_edge_norms(edge_norms,layer_index),
				step=i,
			)
			if log_static_top_edges:
				self.log_image(
					f"Train/KAN/layer_{layer_index}_top_edge_functions_by_norm",
					self._plot_top_edge_functions(nca,layer_index,k=k),
					step=i,
				)

	def log_model_parameters(self,nca,i):
		if uses_fast_kan_diagnostics(nca):
			self.log_fast_kan_diagnostics(nca,i)
		else:
			self._log_legacy_kan_parameters(nca,i)

	def tb_training_loop_log_sequence(self,log_dict,i,model,write_images=True,LOG_EVERY=10):
		for name in log_dict.keys():
			if name not in ["x_latent", "x_processed"]:
				if name.startswith("pool/"):
					self.log_scalar(f"StatePool/{name.removeprefix('pool/')}",log_dict[name],step=i)
				elif name == "learning_rate":
					self.log_scalar("Training/learning_rate", log_dict[name], step=i)
				else:
					self.log_scalar(f"Train/{name}",log_dict[name],step=i)
		if i%LOG_EVERY==0 and i>0:
			if uses_fast_kan_diagnostics(model):
				has_rollout = "x_latent" in log_dict and hasattr(
					model,
					"get_kan_layer_inputs_outputs",
				)
				self.log_fast_kan_diagnostics(
					model,
					i,
					log_static_top_edges=not has_rollout,
				)
				self.log_fast_kan_rollout_diagnostics(model,log_dict,i)
			else:
				self.log_model_parameters(model,i)
			self.log_channel_time_diagnostics(log_dict,i)
			if write_images:
				self.log_model_outputs(log_dict,i)


class kaNCA_Train_pde_log(kaNCA_Train_log):
	def log_model_outputs(self, x, i):
		pass # Saving the trajectory outputs during training generates far too many images

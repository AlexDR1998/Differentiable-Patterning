# Tutorials

marimo notebooks that run on a laptop CPU with the files in this repository.
Open them from the repository root, after setting up (see the main README):

```bash
marimo edit demo/what_is_an_nca.py
```

Work through them in this order:

| # | Notebook | What you learn | Time |
| --- | --- | --- | --- |
| 1 | `what_is_an_nca.py` | What an NCA is: perception kernels, the update network, hidden channels, firing, rollouts and boundaries. Built up by hand. | 15 min |
| 2 | `nca_training_config_walkthrough.py` | A whole experiment in one notebook: data, model, config, training a small emoji NCA, saving and reloading it as a bundle. | 30 min (training takes a few minutes) |
| 3 | `nca_objectives_and_pool_dynamics.py` | What training optimises: loss terms, regularisers, the data augmenter and the training pool. | 30 min |
| 4 | `pde_patterns_demo.py` | The pattern-forming PDEs that can be used as training data. | 10 min |
| 5 | `bundles_and_inference.py` | Finding trained models in the model store, saving a selection, loading a bundle, running it and knocking a channel out. Uses the model from tutorial 2, or the shared store. | 20 min |

After these:
- read [docs/concepts.md](../docs/concepts.md);
- then follow [docs/workflow_micropatterns.md](../docs/workflow_micropatterns.md)
  for a real experiment on the cluster.

`demo_data/Emojis/` holds the five example emojis the tutorials use.

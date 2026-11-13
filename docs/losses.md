# Losses and regularisers

The training loss is a weighted sum of **loss terms**, which compare the NCA's
observed channels with the data, plus **regularisers**, which penalise
unwanted behaviour without using the data:

```yaml
loss:
  terms:
    - type: l2            # a name from the tables below
      weight: 1.0
    - type: radial_profile
      weight: 0.5
      radial_bins: 16     # options depend on the type (config class below)
  regularisers:
    intermediate_state: 1.0
    boundary: 5.0
```

Every term also accepts:
- `channels`: only compare these channels;
- `experiment_groups`: only these staining panels;
- `schedule`: change the weight during training (see
  [configuration.md](configuration.md#loss-weight-schedules)).

Where things live:
- name → function: `Common/trainer/loss_table.py:LOSSES`;
- name → options class: `Common/trainer/config.py:LOSS_TERM_CONFIGS`.

Losses take predictions and targets `[N, C, H, W]` and return one value per
sample `[N]`.

## Choosing a loss

- **Pixels line up** (emoji, PDEs, snowmelt): `l2` is the place to start.
  `spectral` also compares Fourier transforms, which helps with fine
  patterns.
- **Pixels don't line up, but the pattern should look the same** (textures,
  different colonies): use `vgg`, `ott`, the sliced Wasserstein losses, or
  `radial_profile` for ring patterns.
- **Micropatterns:** each snapshot is a different fixed and stained colony,
  so a predicted colony has no one true partner. `multi_target` matches
  predicted and target colonies before comparing them, and combines texture,
  radial profile, channel mean, correlation and l2 parts. The paper models
  use it. The `*_grouped` losses keep channels from the same staining panel
  together (see [data.md](data.md)).

## Loss terms

### Pixel-by-pixel and spectral (`PointwiseLossConfig`, no options)

| Name | What it compares |
| --- | --- |
| `l2` | Mean squared error. |
| `l1` | Mean absolute error. |
| `euclidean` | Root mean squared error. |
| `cosine` | Negative cosine similarity of the whole arrays. |
| `spectral` | L2 distance between Fourier transforms. |
| `spectral_no_phase` | L2 distance between Fourier amplitudes (ignores where features are). |
| `spectral_phase` | L2 distance between Fourier phases only. |
| `average_amplitude` | Squared difference of each channel's mean intensity. |
| `bhattacharyya`, `hellinger` | Distances between channel images treated as distributions (normalised to unit L2 norm). |
| `kl_divergence` | KL(x‖y) between channel images normalised to sum to 1. |

### Distribution losses (`WassersteinLossConfig`: `samples`)

| Name | What it compares |
| --- | --- |
| `sliced_wasserstein_channel` | The distribution of pixel values across channels, through `samples` random projections. Ignores where pixels are. |
| `sliced_wasserstein_spatial` | Sliced Wasserstein distance with random spatial projections. |
| `sliced_wasserstein_rotational` | 1D profiles of the images rotated by random angles. |
| `sliced_wasserstein_full` | Mean squared difference of random projections of the whole image. **Note:** despite the name it does not sort, so it is not a Wasserstein distance. |
| `spectral_wasserstein_full` | As `sliced_wasserstein_full`, on the Fourier transform (same note). |

### Summary statistics (`SummaryLossConfig`: `radial_bins`, `channel_importance`)

| Name | What it compares |
| --- | --- |
| `radial_profile` | Mean of each channel in rings around the centre, inside the colony mask. Good for ring-shaped micropattern fates. |
| `channel_correlation` | Correlations between channels within each image. |
| `radial_profile_grouped`, `channel_correlation_grouped` | The same on the micropattern panel layout. |

### Texture losses

| Name | Options class | What it compares |
| --- | --- | --- |
| `vgg` | `VggLossConfig` | Features of a pretrained VGG image network (LPIPS), on blocks of 3 channels. Compares texture and shape rather than exact pixels. `metric` chooses how features are compared: `l2`, or the sliced OT versions `otch` and `otsp`. |
| `ott` | `OttLossConfig` | Optimal transport between sets of `S` random `K×K` patches, at `D` scales, per channel. |
| `ott_chstack` | `OttLossConfig` | As `ott`, with patches from the same place in every channel stacked together. |
| `vgg_grouped`, `vgg_grouped_and_l2`, `ott_grouped`, `ott_grouped_and_l2`, `l2_grouped` | | The same on the micropattern panel layout (`Common/trainer/loss_micropattern.py`). `channel_importance` (12 weights) changes how much each measured channel counts. |

VGG and OT losses are slow and use a lot of GPU memory. Unit tests don't
check their numbers; the smoke tests do.

### Multi-target (`MultiTargetLossConfig`)

`multi_target` (`NCA/trainer/loss_multi_target.py`) is for small batches of
snapshots where each target colony is a different sample. For each time and
staining panel it:
1. computes the cost of every predicted colony against every target colony;
2. matches them one to one (`assignment: hard`, or `softmin` with temperature `assignment_tau`);
3. returns the matched cost.

The cost is a weighted sum of parts, set by
`multi_target_weights`:
- `l2` (default 0);
- `texture` (VGG/LPIPS);
- `channel_mean`;
- `radial`;
- `correlation`.

Each part can have its own schedule (`multi_target_schedules`).
`Experiments/micropatterns/conf/experiments/nca_nodal_ko_projections_finetune.yaml`
is a full example.

## Regularisers

Set a weight above 0 under `loss.regularisers` to switch one on. The code is
in `NCA/trainer/NCA_regulariser.py`, and the name table is in
`NCA/trainer/preparation.py:_regularisers`.

| Name | What it penalises |
| --- | --- |
| `intermediate_state` | State values outside [0, 1]. Usually on (1.0). |
| `boundary` | Non-zero channels outside the colony mask. Used for micropatterns. |
| `hidden_state_size` | Large hidden channel values. |
| `localised_hidden` | Hidden channel activity where all observed channels are low. |
| `contiguous_growth` | Observed channels growing away from cells that are already high. |
| `update_sensitivity` | Updates that change a lot when the input gets a little noise. |
| `perturbation_conservation` | Updates that don't carry small input perturbations through unchanged. |
| `firing_energy` | The mean learned firing probability (aNCA/gaNCA only): a cost for updating. |
| `wire_cost` | The amount of grown wiring (sNCA/gsNCA only). |

## Adding a loss

1. Write the function `f(x, y, key, where, ...) -> [N]` in
   `Common/trainer/loss.py`. If it depends on the micropattern channel
   layout, put it in `loss_micropattern.py` instead.
2. Add it to `LOSSES` in `Common/trainer/loss_table.py`.
3. Add its name and options class to `LOSS_TERM_CONFIGS` in
   `Common/trainer/config.py`. An import-time check makes sure the two tables
   list the same names.
4. Add a test in `tests/unit/test_loss_table.py`.

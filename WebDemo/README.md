# WebDemo

Static WebGL demo and lightweight analysis page for exported NCA rollouts.

Export the initial demo assets with:

```bash
python WebDemo/export_model.py \
  --model-path demo/models/test_grow_crab.eqx \
  --model-id test_grow_crab \
  --family NCA \
  --channels 20 \
  --kernels ID GRAD LAP \
  --activation relu \
  --padding CIRCULAR \
  --fire-rate 0.5 \
  --grid-size 96 96 \
  --reference-steps 8
```

Serve the static demo from the repo root with:

```bash
python -m http.server 8000 --directory WebDemo/public
```

Then open `http://localhost:8000`. The demo loads the first entry in
`WebDemo/public/models/index.json` by default and shows the live rollout,
model metadata, and reference-rollout channel summaries.

To open a specific exported model directly, pass its model id in the query string:

```text
http://localhost:8000/?model=your_model_id
```

The page also has a model selector populated from `WebDemo/public/models/index.json`.
Export commands refresh this file automatically. If you manually add or remove model
folders, refresh it with:

```bash
python WebDemo/update_model_index.py
```

Run the marimo blog-style page with:

```bash
marimo run WebDemo/marimo_app.py
```

Export it as a static WebAssembly HTML page with:

```bash
marimo export html-wasm WebDemo/marimo_app.py -o WebDemo/site --mode run
```

Emoji models loaded in `Experiments/emoji/thesis_chapter_1_figures.py` can be
exported directly from the notebook:

```python
nca, H = models_reg[0]
x0 = make_emoji_web_initial_state(data, H["channels"])
export_nca_web_assets(nca, "emoji_good_model", x0=x0)
```

The MVP runtime supports only plain `NCA`, ReLU, `CIRCULAR` or `REPLICATE`
padding, float32 assets, and the anisotropic `ID GRAD LAP` perception path.

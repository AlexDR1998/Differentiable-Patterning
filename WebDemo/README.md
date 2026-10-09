# WebDemo

WebGL viewer and marimo page for exported NCA rollouts.

Export a model with:

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

Serve it from the repo root:

```bash
python -m http.server 8000 --directory WebDemo/public
```

Then open `http://localhost:8000`. It loads the first model in
`WebDemo/public/models/index.json`, or a specific one with:

```text
http://localhost:8000/?model=your_model_id
```

The model selector reads `index.json`, which the exporter updates. If you add
or remove model folders by hand, rebuild it with:

```bash
python WebDemo/update_model_index.py
```

Run the marimo page, or export it as static WebAssembly HTML:

```bash
marimo run WebDemo/marimo_app.py
marimo export html-wasm WebDemo/marimo_app.py -o WebDemo/site --mode run
```

Emoji models can also be exported from
`Experiments/emoji/thesis_chapter_1_figures.py`:

```python
nca, H = models_reg[0]
x0 = make_emoji_web_initial_state(data, H["channels"])
export_nca_web_assets(nca, "emoji_good_model", x0=x0)
```

The viewer only supports plain `NCA` with ReLU, `CIRCULAR` or `REPLICATE`
padding, float32 weights and `ID GRAD LAP` kernels.

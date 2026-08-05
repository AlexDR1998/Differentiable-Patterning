# /// script
# dependencies = [
#   "marimo",
#   "numpy",
# ]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import base64
    import html
    import json
    from pathlib import Path

    import numpy as np

    return Path, base64, html, json, np


@app.cell
def _(Path, json, mo, np):
    try:
        notebook_root = Path(mo.notebook_location())
    except Exception:
        notebook_root = Path(__file__).parent

    public_dir = notebook_root / "public"
    model_dir = public_dir / "models" / "test_grow_crab"
    manifest_path = model_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Expected exported WebGL assets at "
            f"{model_dir}. Run WebDemo/export_model.py first."
        )

    manifest = json.loads(manifest_path.read_text())

    def load_float32_asset(asset_name, shape):
        data = np.frombuffer((model_dir / asset_name).read_bytes(), dtype=np.float32)
        return data.reshape(shape)

    x0 = load_float32_asset(
        manifest["initialState"]["path"],
        tuple(manifest["initialState"]["shape"]),
    )
    reference = load_float32_asset(
        manifest["validation"]["reference"]["path"],
        tuple(manifest["validation"]["reference"]["shape"]),
    )
    return manifest, model_dir, public_dir, reference, x0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interactive Neural Cellular Automata

    This page combines a live WebGL rollout with lightweight Python analysis
    cells. The simulation below runs in the browser from static model assets;
    the Python cells read the same exported arrays for small summaries.
    """)
    return


@app.cell(hide_code=True)
def _(base64, html, json, manifest, mo, model_dir, public_dir):
    runtime_source = (public_dir / "js" / "nca_runtime.js").read_text()
    weights_b64 = base64.b64encode(
        (model_dir / manifest["weights"]["path"]).read_bytes()
    ).decode("ascii")
    x0_b64 = base64.b64encode(
        (model_dir / manifest["initialState"]["path"]).read_bytes()
    ).decode("ascii")
    reference_b64 = base64.b64encode(
        (model_dir / manifest["validation"]["reference"]["path"]).read_bytes()
    ).decode("ascii")

    srcdoc = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
          :root {{
            color-scheme: dark;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system,
              BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #111416;
            color: #e8ecef;
          }}
          body {{
            margin: 0;
            min-height: 100vh;
            display: grid;
            place-items: center;
          }}
          main {{
            width: min(900px, calc(100vw - 24px));
            display: grid;
            gap: 14px;
          }}
          canvas {{
            width: min(720px, calc(100vw - 24px));
            aspect-ratio: 1;
            image-rendering: pixelated;
            background: #050607;
            border: 1px solid #31383e;
            touch-action: none;
          }}
          .toolbar {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
          }}
          button, input {{
            font: inherit;
          }}
          button {{
            min-height: 34px;
            border: 1px solid #3a4249;
            background: #1b2024;
            color: #f5f7f8;
            border-radius: 6px;
            padding: 0 12px;
            cursor: pointer;
          }}
          button[aria-pressed="true"] {{
            background: #315f75;
            border-color: #5b93aa;
          }}
          label {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            min-height: 34px;
          }}
          .status {{
            min-height: 20px;
            color: #aab4bb;
            font-size: 14px;
          }}
          .error {{
            color: #ffb1a6;
          }}
        </style>
      </head>
      <body>
        <main>
          <canvas id="nca-canvas" width="768" height="768"></canvas>
          <div class="toolbar">
            <button id="reset" type="button">Reset</button>
            <button id="play-pause" type="button">Play</button>
            <button id="step-once" type="button">Step</button>
            <button id="display-rgb" type="button" aria-pressed="true">RGB</button>
            <button id="display-hidden-grid" type="button" aria-pressed="false">Hidden Grid</button>
            <button id="brush-zero" type="button" aria-pressed="true">Zero</button>
            <button id="brush-random" type="button" aria-pressed="false">Random</button>
            <button id="brush-target-all" type="button" aria-pressed="true">All</button>
            <button id="brush-target-rgb" type="button" aria-pressed="false">RGB Only</button>
            <label>
              Speed
              <input id="playback-speed" type="range" min="-4" max="2" value="0">
              <span id="playback-speed-value">1x</span>
            </label>
            <label>
              Radius
              <input id="brush-radius" type="range" min="1" max="16" value="6">
            </label>
            <label>
              Hidden Saturation
              <input id="hidden-saturation" type="range" min="0.25" max="8" step="0.25" value="1">
              <span id="hidden-saturation-value">1x</span>
            </label>
            <button id="validate" type="button">Validate</button>
          </div>
          <div id="status" class="status">Loading model...</div>
        </main>
        <script type="module">
          const runtimeSource = {json.dumps(runtime_source)};
          const manifest = {json.dumps(manifest)};
          const weightsB64 = {json.dumps(weights_b64)};
          const x0B64 = {json.dumps(x0_b64)};
          const referenceB64 = {json.dumps(reference_b64)};

          const runtimeUrl = URL.createObjectURL(
            new Blob([runtimeSource], {{ type: "text/javascript" }})
          );
          const {{ NCAWebGLRuntime }} = await import(runtimeUrl);

          const canvas = document.getElementById("nca-canvas");
          const status = document.getElementById("status");
          const resetButton = document.getElementById("reset");
          const playPauseButton = document.getElementById("play-pause");
          const stepOnceButton = document.getElementById("step-once");
          const displayRgbButton = document.getElementById("display-rgb");
          const displayHiddenGridButton = document.getElementById("display-hidden-grid");
          const zeroButton = document.getElementById("brush-zero");
          const randomButton = document.getElementById("brush-random");
          const brushTargetAllButton = document.getElementById("brush-target-all");
          const brushTargetRgbButton = document.getElementById("brush-target-rgb");
          const playbackSpeedInput = document.getElementById("playback-speed");
          const playbackSpeedValue = document.getElementById("playback-speed-value");
          const radiusInput = document.getElementById("brush-radius");
          const hiddenSaturationInput = document.getElementById("hidden-saturation");
          const hiddenSaturationValue = document.getElementById("hidden-saturation-value");
          const validateButton = document.getElementById("validate");

          let runtime = null;
          let paused = true;
          let brushMode = "zero";
          let brushTarget = "all";
          let displayMode = "rgb";
          let pointerDown = false;
          let detailMessage = "";
          let stepAccumulator = 0.0;
          playPauseButton.textContent = "Play";

          function setStatus(message, isError = false) {{
            status.textContent = message;
            status.classList.toggle("error", isError);
          }}

          function decodeFloat32(base64Value) {{
            const binary = atob(base64Value);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i += 1) {{
              bytes[i] = binary.charCodeAt(i);
            }}
            return new Float32Array(bytes.buffer);
          }}

          function canvasToGrid(event) {{
            const rect = canvas.getBoundingClientRect();
            const x = (event.clientX - rect.left) / rect.width;
            const y = (event.clientY - rect.top) / rect.height;
            return [
              Math.floor(x * runtime.width),
              Math.floor(y * runtime.height),
            ];
          }}

          function paint(event) {{
            if (!runtime) return;
            const [x, y] = canvasToGrid(event);
            runtime.paint(x, y, Number(radiusInput.value), brushMode, brushTarget);
          }}

          function setBrushMode(mode) {{
            brushMode = mode;
            zeroButton.setAttribute("aria-pressed", String(mode === "zero"));
            randomButton.setAttribute("aria-pressed", String(mode === "random"));
          }}

          function setBrushTarget(target) {{
            brushTarget = target;
            brushTargetAllButton.setAttribute("aria-pressed", String(target === "all"));
            brushTargetRgbButton.setAttribute("aria-pressed", String(target === "rgb"));
          }}

          function setDisplayMode(mode) {{
            displayMode = mode;
            displayRgbButton.setAttribute("aria-pressed", String(mode === "rgb"));
            displayHiddenGridButton.setAttribute("aria-pressed", String(mode === "hidden-grid"));
          }}

          function drawOptions() {{
            return {{
              mode: displayMode === "hidden-grid" ? "hidden-grid" : "rgb",
              gridChannelOffset: 3,
              gridSaturation: Number(hiddenSaturationInput.value),
            }};
          }}

          function playbackSpeed() {{
            return 2 ** Number(playbackSpeedInput.value);
          }}

          function playbackSpeedLabel() {{
            const speed = playbackSpeed();
            return speed < 1 ? `1/${{Math.round(1 / speed)}}x` : `${{speed}}x`;
          }}

          function render() {{
            if (runtime && !paused) {{
              stepAccumulator += playbackSpeed();
              const stepsThisFrame = Math.floor(stepAccumulator);
              stepAccumulator -= stepsThisFrame;
              for (let i = 0; i < stepsThisFrame; i += 1) {{
                runtime.step();
              }}
            }}
            if (runtime) {{
              runtime.draw(drawOptions());
              const state = paused ? "Paused" : "Playing";
              setStatus(
                `${{state}} | Step ${{runtime.stepCount}}${{detailMessage ? ` | ${{detailMessage}}` : ""}}`
              );
            }}
            requestAnimationFrame(render);
          }}

          try {{
            runtime = new NCAWebGLRuntime(
              canvas,
              manifest,
              decodeFloat32(weightsB64),
              decodeFloat32(x0B64)
            );
            runtime.reset();
            requestAnimationFrame(render);
          }} catch (error) {{
            console.error(error);
            setStatus(error.message, true);
          }}

          resetButton.addEventListener("click", () => {{
            detailMessage = "";
            stepAccumulator = 0.0;
            runtime?.reset();
          }});
          playPauseButton.addEventListener("click", () => {{
            paused = !paused;
            stepAccumulator = 0.0;
            playPauseButton.textContent = paused ? "Play" : "Pause";
          }});
          stepOnceButton.addEventListener("click", () => {{
            if (!runtime) return;
            detailMessage = "";
            stepAccumulator = 0.0;
            runtime.step();
            runtime.draw(drawOptions());
          }});
          playbackSpeedInput.addEventListener("input", () => {{
            playbackSpeedValue.textContent = playbackSpeedLabel();
          }});
          displayRgbButton.addEventListener("click", () => setDisplayMode("rgb"));
          displayHiddenGridButton.addEventListener("click", () => setDisplayMode("hidden-grid"));
          hiddenSaturationInput.addEventListener("input", () => {{
            hiddenSaturationValue.textContent = `${{hiddenSaturationInput.value}}x`;
          }});
          zeroButton.addEventListener("click", () => setBrushMode("zero"));
          randomButton.addEventListener("click", () => setBrushMode("random"));
          brushTargetAllButton.addEventListener("click", () => setBrushTarget("all"));
          brushTargetRgbButton.addEventListener("click", () => setBrushTarget("rgb"));
          validateButton.addEventListener("click", () => {{
            if (!runtime) return;
            const result = runtime.validate(decodeFloat32(referenceB64));
            detailMessage =
              `Validation max abs error: ${{result.maxAbsError.toExponential(3)}}`;
            console.log(result);
          }});
          canvas.addEventListener("pointerdown", (event) => {{
            pointerDown = true;
            canvas.setPointerCapture(event.pointerId);
            paint(event);
          }});
          canvas.addEventListener("pointermove", (event) => {{
            if (pointerDown) paint(event);
          }});
          canvas.addEventListener("pointerup", () => {{
            pointerDown = false;
          }});
          canvas.addEventListener("pointercancel", () => {{
            pointerDown = false;
          }});
        </script>
      </body>
    </html>
    """
    mo.Html(
        f"""
        <iframe
          srcdoc="{html.escape(srcdoc, quote=True)}"
          style="
            width: 100%;
            height: 900px;
            border: 1px solid #30363d;
            border-radius: 8px;
            background: #111416;
          "
          loading="lazy"
        ></iframe>
        """
    )
    return


@app.cell(hide_code=True)
def _(manifest, mo):
    mo.md(f"""
    ## Exported model

    - model id: `{manifest["modelId"]}`
    - family: `{manifest["family"]}`
    - channels: `{manifest["channels"]}`
    - kernels: `{", ".join(manifest["kernels"])}`
    - grid: `{manifest["gridSize"][0]} x {manifest["gridSize"][1]}`
    - browser fire rate: `{manifest["fireRate"]}`
    - validation steps: `{manifest["validation"]["referenceSteps"]}`
    """)
    return


@app.cell
def _(manifest, np, reference, x0):
    def channel_summary(state):
        channels = state.shape[0]
        flat = state.reshape(channels, -1)
        return {
            "sum": flat.sum(axis=1),
            "mean": flat.mean(axis=1),
            "max": flat.max(axis=1),
            "min": flat.min(axis=1),
        }

    initial_summary = channel_summary(x0)
    reference_summary = channel_summary(reference)
    delta_sum = reference_summary["sum"] - initial_summary["sum"]
    top_delta_channels = np.argsort(np.abs(delta_sum))[::-1][:8]
    summary_rows = [
        {
            "channel": int(ch),
            "initial_sum": float(initial_summary["sum"][ch]),
            "reference_sum": float(reference_summary["sum"][ch]),
            "delta_sum": float(delta_sum[ch]),
            "reference_max": float(reference_summary["max"][ch]),
        }
        for ch in top_delta_channels
    ]
    visible_channels = manifest["display"]["channels"]
    return delta_sum, summary_rows, top_delta_channels


@app.cell(hide_code=True)
def _(html, mo, summary_rows):
    def fmt(value):
        return f"{value:.4g}" if isinstance(value, float) else str(value)

    table_rows = "\n".join(
        "<tr>"
        + "".join(f"<td>{html.escape(fmt(row[key]))}</td>" for key in row)
        + "</tr>"
        for row in summary_rows
    )
    header = "".join(f"<th>{html.escape(key)}</th>" for key in summary_rows[0])
    mo.Html(
        f"""
        <h2>Largest channel-mass changes after the reference rollout</h2>
        <table style="border-collapse: collapse; width: 100%; font-size: 0.95rem;">
          <thead><tr>{header}</tr></thead>
          <tbody>{table_rows}</tbody>
        </table>
        <style>
          table th, table td {{
            border-bottom: 1px solid #3a4249;
            padding: 0.35rem 0.5rem;
            text-align: right;
          }}
          table th:first-child, table td:first-child {{
            text-align: left;
          }}
        </style>
        """
    )
    return


@app.cell(hide_code=True)
def _(delta_sum, html, mo, top_delta_channels):
    values = [float(delta_sum[ch]) for ch in top_delta_channels]
    max_abs = max(max(abs(v) for v in values), 1e-8)
    width = 680
    row_h = 28
    left = 120
    mid = 380
    scale = 250 / max_abs
    bars = []
    for i, ch in enumerate(top_delta_channels):
        y = 28 + i * row_h
        value = float(delta_sum[ch])
        bar_w = abs(value) * scale
        x = mid if value >= 0 else mid - bar_w
        color = "#5fb3d4" if value >= 0 else "#e0776d"
        bars.append(
            f'<text x="12" y="{y + 15}" fill="#d8dee4">channel {int(ch)}</text>'
            f'<rect x="{x:.2f}" y="{y}" width="{bar_w:.2f}" height="18" '
            f'fill="{color}" rx="3"></rect>'
            f'<text x="{mid + 260}" y="{y + 15}" fill="#aab4bb">'
            f'{html.escape(f"{value:.4g}")}</text>'
        )
    mo.Html(
        f"""
        <svg viewBox="0 0 {width} {len(values) * row_h + 48}" style="width: 100%; max-width: {width}px;">
          <line x1="{mid}" y1="18" x2="{mid}" y2="{len(values) * row_h + 28}" stroke="#6b737c"></line>
          {''.join(bars)}
        </svg>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Notes

    The embedded canvas is intentionally decoupled from these Python cells:
    the WebGL runtime owns the live simulation state, while marimo reads
    exported assets and reference outputs for small analyses. A later custom
    widget can pass live summaries, selected-pixel values, or sampled states
    back into marimo.
    """)
    return


if __name__ == "__main__":
    app.run()

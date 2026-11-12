"""A widget for placing a square zoom box on an image by clicking or dragging.

Wrap it with ``mo.ui.anywidget(BoxPicker(...))`` in marimo. ``x`` and ``y``
are the box centre as fractions of the image width and height (0 at the
left and top), and ``size_um`` its side in µm; ``image_um`` is the image
width in µm. The centre is sent to Python when the mouse is released.
"""

import base64
import io

import anywidget
import numpy as np
import traitlets


def png_data_url(image, max_side=640):
    """Data URL of a greyscale ``[H, W]`` image in [0, 1] (or RGB ``[H, W, 3]``), shrunk to at most ``max_side`` px."""

    import matplotlib.pyplot as plt

    image = np.asarray(image)
    step = max(1, int(np.ceil(max(image.shape[:2]) / max_side)))
    buffer = io.BytesIO()
    plt.imsave(buffer, np.clip(image[::step, ::step], 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0, format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class BoxPicker(anywidget.AnyWidget):
    """Click or drag on ``background`` (a data URL) to move a square box; the slider sets its size."""

    _esm = """
    function render({ model, el }) {
      const canvas = document.createElement("canvas");
      canvas.style.cssText = "cursor: crosshair; touch-action: none; border-radius: 4px; background: black; width: min(480px, 100%);";
      const context = canvas.getContext("2d");
      const bar = document.createElement("div");
      bar.style.cssText = "display: flex; gap: 8px; align-items: center; margin-top: 6px; font: 12px sans-serif;";
      const size = document.createElement("input");
      size.type = "range";
      size.min = 10;
      size.max = 400;
      size.step = 10;
      const sizeLabel = document.createElement("span");
      bar.append("Box", size, sizeLabel);
      el.append(canvas, bar);

      const image = new Image();
      const clamp = (value) => Math.min(Math.max(value, 0), 1);
      function draw() {
        if (!image.complete || !image.width) return;
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        const side = (model.get("size_um") / model.get("image_um")) * canvas.width;
        const x = model.get("x") * canvas.width;
        const y = model.get("y") * canvas.height;
        context.lineWidth = Math.max(1.5, canvas.width / 300);
        context.strokeStyle = "yellow";
        context.strokeRect(x - side / 2, y - side / 2, side, side);
        size.value = model.get("size_um");
        sizeLabel.textContent = `${model.get("size_um")} µm`;
      }
      function load() {
        image.onload = () => {
          canvas.width = image.width;
          canvas.height = image.height;
          draw();
        };
        image.src = model.get("background");
      }

      let dragging = false;
      function moveTo(event) {
        const box = canvas.getBoundingClientRect();
        model.set("x", clamp((event.clientX - box.left) / box.width));
        model.set("y", clamp((event.clientY - box.top) / box.height));
        draw();
      }
      canvas.addEventListener("pointerdown", (event) => {
        dragging = true;
        canvas.setPointerCapture(event.pointerId);
        moveTo(event);
      });
      canvas.addEventListener("pointermove", (event) => { if (dragging) moveTo(event); });
      const release = () => {
        if (!dragging) return;
        dragging = false;
        model.save_changes();
      };
      canvas.addEventListener("pointerup", release);
      canvas.addEventListener("pointercancel", release);
      size.addEventListener("input", () => { model.set("size_um", Number(size.value)); draw(); });
      size.addEventListener("change", () => model.save_changes());

      model.on("change:background", load);
      model.on("change:x", draw);
      model.on("change:y", draw);
      model.on("change:size_um", draw);
      load();
    }
    export default { render };
    """

    background = traitlets.Unicode("").tag(sync=True)
    image_um = traitlets.Float(1.0).tag(sync=True)
    x = traitlets.Float(0.5).tag(sync=True)
    y = traitlets.Float(0.5).tag(sync=True)
    size_um = traitlets.Float(80.0).tag(sync=True)


def rgb_data_url(image):
    """Data URL of an RGB image ``[H, W, 3]`` in [0, 1], at full size."""

    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    plt.imsave(buffer, np.clip(np.asarray(image), 0.0, 1.0), format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class PointPicker(anywidget.AnyWidget):
    """Click on ``background`` (a data URL) to mark points; click a mark to remove it.

    ``points`` are ``[row, column]`` in pixels of the background image. The
    image is drawn ``display`` px wide with square pixels; ``overlay`` (a data
    URL of the same size, optional) can be shown on top with the Overlay box.
    """

    _esm = """
    function render({ model, el }) {
      const canvas = document.createElement("canvas");
      canvas.style.cssText = "cursor: crosshair; image-rendering: pixelated; border-radius: 4px; background: black;";
      const context = canvas.getContext("2d");
      const bar = document.createElement("div");
      bar.style.cssText = "display: flex; gap: 10px; align-items: center; margin-top: 6px; font: 12px sans-serif;";
      const count = document.createElement("span");
      const overlayBox = document.createElement("input");
      overlayBox.type = "checkbox";
      const overlayLabel = document.createElement("label");
      overlayLabel.append(overlayBox, " Overlay");
      const undo = document.createElement("button");
      undo.textContent = "Undo";
      undo.style.cssText = "padding: 2px 8px; border: 1px solid #888; border-radius: 4px; background: none; color: inherit; cursor: pointer;";
      bar.append(count, undo, overlayLabel);
      el.append(canvas, bar);

      const image = new Image();
      const overlay = new Image();
      let scale = 1;
      function draw() {
        if (!image.width) return;
        context.imageSmoothingEnabled = false;
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        if (overlayBox.checked && overlay.width) context.drawImage(overlay, 0, 0, canvas.width, canvas.height);
        const radius = Math.max(3, scale * 1.2);
        for (const [row, column] of model.get("points")) {
          const x = (column + 0.5) * scale, y = (row + 0.5) * scale;
          context.beginPath();
          context.arc(x, y, radius, 0, 2 * Math.PI);
          context.fillStyle = "rgba(255, 230, 0, 0.9)";
          context.fill();
          context.lineWidth = 1;
          context.strokeStyle = "black";
          context.stroke();
        }
        count.textContent = `${model.get("points").length} nuclei marked`;
      }
      function load() {
        image.onload = () => {
          scale = model.get("display") / image.width;
          canvas.width = Math.round(image.width * scale);
          canvas.height = Math.round(image.height * scale);
          draw();
        };
        image.src = model.get("background");
        overlay.onload = draw;
        if (model.get("overlay")) overlay.src = model.get("overlay");
      }
      canvas.addEventListener("click", (event) => {
        const box = canvas.getBoundingClientRect();
        const column = ((event.clientX - box.left) / box.width) * image.width - 0.5;
        const row = ((event.clientY - box.top) / box.height) * image.height - 0.5;
        const points = [...model.get("points")];
        // A click on a mark removes it, elsewhere adds one
        const near = points.findIndex(([r, c]) => Math.hypot(r - row, c - column) * scale <= Math.max(6, scale * 1.5));
        if (near >= 0) points.splice(near, 1);
        else points.push([row, column]);
        model.set("points", points);
        model.save_changes();
        draw();
      });
      undo.addEventListener("click", () => {
        model.set("points", model.get("points").slice(0, -1));
        model.save_changes();
        draw();
      });
      overlayBox.addEventListener("change", draw);
      model.on("change:points", draw);
      model.on("change:background", load);
      model.on("change:overlay", load);
      load();
    }
    export default { render };
    """

    background = traitlets.Unicode("").tag(sync=True)
    overlay = traitlets.Unicode("").tag(sync=True)
    points = traitlets.List([]).tag(sync=True)
    display = traitlets.Int(640).tag(sync=True)

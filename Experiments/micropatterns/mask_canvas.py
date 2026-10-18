"""A canvas widget for drawing a micropattern adhesion mask with the mouse.

Wrap it with ``mo.ui.anywidget(MaskCanvas(...))`` in marimo. The mask is an
``height`` x ``width`` grid of 0/1 bytes, sent to Python as base64 when a
stroke ends. ``encode_mask`` and ``decode_mask`` convert to and from NumPy.
The JavaScript is plain canvas code, so it can be reused in ``WebDemo/``.
"""

import base64

import anywidget
import numpy as np
import traitlets


def encode_mask(mask):
    """Boolean ``[H, W]`` array to the widget's base64 string."""

    return base64.b64encode(np.asarray(mask, dtype=bool).astype(np.uint8).tobytes()).decode()


def decode_mask(encoded, height, width):
    """The widget's base64 string to a boolean ``[H, W]`` array."""

    values = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    if values.size != height * width:
        raise ValueError(f"mask has {values.size} pixels, expected {height} x {width}")
    return values.reshape(height, width).astype(bool)


class MaskCanvas(anywidget.AnyWidget):
    """Paint (or erase) the adhesive region on a grid.

    ``background`` is an optional image (data URL) drawn under the mask, for
    example the initial biology. Outside the mask it is dimmed. ``reset_mask``
    is the mask restored by the Reset button. Keys: d draws, e erases.
    """

    _esm = """
    function render({ model, el }) {
      const view = 512;
      const canvas = document.createElement("canvas");
      canvas.width = view;
      canvas.height = view;
      canvas.tabIndex = 0;
      canvas.style.cssText = "cursor: crosshair; touch-action: none; border-radius: 4px; background: black; max-width: 100%;";
      const bar = document.createElement("div");
      bar.style.cssText = "display: flex; gap: 6px; align-items: center; margin-top: 6px; font: 12px sans-serif; flex-wrap: wrap;";
      const button = (text, action) => {
        const b = document.createElement("button");
        b.textContent = text;
        b.style.cssText = "padding: 2px 8px; border: 1px solid #888; border-radius: 4px; background: none; color: inherit; cursor: pointer;";
        b.addEventListener("click", action);
        bar.append(b);
        return b;
      };
      const modeButton = button("", () => setMode(model.get("mode") === "draw" ? "erase" : "draw"));
      button("Clear", () => { mask.fill(0); draw(); send(); });
      button("Fill", () => { mask.fill(1); draw(); send(); });
      button("Invert", () => { for (let i = 0; i < mask.length; i++) mask[i] = 1 - mask[i]; draw(); send(); });
      button("Reset", () => { mask = decode(model.get("reset_mask")); draw(); send(); });
      const sizeLabel = document.createElement("span");
      const size = document.createElement("input");
      size.type = "range";
      size.min = "1";
      size.max = "40";
      size.step = "0.5";
      size.value = String(model.get("brush_radius"));
      size.addEventListener("input", () => { model.set("brush_radius", Number(size.value)); model.save_changes(); draw(); });
      bar.append(sizeLabel, size);
      el.append(canvas, bar);

      const context = canvas.getContext("2d");
      const overlay = document.createElement("canvas");
      const picture = new Image();
      const H = () => model.get("height");
      const W = () => model.get("width");
      let mask = decode(model.get("mask"));
      let last = null;
      let hover = null;

      function decode(text) {
        const out = new Uint8Array(H() * W());
        if (!text) return out;
        const binary = atob(text);
        if (binary.length !== out.length) return out;
        for (let i = 0; i < out.length; i++) out[i] = binary.charCodeAt(i);
        return out;
      }
      function encode() {
        let binary = "";
        for (let i = 0; i < mask.length; i += 8192) {
          binary += String.fromCharCode.apply(null, mask.subarray(i, i + 8192));
        }
        return btoa(binary);
      }
      function send() {
        model.set("mask", encode());
        model.save_changes();
      }
      function setMode(mode) {
        model.set("mode", mode);
        model.save_changes();
        modeButton.textContent = mode === "draw" ? "Mode: draw" : "Mode: erase";
      }
      // Grid coordinates of a pointer event, allowing for CSS scaling.
      function toGrid(event) {
        const box = canvas.getBoundingClientRect();
        return {
          x: (event.clientX - box.left) / box.width * W() - 0.5,
          y: (event.clientY - box.top) / box.height * H() - 0.5,
        };
      }
      function stamp(point) {
        const r = model.get("brush_radius");
        const value = model.get("mode") === "draw" ? 1 : 0;
        const x0 = Math.max(0, Math.floor(point.x - r)), x1 = Math.min(W() - 1, Math.ceil(point.x + r));
        const y0 = Math.max(0, Math.floor(point.y - r)), y1 = Math.min(H() - 1, Math.ceil(point.y + r));
        for (let y = y0; y <= y1; y++) {
          for (let x = x0; x <= x1; x++) {
            if ((x - point.x) ** 2 + (y - point.y) ** 2 <= r * r) mask[y * W() + x] = value;
          }
        }
      }
      // Stamp along the segment so fast strokes have no gaps.
      function stroke(from, to) {
        const spacing = Math.max(0.5, 0.5 * model.get("brush_radius"));
        const steps = Math.max(1, Math.ceil(Math.hypot(to.x - from.x, to.y - from.y) / spacing));
        for (let i = 1; i <= steps; i++) {
          stamp({ x: from.x + (to.x - from.x) * i / steps, y: from.y + (to.y - from.y) * i / steps });
        }
      }
      function draw() {
        const h = H(), w = W();
        context.imageSmoothingEnabled = false;
        context.fillStyle = "black";
        context.fillRect(0, 0, view, view);
        if (picture.complete && picture.naturalWidth) {
          context.drawImage(picture, 0, 0, view, view);
        } else {
          context.fillStyle = "#777";
          context.fillRect(0, 0, view, view);
        }
        overlay.width = w;
        overlay.height = h;
        const pixels = new ImageData(w, h);
        for (let i = 0; i < mask.length; i++) {
          if (!mask[i]) {
            pixels.data[4 * i + 3] = 200;
          } else {
            // Outline: a mask pixel next to a non-mask pixel.
            const x = i % w, y = (i - x) / w;
            const edge = (x > 0 && !mask[i - 1]) || (x < w - 1 && !mask[i + 1])
              || (y > 0 && !mask[i - w]) || (y < h - 1 && !mask[i + w]);
            if (edge) pixels.data.set([255, 255, 255, 255], 4 * i);
          }
        }
        overlay.getContext("2d").putImageData(pixels, 0, 0);
        context.drawImage(overlay, 0, 0, view, view);
        if (hover) {
          const scale = view / w;
          context.strokeStyle = model.get("mode") === "draw" ? "#4caf50" : "#e53935";
          context.lineWidth = 1.5;
          context.beginPath();
          context.arc((hover.x + 0.5) * scale, (hover.y + 0.5) * scale, model.get("brush_radius") * scale, 0, 2 * Math.PI);
          context.stroke();
        }
        let area = 0;
        for (let i = 0; i < mask.length; i++) area += mask[i];
        sizeLabel.textContent = `${w} x ${h} grid · area ${(100 * area / mask.length).toFixed(1)}% · brush ${model.get("brush_radius")} px`;
      }

      canvas.addEventListener("pointerdown", (event) => {
        canvas.setPointerCapture(event.pointerId);
        canvas.focus();
        last = toGrid(event);
        stamp(last);
        draw();
      });
      canvas.addEventListener("pointermove", (event) => {
        hover = toGrid(event);
        if (last) {
          stroke(last, hover);
          last = hover;
        }
        draw();
      });
      canvas.addEventListener("pointerup", () => {
        if (!last) return;
        last = null;
        send();
      });
      canvas.addEventListener("pointerleave", () => { hover = null; draw(); });
      canvas.addEventListener("keydown", (event) => {
        if (event.key === "d") setMode("draw");
        if (event.key === "e") setMode("erase");
        draw();
      });

      picture.onload = draw;
      if (model.get("background")) picture.src = model.get("background");
      model.on("change:background", () => { picture.src = model.get("background"); });
      model.on("change:mask", () => { mask = decode(model.get("mask")); draw(); });
      model.on("change:brush_radius", () => { size.value = String(model.get("brush_radius")); draw(); });
      model.on("change:mode", () => setMode(model.get("mode")));
      setMode(model.get("mode"));
      draw();
    }
    export default { render };
    """
    height = traitlets.Int(64).tag(sync=True)
    width = traitlets.Int(64).tag(sync=True)
    mask = traitlets.Unicode("").tag(sync=True)
    reset_mask = traitlets.Unicode("").tag(sync=True)
    background = traitlets.Unicode("").tag(sync=True)
    brush_radius = traitlets.Float(6.0).tag(sync=True)
    mode = traitlets.Unicode("draw").tag(sync=True)

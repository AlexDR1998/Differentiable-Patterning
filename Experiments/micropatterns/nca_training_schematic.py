"""Schematic of how an NCA is defined and trained on time series data.

Run with:
    marimo edit Experiments/micropatterns/nca_training_schematic.py

- Panel a: one NCA update step. State channels (observed, hidden, boundary mask)
  → fixed perception kernels → per-pixel network → gated linear unit → random
  fire mask (asynchronous updates) → residual add → boundary → next state
- Panel b: training as a mock phase portrait, time against a 1D state space.
  Every interval between observations is rolled out at once (parallel in time),
  for a batch per timepoint (one fibre each); fibres turn orange where a
  regulariser penalises them. Predictions are compared with the data and become
  the next start states (or the data does, with probability p).
  A small plot shows loss weight curricula over training
- Panel a's colony is the toy local rule of project_overview.py (panel b), at a
  chosen number of pixels; panel b's values are made up. No data is loaded
- Figures are saved to ``figures/schematics/``
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="NCA training schematic")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

    from Experiments.micropatterns import nodal_ko_schematic as ko_drawing
    from Experiments.micropatterns import project_overview as overview
    from Experiments.micropatterns.schematics import colour_or

    # 3x3 perception kernels, as in Common/model/spatial_operators.py (SMOOTHING=1)
    KERNELS = {
        "ID": (np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float), "identity"),
        "LAP": (np.array([[0.25, 0.5, 0.25], [0.5, -3, 0.5], [0.25, 0.5, 0.25]]), "Laplacian"),
        "DIFF": (np.outer([1, 2, 1], [-1, 0, 1]) / 8, r"$|\nabla|$"),
        "GRAD": (np.outer([1, 2, 1], [-1, 0, 1]) / 8, r"$\nabla_x, \nabla_y$"),
        "AV": (np.full((3, 3), 1 / 9), "average"),
    }

    # Example loss weight curricula for panel b, in the form of
    # Common/trainer/config.py:LossWeightScheduleConfig. Illustrative only.
    CURRICULA = (
        {"type": "linear", "initial": 1.0, "final": 0.3, "start": 0.2, "end": 0.6, "stages": None},
        {"type": "cosine", "initial": 0.0, "final": 1.0, "start": 0.2, "end": 0.7, "stages": None},
        {"type": "linear", "initial": 0.0, "final": 0.8, "start": 0.4, "end": 1.0, "stages": 4},
    )
    CURVE_COLOURS = ("#1f77b4", "#8c6bb1", "#41ab5d")
    # Data reinjection probability: starts at p, falls linearly after a fraction of training
    REINJECTION = {"type": "linear", "initial": 0.5, "final": 0.15, "start": 0.25, "end": 1.0, "stages": None}


@app.function(hide_code=True)
def schedule_factor(progress, curve):
    """Weight factor at training progress in [0, 1]; a numpy copy of
    NCA/trainer/loss_schedule.py:schedule_factor."""
    transition = np.clip((progress - curve["start"]) / (curve["end"] - curve["start"]), 0, 1)
    if curve["stages"]:
        stage = np.minimum((transition * curve["stages"]).astype(int), curve["stages"] - 1)
        transition = stage / (curve["stages"] - 1)
    if curve["type"] == "cosine":
        transition = 0.5 - 0.5 * np.cos(np.pi * transition)
    return curve["initial"] + (curve["final"] - curve["initial"]) * transition


@app.function(hide_code=True)
def text_width(text, font):
    """Rough width of ``text`` in data units, as estimated in project_overview.py."""
    return 0.09 * max(len(line) for line in text.split("\n")) * font / 7


@app.function(hide_code=True)
def smooth_field(mask, seed, scale=3.0):
    """Smooth random field in [0, 1], zero outside ``mask``: stands in for a hidden channel."""
    rng = np.random.default_rng(seed)
    size = mask.shape[0]
    frequencies = np.fft.fftfreq(size)
    radius = np.hypot(*np.meshgrid(frequencies, frequencies))
    noise = np.fft.fft2(rng.normal(size=mask.shape))
    field = np.real(np.fft.ifft2(noise * np.exp(-(radius * size / scale) ** 2)))
    field = (field - field.min()) / (np.ptp(field) + 1e-9)
    return field * mask


@app.function(hide_code=True)
def resample_toy(toy, pixels):
    """Toy frames [F, 3, H, W] resampled to ``pixels`` per side (linear interpolation),
    so the pattern looks the same at any resolution, with the colony mask redrawn."""
    from scipy.ndimage import zoom

    frames, _ = toy
    resampled = zoom(frames, (1, 1, pixels / frames.shape[-2], pixels / frames.shape[-1]), order=1)
    rows, cols = np.indices((pixels, pixels))
    centre = (pixels - 1) / 2
    mask = np.hypot(rows - centre, cols - centre) <= pixels / 2 - 0.5
    return np.clip(resampled, 0, 1) * mask, mask


@app.function(hide_code=True)
def grey_rgba(values, mask):
    """RGBA of one channel in [0, 1], white on black, transparent outside ``mask``."""
    rgb = np.repeat(np.clip(values, 0, 1)[..., None], 3, axis=-1)
    return np.concatenate([rgb, mask[..., None].astype(float)], axis=-1)


@app.function(hide_code=True)
def operator(ax, centre, symbol, style, colour=None, radius=0.16):
    """Circle with a symbol in it, e.g. + or ⊙."""
    theme = style["theme"]
    colour = colour or theme["fg"]
    ax.add_patch(Circle(centre, radius, facecolor=theme["bg"], edgecolor=colour, lw=1.0, zorder=8))
    ax.text(centre[0], centre[1], symbol, ha="center", va="center", fontsize=style["font"] + 1, color=colour,
            zorder=9)


@app.function(hide_code=True)
def rounded_box(ax, corner, width, height, colour, z=2):
    """Filled, outlined rounded box, like the training block of project_overview.py."""
    for kwargs in (dict(facecolor=colour, alpha=0.15, edgecolor="none", zorder=z),
                   dict(fill=False, edgecolor=colour, lw=1.0, zorder=z + 0.1)):
        ax.add_patch(FancyBboxPatch(corner, width, height, boxstyle="round,pad=0,rounding_size=0.1", **kwargs))


@app.function(hide_code=True)
def polyline_arrow(ax, points, colour, lw=0.9, **kwargs):
    """Line through ``points`` with an arrow head on the last segment."""
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=colour, lw=lw, solid_capstyle="butt", **kwargs)
    overview.draw_arrow(ax, points[-2], points[-1], colour, lw=lw, **kwargs)


@app.function(hide_code=True)
def draw_kernel(ax, corner, size, kernel):
    """3x3 kernel as a small grid: red positive, blue negative, white zero."""
    cell = size / 3
    scale = np.abs(kernel).max() or 1
    for i in range(3):
        for j in range(3):
            value = kernel[i, j] / scale
            colour = (1, 1 - value, 1 - value) if value > 0 else (1 + value, 1 + value, 1)
            ax.add_patch(Rectangle((corner[0] + j * cell, corner[1] + (2 - i) * cell), cell, cell,
                                   facecolor=colour, edgecolor="#555555", lw=0.3, zorder=4))


@app.function(hide_code=True)
def state_images(toy, style):
    """Card images for panel a: the state at the chosen frame and the next one, the
    update, a random fire mask and the masked update."""
    frames, mask = toy
    k = min(style["zoom_frame"], len(frames) - 2)
    # Hidden channels: smooth fields that drift a little between the two states
    hidden = [[grey_rgba(smooth_field(mask, 10 * s + t), mask) for s in range(3)] for t in (0, 1)]
    mask_rgba = grey_rgba(mask.astype(float), mask)
    fired = (np.random.default_rng(style["seed"]).uniform(size=mask.shape) < style["fire_rate"]) & mask
    change = np.clip(np.abs(frames[k + 1] - frames[k]) * style["update_gain"], 0, 1)
    return {
        "before": {"observed": overview.toy_rgba(frames[k], mask), "hidden": hidden[0], "mask": mask_rgba},
        "after": {"observed": overview.toy_rgba(frames[k + 1], mask), "hidden": hidden[1], "mask": mask_rgba},
        "update": overview.toy_rgba(change, mask),
        "fired": grey_rgba(fired.astype(float), mask),
        "fired_update": overview.toy_rgba(change * fired, mask),
        "mask": mask_rgba,
    }


@app.function(hide_code=True)
def column_rows(style):
    """Channel groups of the state: the mask is a state channel only with a soft boundary."""
    return ("observed", "hidden", "mask") if style["boundary_mode"] == "soft" else ("observed", "hidden")


@app.function(hide_code=True)
def draw_state_column(ax, x, centre_y, images, style, label, side, label_y):
    """State channels as a column of cards (hidden channels as a small stack). A bracket
    on ``side`` ("right" or "left") joins them to one point. Labels go on the other side;
    ``label`` is centred at height ``label_y``. Returns (bracket x, column bottom)."""
    theme = style["theme"]
    small = style["font"] - 1
    size, gap = style["card"], style["row_gap"]
    rows = column_rows(style)
    height = len(rows) * size + (len(rows) - 1) * gap
    top = centre_y + height / 2
    names = {"observed": style["observed_label"], "hidden": style["hidden_label"], "mask": style["mask_label"]}
    middles = []
    for k, name in enumerate(rows):
        y = top - (k + 1) * size - k * gap
        middles.append(y + size / 2)
        if name == "hidden":
            step = 0.07
            tiles = [(rgba, ()) for rgba in images["hidden"]]
            overview.draw_paper_stack(ax, tiles, (x, y), size - 2 * step, (step, step), style, pixelated=True)
        else:
            edge = style["mask_colour"] if name == "mask" else None
            overview.draw_card(ax, (x, y), size, 10, style, images[name], edge=edge, pixelated=True)
        text_x, ha = (x - 0.1, "right") if side == "right" else (x + size + 0.1, "left")
        ax.text(text_x, y + size / 2, names[name], ha=ha, va="center", fontsize=small, linespacing=1.2,
                color=style["mask_colour"] if name == "mask" else theme["fg"])
    ax.text(x + size / 2, label_y, label, ha="center", va="center", fontsize=style["font"] + 1)

    # Bracket: a short stub from each card to a shared vertical bar
    edge_x = x + size + 0.05 if side == "right" else x - 0.05
    bar_x = edge_x + (0.2 if side == "right" else -0.2)
    for middle in middles:
        ax.plot([edge_x, bar_x], [middle, middle], color=theme["fg"], lw=0.8)
    ax.plot([bar_x, bar_x], [min(middles), max(middles)], color=theme["fg"], lw=0.8)
    return bar_x, top - height


@app.function(hide_code=True)
def draw_network(ax, x, centre_y, style):
    """Per-pixel network: 1x1 conv → ReLU → 1x1 conv, drawn as three bars. The last bar
    is split into the GLU halves a and b (gated) and a firing logit p (learned firing).
    Returns (right edge, centre height of each part of the last bar)."""
    theme = style["theme"]
    model, accent, small = style["model_colour"], theme["accent"], style["font"] - 1
    bar_w, bar_gap, bar_h = 0.22, 0.32, style["network_height"]
    learned = style["firing"] == "learned"
    parts = (["a", "b"] if style["gated"] else [r"$\Delta$"]) + (["p"] if learned else [])
    heights = [1.0] * (len(parts) - learned) + ([0.35] if learned else [])
    for k, name in enumerate(("1×1 conv", "ReLU", "1×1 conv")):
        left = x + k * (bar_w + bar_gap)
        if k < 2:
            ax.add_patch(Rectangle((left, centre_y - bar_h / 2), bar_w, bar_h, facecolor=model,
                                   alpha=0.25 if k == 0 else 0.08, edgecolor=model, lw=0.8, zorder=4))
            ax.text(left + bar_w / 2, centre_y, name, ha="center", va="center", rotation=90, fontsize=small - 0.5,
                    zorder=5)
            overview.draw_arrow(ax, (left + bar_w + 0.03, centre_y), (left + bar_w + bar_gap - 0.03, centre_y),
                                theme["fg"], lw=0.8)
            continue
        unit = bar_h / sum(heights)
        y = centre_y + bar_h / 2
        centres = {}
        for part, h in zip(parts, heights):
            colour = accent if part == "p" else model
            ax.add_patch(Rectangle((left, y - h * unit), bar_w, h * unit, facecolor=colour, alpha=0.25,
                                   edgecolor=colour, lw=0.8, zorder=4))
            ax.text(left + bar_w / 2, y - h * unit / 2, part, ha="center", va="center", fontsize=small, zorder=5)
            centres[part] = y - h * unit / 2
            y -= h * unit
    right = x + 3 * bar_w + 2 * bar_gap
    ax.text((x + right) / 2, centre_y + bar_h / 2 + 0.1, style["network_label"], ha="center", va="bottom",
            fontsize=small, linespacing=1.2)
    return right, centres


@app.function(hide_code=True)
def draw_architecture_panel(ax, toy, style):
    """Panel a: one NCA update step, left to right."""
    theme = style["theme"]
    fg, muted, accent, mask_colour = theme["fg"], theme["muted"], theme["accent"], style["mask_colour"]
    font, small = style["font"], style["font"] - 1
    images = state_images(toy, style)
    size, cy = style["card"], 0.0
    learned = style["firing"] == "learned"
    names = [style[f"{r}_label"] for r in column_rows(style)]

    # State x_t. The residual line runs at the height of its label, above the network
    left = max(text_width(n, small) for n in names) + 0.15
    rows = len(column_rows(style))
    column_top = cy + (rows * size + (rows - 1) * style["row_gap"]) / 2
    skip_y = max(column_top + 0.25, cy + style["network_height"] / 2 + 0.65)
    bar_x, column_bottom = draw_state_column(ax, left, cy, images["before"], style, "$x_t$", "right", skip_y)
    x = bar_x + 0.45
    overview.draw_arrow(ax, (bar_x, cy), (x - 0.12, cy), fg)

    # Perception: fixed local kernels, applied to every channel
    kernels = [k for k in style["kernels"] if k in KERNELS] or ["ID"]
    k_size, k_gap = 0.4, 0.2
    k_height = len(kernels) * k_size + (len(kernels) - 1) * k_gap
    k_top = cy + k_height / 2
    for k, name in enumerate(kernels):
        ky = k_top - (k + 1) * k_size - k * k_gap
        draw_kernel(ax, (x, ky), k_size, KERNELS[name][0])
        ax.text(x + k_size + 0.14, ky + k_size / 2, KERNELS[name][1], ha="left", va="center", fontsize=small - 0.5)
    ax.add_patch(FancyBboxPatch((x - 0.08, k_top - k_height - 0.08), k_size + 0.16, k_height + 0.16,
                                boxstyle="round,pad=0,rounding_size=0.05", fill=False,
                                edgecolor=style["neighbourhood_colour"], lw=1.0, zorder=5))
    ax.text(x + k_size / 2, k_top + 0.25, style["perception_label"], ha="center", va="bottom", fontsize=small,
            linespacing=1.2)
    x += k_size + 0.25 + 1.4 * max(text_width(KERNELS[k][1].strip("$\\"), small - 0.5) for k in kernels)
    overview.draw_arrow(ax, (x, cy), (x + 0.3, cy), fg)
    x += 0.35

    net_right, parts = draw_network(ax, x, cy, style)
    x = net_right

    # Gated linear unit: Δx = a ⊙ s(b), s the logistic sigmoid
    if style["gated"]:
        a_y, b_y = parts["a"], parts["b"]
        sigmoid_x = x + 0.2
        gate = (sigmoid_x + 0.8, cy)
        ax.plot([x, sigmoid_x + 0.32], [a_y, a_y], color=fg, lw=0.8)
        overview.draw_arrow(ax, (sigmoid_x + 0.32, a_y), (gate[0] - 0.11, gate[1] + 0.11), fg, lw=0.8)
        ax.plot([x, sigmoid_x], [b_y, b_y], color=fg, lw=0.8)
        ax.add_patch(Rectangle((sigmoid_x, b_y - 0.13), 0.32, 0.26, facecolor=theme["bg"], edgecolor=fg, lw=0.8,
                               zorder=6))
        ax.plot(sigmoid_x + 0.04 + 0.24 * np.linspace(0, 1, 30),
                b_y - 0.08 + 0.16 / (1 + np.exp(-12 * (np.linspace(0, 1, 30) - 0.5))), color=fg, lw=0.7, zorder=7)
        overview.draw_arrow(ax, (sigmoid_x + 0.32, b_y), (gate[0] - 0.11, gate[1] - 0.11), fg, lw=0.8)
        operator(ax, gate, "⊙", style, colour=style["model_colour"])
        ax.text(gate[0], a_y + 0.15, style["gate_label"], ha="center", va="bottom", fontsize=small,
                color=style["model_colour"], linespacing=1.2)
        x = gate[0] + 0.16

    # Update Δx
    overview.draw_arrow(ax, (x + 0.03, cy), (x + 0.5, cy), fg)
    x += 0.55
    overview.draw_card(ax, (x, cy - size / 2), size, 10, style, images["update"], pixelated=True)
    ax.text(x + size / 2, cy + size / 2 + 0.08, r"$\Delta x$", ha="center", va="bottom", fontsize=font)
    x += size

    # Asynchronous updates: multiply by a random fire mask
    multiply = (x + 0.55, cy)
    overview.draw_arrow(ax, (x + 0.03, cy), (multiply[0] - 0.17, cy), fg)
    operator(ax, multiply, "⊙", style, colour=accent)
    fire_top = cy - size / 2 - 0.15
    fire_x = multiply[0] - size / 2
    overview.draw_card(ax, (fire_x, fire_top - size), size, 10, style, images["fired"], edge=accent,
                       pixelated=True)
    overview.draw_arrow(ax, (multiply[0], fire_top + 0.02), (multiply[0], cy - 0.17), accent)
    ax.text(multiply[0], fire_top - size - 0.08, style["fire_label_learned" if learned else "fire_label"],
            ha="center", va="top", fontsize=small, color=accent, linespacing=1.2)
    if learned:
        # The firing logit p sets each pixel's firing probability
        p_y = parts["p"]
        polyline_arrow(ax, [(net_right + 0.02, p_y), (fire_x - 0.25, p_y), (fire_x - 0.25, fire_top - size / 2),
                            (fire_x - 0.03, fire_top - size / 2)], accent, ls=(0, (3, 1.5)))
    x = multiply[0] + 0.17

    overview.draw_arrow(ax, (x + 0.03, cy), (x + 0.4, cy), fg)
    x += 0.45
    overview.draw_card(ax, (x, cy - size / 2), size, 10, style, images["fired_update"], pixelated=True)
    ax.text(x + size / 2, cy + size / 2 + 0.08, r"$\sigma \odot \Delta x$", ha="center", va="bottom",
            fontsize=font)
    x += size

    # Residual update x_t + σ ⊙ Δx, with the skip connection over the top
    add = (x + 0.45, cy)
    overview.draw_arrow(ax, (x + 0.03, cy), (add[0] - 0.17, cy), fg)
    operator(ax, add, "+", style)
    ax.plot([left + size / 2 + 0.4, add[0]], [skip_y, skip_y], color=muted, lw=0.9)
    overview.draw_arrow(ax, (add[0], skip_y), (add[0], cy + 0.17), muted, lw=0.9)
    ax.text(add[0] - 0.15, skip_y + 0.1, style["skip_label"], ha="right", va="bottom", fontsize=small, color=muted)
    x = add[0] + 0.17

    # Boundary: the adhesion mask m is applied after every step
    overview.draw_arrow(ax, (x + 0.03, cy), (x + 0.3, cy), fg)
    x += 0.32
    box_w, box_h = 1.8, 0.85
    rounded_box(ax, (x, cy - box_h / 2), box_w, box_h, mask_colour, z=5)
    ax.text(x + box_w / 2, cy + 0.13, style["boundary_label"], ha="center", va="center", fontsize=small,
            fontweight="bold", zorder=6)
    rule = r"mask channel $\leftarrow m$" if style["boundary_mode"] == "soft" else r"$x \leftarrow x \odot m$"
    ax.text(x + box_w / 2, cy - 0.15, rule, ha="center", va="center", fontsize=small - 0.5, zorder=6)
    m_size = 0.8 * size
    m_top = cy - box_h / 2 - 0.35
    overview.draw_card(ax, (x + (box_w - m_size) / 2, m_top - m_size), m_size, 10, style, images["mask"],
                       edge=mask_colour, pixelated=True)
    overview.draw_arrow(ax, (x + box_w / 2, m_top + 0.02), (x + box_w / 2, cy - box_h / 2 - 0.02), mask_colour)
    ax.text(x + box_w / 2, m_top - m_size - 0.08, style["adhesion_label"], ha="center", va="top",
            fontsize=small, color=mask_colour, linespacing=1.2)
    x += box_w

    # State x_{t+1}
    column_x = x + 0.8
    overview.draw_arrow(ax, (x + 0.03, cy), (column_x - 0.25, cy), fg)
    draw_state_column(ax, column_x, cy, images["after"], style, "$x_{t+1}$", "left", skip_y)
    right = column_x + size + 0.15 + max(text_width(n, small) for n in names)

    # Repeat: dashed loop from x_{t+1} back to x_t, under everything
    low = min(fire_top - size, m_top - m_size) - 0.65
    if style["repeat_label"]:
        low_label = low
        low = min(low, column_bottom - 0.25)
        polyline_arrow(ax, [(column_x + size / 2, column_bottom - 0.05), (column_x + size / 2, low),
                            (left + size / 2, low), (left + size / 2, column_bottom - 0.05)], muted,
                       ls=(0, (3, 2)))
        ax.text((left + column_x + size) / 2, low - 0.08, style["repeat_label"], ha="center", va="top",
                fontsize=small, color=muted)
        low = min(low - 0.3, low_label)
    ax.set_xlim(-0.1, right)
    ax.set_ylim(low - 0.2, skip_y + 0.65)


@app.function(hide_code=True)
def mock_trajectories(count, style):
    """Mock 1D values in state space for panel b, for a batch of B members per timepoint.

    Returns data [T, B] (replicates around a mean curve), predictions [T, B] (offset
    from the data, alternately above and below, kept inside [0, 1]) and a bulge per
    interval and member [T - 1, B]. Values in [0, 1] are the valid range of the state."""
    rng = np.random.default_rng(style["seed"] + 7)
    batch, error = style["batch"], style["prediction_error"]
    t = np.linspace(0, 1, count)
    mean = 0.15 + 0.6 * (1 - np.exp(-2.5 * t)) / (1 - np.exp(-2.5)) + 0.03 * rng.normal(size=count)
    data = mean[:, None] + 0.05 * rng.normal(size=(count, batch))
    signs = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
    signs = np.where((mean + signs * error > 0.95) | (mean + signs * error < 0.05), -signs, signs)
    shift = signs * error * rng.uniform(0.8, 1.0, size=count)
    predicted = (mean + shift)[:, None] + 0.05 * rng.normal(size=(count, batch))
    predicted[0] = data[0]
    bulges = 0.04 * rng.normal(size=(count - 1, 1)) + 0.06 * rng.normal(size=(count - 1, batch))
    if 0 <= style["overshoot_interval"] < count - 1:
        bulges[style["overshoot_interval"]] += style["overshoot"] * rng.uniform(0.6, 1.2, size=batch)
    return data, predicted, bulges


@app.function(hide_code=True)
def rollout_curve(start, end, bulge, tau):
    """State along one rollout, at fractions ``tau`` of the interval."""
    return start + (end - start) * (3 * tau ** 2 - 2 * tau ** 3) + bulge * np.sin(np.pi * tau)


@app.function(hide_code=True)
def draw_fibre(ax, xs, values, to_y, colour, outside_colour, **kwargs):
    """One rollout curve, in ``outside_colour`` wherever it leaves the valid range [0, 1]."""
    outside = (values > 1) | (values < 0)
    # Widen the outside parts by one sample so the two colours join up
    widened = outside | np.roll(outside, 1) | np.roll(outside, -1)
    ys = to_y(values)
    ax.plot(xs, np.where(outside, np.nan, ys), color=colour, **kwargs)
    if outside.any():
        ax.plot(xs, np.where(widened, ys, np.nan), color=outside_colour, **{**kwargs, "zorder": kwargs["zorder"] + 1})


@app.function(hide_code=True)
def draw_training_panel(ax, toy, style):
    """Panel b: training as a phase portrait, time on x and a 1D state space on y.

    Every interval [t_i, t_{i+1}] is rolled out at once (parallel in time), for a batch
    of members per timepoint: one fibre each. A fibre starts at the data at t_0, later
    at its previous prediction (pool) or, with probability p, at the data. Parts of a
    rollout outside the valid range are penalised by a regulariser. Predictions are
    compared with the data (loss). Right: loss weight curricula over training."""
    theme = style["theme"]
    fg, muted, model = theme["fg"], theme["muted"], style["model_colour"]
    loss_colour, reg_colour = style["loss_colour"], style["regulariser_colour"]
    font, small = style["font"], style["font"] - 1
    count = len(toy[0])
    data, predicted, bulges = mock_trajectories(count, style)
    width, height = style["phase_width"], style["phase_height"]
    v_low, v_high = -0.18, 1.32
    hours = np.linspace(0, ko_drawing.TIMEPOINTS[-1], count)

    def to_x(fraction):
        return width * fraction

    def to_y(value):
        return height * (value - v_low) / (v_high - v_low)

    times = np.linspace(0, 1, count)

    # Outside the valid range [0, 1]: penalised by the state range regulariser
    for lower, upper in ((1.0, v_high), (v_low, 0.0)):
        ax.add_patch(Rectangle((0, to_y(lower)), width, to_y(upper) - to_y(lower), facecolor=reg_colour,
                               alpha=0.08, edgecolor="none", zorder=1))
        ax.add_patch(Rectangle((0, to_y(lower)), width, to_y(upper) - to_y(lower), fill=False, hatch="////",
                               edgecolor=reg_colour, alpha=0.25, lw=0, zorder=1))
    for value in (0.0, 1.0):
        ax.plot([0, width], [to_y(value)] * 2, color=reg_colour, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.text(width - 0.05, (to_y(v_low) + to_y(0.0)) / 2, style["region_label"], ha="right", va="center",
            fontsize=small, color=reg_colour, zorder=3)

    # Axes: time and the state space
    ax.plot([0, width], [0, 0], color=fg, lw=0.7)
    overview.draw_arrow(ax, (0, 0), (0, height + 0.1), fg, lw=0.7)
    for fraction, hour in zip(times, hours):
        ax.plot([to_x(fraction)] * 2, [0, -0.06], color=fg, lw=0.7)
        ax.text(to_x(fraction), -0.1, f"{hour:g}h", ha="center", va="top", fontsize=small)
    ax.text(width, -0.38, style["time_label"], ha="right", va="top", fontsize=small)
    ax.text(-0.12, height / 2, style["space_label"], ha="center", va="bottom", rotation=90, rotation_mode="anchor",
            fontsize=small)

    # Rollouts: one fibre per batch member and interval
    tau = np.linspace(0, 1, 200)
    # Finer, see-through lines when a batch has several members
    if style["batch"] > 1:
        fibre_width, fibre_alpha = style["fibre_width"], style["fibre_alpha"]
    else:
        fibre_width, fibre_alpha = 1.4, 1.0
    for i in range(count - 1):
        x0, x1 = to_x(times[i]), to_x(times[i + 1])
        xs = x0 + (x1 - x0) * tau
        for b in range(style["batch"]):
            if i > 0 and style["show_reinjection_paths"]:
                # The same member started from the data instead (probability p)
                ghost = rollout_curve(data[i, b], predicted[i + 1, b], bulges[i, b], tau)
                ax.plot(xs, to_y(ghost), color=muted, lw=0.7 * fibre_width, ls=(0, (2, 1.5)), alpha=fibre_alpha,
                        zorder=4)
            start = data[0, b] if i == 0 else predicted[i, b]
            values = rollout_curve(start, predicted[i + 1, b], bulges[i, b], tau)
            draw_fibre(ax, xs, values, to_y, model, reg_colour, lw=fibre_width, alpha=fibre_alpha, zorder=5)
        if i == 0:
            low = to_y(min(rollout_curve(data[0], predicted[1], bulges[0], 0.5)))
            ax.text((x0 + x1) / 2, low - 0.15, style["steps_label"], ha="left", va="top", fontsize=small,
                    color=model, zorder=8)

    # Loss: batch of predictions against the batch of data, at each observation time
    for i in range(1, count):
        x = to_x(times[i])
        ax.plot([x, x], [to_y(data[i].mean()), to_y(predicted[i].mean())], color=loss_colour, lw=1.6, zorder=7)
        ax.text(x + 0.14, (to_y(data[i].mean()) + to_y(predicted[i].mean())) / 2, r"$\mathcal{L}$", ha="left",
                va="center", fontsize=font, color=loss_colour, zorder=10)

    # Data and predictions, one marker per batch member
    for i, fraction in enumerate(times):
        x = to_x(fraction)
        ax.plot([x] * style["batch"], to_y(data[i]), "s", markersize=4.5, color=fg, markeredgecolor=theme["bg"],
                markeredgewidth=0.5, ls="none", zorder=9)
        if i > 0:
            ax.plot([x] * style["batch"], to_y(predicted[i]), "o", markersize=5, color=theme["bg"],
                    markeredgecolor=model, markeredgewidth=1.2, ls="none", zorder=9)

    # Parallel in time: one bracket, ticked at every observation time
    bracket_y = height + 0.35
    ax.plot([0, width], [bracket_y, bracket_y], color=model, lw=0.9)
    for fraction in times:
        ax.plot([to_x(fraction)] * 2, [bracket_y - 0.08, bracket_y], color=model, lw=0.9)
    ax.text(width / 2, bracket_y + 0.08, style["parallel_label"], ha="center", va="bottom", fontsize=font,
            color=model)

    # Key, under the time axis
    key_y = -0.75
    entries = [
        (dict(marker="s", markersize=4.5, color=fg, markeredgecolor=theme["bg"]), None, style["data_key"]),
        (dict(marker="o", markersize=5, color=theme["bg"], markeredgecolor=model, markeredgewidth=1.2), None,
         style["prediction_key"]),
        (None, dict(color=model, lw=1.0), style["state_key"]),
        (None, dict(color=reg_colour, lw=1.0), style["penalised_key"]),
    ]
    if style["show_reinjection_paths"]:
        entries.append((None, dict(color=muted, lw=0.9, ls=(0, (2, 1.5))), style["reinjection_key"]))
    # Two rows; each column as wide as its widest entry
    columns = [entries[k:k + 2] for k in range(0, len(entries), 2)]
    x = 0.05
    for column in columns:
        for row, (marker, line, text) in enumerate(column):
            y = key_y - row * 0.3
            if marker:
                ax.plot(x + 0.12, y, ls="none", **marker)
            else:
                ax.plot([x, x + 0.25], [y, y], **line)
            ax.text(x + 0.32, y, text, ha="left", va="center", fontsize=small)
        x += 0.32 + 1.2 * max(text_width(text, small) for *_, text in column) + 0.4
    key_right = x
    key_y -= 0.3
    bottom = key_y - 0.3

    # Loss curricula: weight factor of each loss term against training iteration
    # Same height and baseline as the phase portrait
    px, pw, ph = width + style["plot_gap"], 2.6, height
    py = 0.0
    ax.plot([px, px], [py, py + ph], color=fg, lw=0.7)
    ax.plot([px, px + pw], [py, py], color=fg, lw=0.7)
    progress = np.linspace(0, 1, 400)
    labels = [t.strip() for t in style["curve_labels"].split(",")]
    curves = list(zip(CURRICULA[:style["curve_count"]], CURVE_COLOURS, labels + [""] * 3))
    lines = [(schedule_factor(progress, c), colour, "-", text) for c, colour, text in curves]
    if style["show_reinjection"]:
        lines.append((schedule_factor(progress, REINJECTION), muted, (0, (3, 1.5)), style["reinjection_label"]))
    for values, colour, line, _ in lines:
        ax.plot(px + pw * progress, py + 0.9 * ph * values, color=colour, lw=1.3, ls=line, zorder=4)
    ax.text(px + pw / 2, py - 0.08, style["curve_x_label"], ha="center", va="top", fontsize=small)
    ax.text(px - 0.1, py + ph / 2, style["curve_y_label"], ha="center", va="bottom", rotation=90,
            rotation_mode="anchor", fontsize=small)
    ax.text(px + pw / 2, py + ph + 0.12, style["curve_title"], ha="center", va="bottom", fontsize=font)
    key_x = px + pw + 0.15
    for k, (_, colour, line, text) in enumerate(lines):
        ly = py + ph - 0.1 - k * 0.27
        ax.plot([key_x, key_x + 0.3], [ly, ly], color=colour, lw=1.3, ls=line)
        ax.text(key_x + 0.37, ly, text, ha="left", va="center", fontsize=small)
    plot_right = key_x + 0.4 + max(text_width(text, small) for *_, text in lines)

    # Total loss, under the plot
    if style["equation"]:
        ax.text((px + plot_right) / 2, py - 0.45, style["equation"], ha="center", va="top", fontsize=font)
    ax.set_xlim(-0.45, max(plot_right, key_right))
    ax.set_ylim(min(bottom, py - 1.0), bracket_y + 0.8)


@app.function(hide_code=True)
def schematic_figure(panels, toy, style):
    """The chosen panels stacked top to bottom; with ``style["same_scale"]`` every panel
    uses the same data-units-per-inch."""
    theme = style["theme"]
    draw = {
        "a": lambda ax: draw_architecture_panel(ax, toy, style),
        "b": lambda ax: draw_training_panel(ax, toy, style),
    }
    with plt.rc_context({"text.color": theme["fg"], "font.size": style["font"]}):
        fig = plt.figure(figsize=(style["width"], 2.0), facecolor=theme["bg"], layout="constrained")
        axes = fig.subplots(len(panels), 1, squeeze=False)[:, 0]
        for ax, key in zip(axes, panels):
            ax.set_facecolor(theme["bg"])
            draw[key](ax)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.text(0.0, 1.0, key, transform=ax.transAxes, ha="left", va="top", fontsize=style["font"] + 3,
                    fontweight="bold")
            if style["titles"][key]:
                ax.text(0.03, 1.0, style["titles"][key], transform=ax.transAxes, ha="left", va="top",
                        fontsize=style["font"] + 1)
        widths = [ax.get_xlim()[1] - ax.get_xlim()[0] for ax in axes]
        if style["same_scale"]:
            for ax in axes:
                ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[0] + max(widths))
        ratios = [(ax.get_ylim()[1] - ax.get_ylim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]) for ax in axes]
        axes[0].get_subplotspec().get_gridspec().set_height_ratios(ratios)
        fig.set_size_inches(style["width"], style["width"] * sum(ratios) + 0.1 * len(panels))
    return fig


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # NCA training schematic

    - a: one NCA update step; b: training on a time series, parallel in time
    - a: toy colony from `project_overview.py`; b: made-up values, not data
    """)
    return


@app.cell(hide_code=True)
def _():
    background = mo.ui.dropdown(options=list(ko_drawing.THEMES), value="light", label="Page background")
    figure_width = mo.ui.slider(start=5.0, stop=14.0, step=0.25, value=7.5, label="Width (in)", show_value=True)
    font_size = mo.ui.slider(start=5, stop=12, step=0.5, value=7, label="Font size", show_value=True)
    panels = mo.ui.multiselect(options=["a", "b"], value=["a", "b"], label="Panels")
    same_scale = mo.ui.switch(value=True, label="Same scale in every panel")
    card_size = mo.ui.slider(start=0.6, stop=1.6, step=0.05, value=1.2, label="Image size", show_value=True)
    model_colour = mo.ui.text(value="", placeholder="theme", label="Model colour")
    accent_colour = mo.ui.text(value="", placeholder="theme", label="Firing colour")
    mask_colour = mo.ui.text(value="#e6a700", label="Boundary colour")
    neighbourhood_colour = mo.ui.text(value="#00b300", label="Kernel colour")
    loss_colour = mo.ui.text(value="#d62728", label="Loss colour")
    regulariser_colour = mo.ui.text(value="#ff7f0e", label="Regulariser colour")
    mo.vstack([
        mo.md("**Display**"),
        mo.hstack([background, figure_width, font_size, card_size], justify="start"),
        mo.hstack([panels, same_scale], justify="start"),
        mo.hstack([model_colour, accent_colour, mask_colour], justify="start"),
        mo.hstack([neighbourhood_colour, loss_colour, regulariser_colour], justify="start"),
    ])
    return (
        accent_colour,
        background,
        card_size,
        figure_width,
        font_size,
        loss_colour,
        mask_colour,
        model_colour,
        neighbourhood_colour,
        panels,
        regulariser_colour,
        same_scale,
    )


@app.cell(hide_code=True)
def _():
    kernels = mo.ui.multiselect(options=list(KERNELS), value=["ID", "LAP", "DIFF"], label="Kernels")
    gated = mo.ui.switch(value=True, label="Gated (GLU)")
    firing = mo.ui.dropdown(options=["fixed", "learned"], value="fixed", label="Firing rate")
    fire_rate = mo.ui.slider(start=0.1, stop=1.0, step=0.05, value=0.5, label="Fire rate", show_value=True)
    boundary_mode = mo.ui.dropdown(options=["soft", "hard"], value="soft", label="Boundary")
    update_gain = mo.ui.slider(start=1.0, stop=10.0, step=0.5, value=4.0, label="Update brightness",
                               show_value=True)
    network_height = mo.ui.slider(start=1.2, stop=2.6, step=0.1, value=2.2, label="Network height",
                                  show_value=True)
    mo.vstack([
        mo.md("**Panel a** (architecture)"),
        mo.hstack([kernels, gated, firing, fire_rate, boundary_mode], justify="start"),
        mo.hstack([update_gain, network_height], justify="start"),
    ])
    return (
        boundary_mode,
        fire_rate,
        firing,
        gated,
        kernels,
        network_height,
        update_gain,
    )


@app.cell(hide_code=True)
def _():
    toy_size = mo.ui.number(start=6, stop=96, step=1, value=24, label="Pixels per side")
    toy_frames = mo.ui.slider(start=3, stop=7, step=1, value=5, label="Timepoints", show_value=True)
    toy_steps = mo.ui.slider(start=5, stop=120, step=5, value=40, label="Steps per frame", show_value=True)
    toy_seed = mo.ui.number(start=0, stop=1000, step=1, value=0, label="Seed")
    zoom_frame = mo.ui.slider(start=0, stop=5, step=1, value=2, label="Frame shown", show_value=True)
    mo.vstack([
        mo.md("**Toy colony** (panel a)"),
        mo.hstack([toy_size, toy_frames, toy_steps, toy_seed, zoom_frame], justify="start"),
    ])
    return toy_frames, toy_seed, toy_size, toy_steps, zoom_frame


@app.cell(hide_code=True)
def _():
    phase_width = mo.ui.slider(start=5.0, stop=12.0, step=0.25, value=10.0, label="Portrait width",
                               show_value=True)
    phase_height = mo.ui.slider(start=2.0, stop=5.0, step=0.1, value=3.0, label="Portrait height",
                                show_value=True)
    batch = mo.ui.slider(start=1, stop=8, step=1, value=8, label="Batch per timepoint", show_value=True)
    fibre_width = mo.ui.slider(start=0.2, stop=1.5, step=0.05, value=0.6, label="Line width (batch > 1)",
                               show_value=True)
    fibre_alpha = mo.ui.slider(start=0.1, stop=1.0, step=0.05, value=0.6, label="Line alpha (batch > 1)",
                               show_value=True)
    prediction_error = mo.ui.slider(start=0.0, stop=0.45, step=0.01, value=0.42, label="Prediction error",
                                    show_value=True)
    overshoot = mo.ui.slider(start=0.0, stop=0.8, step=0.02, value=0.49, label="Overshoot", show_value=True)
    overshoot_interval = mo.ui.slider(start=-1, stop=5, step=1, value=0, label="Overshooting interval (-1: none)",
                                      show_value=True)
    show_reinjection_paths = mo.ui.switch(value=True, label="Rollouts from reinjected data")
    plot_gap = mo.ui.slider(start=0.6, stop=2.5, step=0.05, value=1.3, label="Gap before plot", show_value=True)
    curve_count = mo.ui.slider(start=1, stop=3, step=1, value=3, label="Loss curves", show_value=True)
    show_reinjection = mo.ui.switch(value=True, label="Reinjection curve")
    mo.vstack([
        mo.md("**Panel b** (mock phase portrait)"),
        mo.hstack([phase_width, phase_height, batch, fibre_width, fibre_alpha], justify="start"),
        mo.hstack([prediction_error, overshoot, overshoot_interval, show_reinjection_paths], justify="start"),
        mo.hstack([plot_gap, curve_count, show_reinjection], justify="start"),
    ])
    return (
        batch,
        curve_count,
        fibre_alpha,
        fibre_width,
        overshoot,
        overshoot_interval,
        phase_height,
        phase_width,
        plot_gap,
        prediction_error,
        show_reinjection,
        show_reinjection_paths,
    )


@app.cell(hide_code=True)
def _():
    labels = mo.ui.dictionary({
        "title_a": mo.ui.text(value="Defining the NCA update", label="Title a"),
        "title_b": mo.ui.text(value="Training on time series data", label="Title b"),
        "observed_label": mo.ui.text(value="observed\\nchannels", label="Observed"),
        "hidden_label": mo.ui.text(value="hidden\\nchannels", label="Hidden"),
        "mask_label": mo.ui.text(value="mask\\nchannel", label="Mask"),
        "perception_label": mo.ui.text(value="perception\\nkernels", label="Perception"),
        "network_label": mo.ui.text(value="channel mixing\\nnetwork $f_\\theta$", label="Network"),
        "gate_label": mo.ui.text(value="gated\\nlinear unit", label="Gate"),
        "fire_label": mo.ui.text(value="random fire mask\\n$\\sigma \\sim$ Bernoulli$(p)$", label="Fire mask"),
        "fire_label_learned": mo.ui.text(value="learned fire mask\\n$\\sigma \\sim$ Bernoulli$(p_\\theta)$",
                                         label="Fire mask (learned)"),
        "skip_label": mo.ui.text(value="residual", label="Skip"),
        "boundary_label": mo.ui.text(value="boundary", label="Boundary"),
        "adhesion_label": mo.ui.text(value="micropattern\\nmask $m$", label="Adhesion mask"),
        "repeat_label": mo.ui.text(value="repeat for $K$ steps", label="Repeat"),
        "space_label": mo.ui.text(value="state space $\\mathcal{X}$", label="State space"),
        "time_label": mo.ui.text(value="time", label="Time"),
        "steps_label": mo.ui.text(value="$K$ NCA steps", label="Rollout"),
        "parallel_label": mo.ui.text(value="intervals rolled out in parallel, outputs feed forward at the next training iteration",
                                     label="Parallel"),
        "region_label": mo.ui.text(value="outside valid range: penalised by regulariser $R$", label="Region"),
        "data_key": mo.ui.text(value="data $d_i$", label="Key: data"),
        "prediction_key": mo.ui.text(value="prediction $\\hat{x}_i$ (pool)", label="Key: prediction"),
        "state_key": mo.ui.text(value="rollout, one per batch member", label="Key: rollout"),
        "penalised_key": mo.ui.text(value="penalised by regulariser", label="Key: penalised"),
        "reinjection_key": mo.ui.text(value="start from data (prob. $q$)", label="Key: reinjection"),
        "curve_title": mo.ui.text(value="loss curricula", label="Plot title"),
        "curve_x_label": mo.ui.text(value="training iteration", label="Plot x"),
        "curve_y_label": mo.ui.text(value="weight", label="Plot y"),
        "curve_labels": mo.ui.text(value="loss term 1, loss term 2, loss term 3", label="Curve labels"),
        "reinjection_label": mo.ui.text(value="reinjection $q$", label="Reinjection"),
        "equation": mo.ui.text(
            value=r"$\mathcal{L} = \sum_k w_k(n)\, \mathcal{L}_k(\hat{x}, d) + \sum_r \lambda_r \sum_s R_r(x_s)$",
            label="Equation", full_width=True),
    })
    mo.vstack([mo.md("**Labels** (`\\n` for a new line, `$...$` for maths)"),
               mo.hstack(list(labels.values()), wrap=True, justify="start")])
    return (labels,)


@app.cell(hide_code=True)
def _(toy_frames, toy_seed, toy_size, toy_steps):
    # The toy rule's length scales are in pixels, so run it at 24 px and resample to the chosen size
    _base = overview.toy_patterning(24, int(toy_frames.value), int(toy_steps.value), int(toy_seed.value))
    toy = resample_toy(_base, int(toy_size.value or 24))
    return (toy,)


@app.cell(hide_code=True)
def _(
    accent_colour,
    background,
    batch,
    boundary_mode,
    card_size,
    curve_count,
    fibre_alpha,
    fibre_width,
    figure_width,
    fire_rate,
    firing,
    font_size,
    gated,
    kernels,
    labels,
    loss_colour,
    mask_colour,
    model_colour,
    neighbourhood_colour,
    network_height,
    overshoot,
    overshoot_interval,
    panels,
    phase_height,
    phase_width,
    plot_gap,
    prediction_error,
    regulariser_colour,
    same_scale,
    show_reinjection,
    show_reinjection_paths,
    toy,
    toy_seed,
    update_gain,
    zoom_frame,
):
    mo.stop(not panels.value, mo.callout("Select at least one panel.", kind="warn"))
    _theme = dict(ko_drawing.THEMES[background.value])
    _theme["accent"] = colour_or(accent_colour.value, _theme["accent"])
    _text = {k: v.replace("\\n", "\n") for k, v in labels.value.items()}
    style = {
        "theme": _theme,
        "font": float(font_size.value),
        "width": float(figure_width.value),
        "same_scale": same_scale.value,
        "back_cards": "full",
        "card": float(card_size.value),
        "row_gap": 0.3,
        "model_colour": colour_or(model_colour.value, _theme["model"]),
        "mask_colour": colour_or(mask_colour.value, "#e6a700"),
        "neighbourhood_colour": colour_or(neighbourhood_colour.value, "#00b300"),
        "loss_colour": colour_or(loss_colour.value, "#d62728"),
        "regulariser_colour": colour_or(regulariser_colour.value, "#ff7f0e"),
        "kernels": [k for k in KERNELS if k in kernels.value],
        "gated": gated.value,
        "firing": firing.value,
        "fire_rate": float(fire_rate.value),
        "boundary_mode": boundary_mode.value,
        "update_gain": float(update_gain.value),
        "network_height": float(network_height.value),
        "zoom_frame": int(zoom_frame.value),
        "seed": int(toy_seed.value),
        "phase_width": float(phase_width.value),
        "phase_height": float(phase_height.value),
        "batch": int(batch.value),
        "fibre_width": float(fibre_width.value),
        "fibre_alpha": float(fibre_alpha.value),
        "prediction_error": float(prediction_error.value),
        "overshoot": float(overshoot.value),
        "overshoot_interval": int(overshoot_interval.value),
        "show_reinjection_paths": show_reinjection_paths.value,
        "plot_gap": float(plot_gap.value),
        "curve_count": int(curve_count.value),
        "show_reinjection": show_reinjection.value,
        "titles": {"a": _text["title_a"], "b": _text["title_b"]},
        **{k: v for k, v in _text.items() if not k.startswith("title_")},
    }
    schematic = schematic_figure(sorted(panels.value), toy, style)
    schematic
    return (schematic,)


@app.cell(hide_code=True)
def _():
    figure_name = mo.ui.text(value="nca_training_schematic", label="File name")
    save_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    save_figure = mo.ui.run_button(label="Save figure")
    mo.hstack([figure_name, save_svg, save_figure], justify="start")
    return figure_name, save_figure, save_svg


@app.cell
def _(figure_name, save_figure, save_svg, schematic):
    mo.stop(not save_figure.value)
    _directory = Path(mo.notebook_dir()).parents[1] / "figures" / "schematics"
    _directory.mkdir(parents=True, exist_ok=True)
    _suffixes = ("pdf", "png", "svg") if save_svg.value else ("pdf", "png")
    # svg.fonttype "none" keeps text as editable text in Inkscape
    with plt.rc_context({"svg.fonttype": "none"}):
        for _suffix in _suffixes:
            schematic.savefig(_directory / f"{figure_name.value}.{_suffix}", dpi=400,
                              facecolor=schematic.get_facecolor())
    mo.md(f"Saved `{figure_name.value}` as " + ", ".join(_suffixes) + f" to `{_directory}`")
    return


if __name__ == "__main__":
    app.run()

"""Render snowmelt NCA trajectories to browser-playable (H.264) mp4 files.

A video is a row (or two) of map panels. Each panel is either one channel
through a colormap, or up to three channels mixed into one colour image:

``rgb``  channels drive red, green and blue (e.g. B4, B3, B2 for true colour).
``cmy``  channels drive cyan, magenta and yellow on a white background, so
         where none is present the map is white and overlaps mix like inks.

Frames are drawn with matplotlib (titles, colourbars and km axes like the
notebook figures) and written with ``imageio-ffmpeg``, which ships its own
ffmpeg binary.
"""

from dataclasses import dataclass

import numpy as np

COMPOSITE_MODES = ("rgb", "cmy")
BACKGROUND = "#e6e6e6"


@dataclass(frozen=True)
class Panel:
    """One map in the video.

    ``frames`` is ``[F, H, W]`` for a colormap panel (drawn with ``cmap``,
    ``vmin`` and ``vmax``) or ``[F, H, W, 3]`` for a composite already in
    ``[0, 1]``. Pixels outside ``mask`` are drawn as background.
    """

    title: str
    frames: np.ndarray
    mask: np.ndarray
    cmap: str | None = None
    vmin: float = 0.0
    vmax: float = 1.0
    units: str = ""


def stretch(values, low, high, gamma=1.0):
    """Linear stretch to ``[0, 1]`` with an optional gamma."""
    return np.clip((values - low) / (high - low + 1e-12), 0.0, 1.0) ** (1.0 / gamma)


def composite(channels, mode):
    """Mix up to three stretched ``[..., H, W]`` arrays (``None`` = absent) into ``[..., H, W, 3]``."""
    if mode not in COMPOSITE_MODES:
        raise ValueError(f"mode must be one of {COMPOSITE_MODES}, got {mode!r}")
    shape = next(np.shape(c) for c in channels if c is not None)
    a, b, c = (np.zeros(shape, np.float32) if value is None else value for value in channels)
    if mode == "rgb":
        return np.stack([a, b, c], axis=-1)
    # Subtractive: cyan removes red, magenta removes green, yellow removes blue.
    return np.stack([1.0 - a, 1.0 - b, 1.0 - c], axis=-1)


def _rgba(frame, mask):
    """Composite frame with transparent pixels outside the mask."""
    return np.concatenate([frame, mask[..., None].astype(frame.dtype)], axis=-1)


def render_mp4(path, panels, frame_titles, extent, fps=24, ncols=None, draw_outline=None, progress=None):
    """Draw ``panels`` frame by frame and write them to ``path`` as H.264 mp4.

    Panels fill a grid of ``ncols`` columns row by row; a ``None`` panel leaves
    that slot empty.
    ``frame_titles[i]`` is the figure title of frame ``i``; ``extent`` is the
    imshow extent (km) shared by all panels. ``draw_outline(ax, mask, extent)``
    adds the catchment outline, and ``progress`` wraps the frame iterator
    (e.g. ``mo.status.progress_bar``).
    """
    import imageio_ffmpeg
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    ncols = ncols or len(panels)
    nrows = -(-len(panels) // ncols)
    # A standalone off-screen figure, so the notebook's pyplot state is untouched
    fig = Figure(figsize=(4.4 * ncols, 4.0 * nrows + 0.4), dpi=100, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=True)
    images = []
    for ax, panel in zip(axes.ravel(), panels):
        if panel is None:
            ax.axis("off")
            images.append(None)
            continue
        ax.set_facecolor(BACKGROUND)
        if panel.frames.ndim == 3:
            first = np.where(panel.mask, panel.frames[0], np.nan)
            image = ax.imshow(first, cmap=panel.cmap, vmin=panel.vmin, vmax=panel.vmax,
                              extent=extent, interpolation="nearest")
            fig.colorbar(image, ax=ax, shrink=0.8, label=panel.units)
        else:
            image = ax.imshow(_rgba(panel.frames[0], panel.mask), extent=extent, interpolation="nearest")
        if draw_outline is not None:
            draw_outline(ax, panel.mask, extent)
        ax.set_title(panel.title, fontsize=10)
        ax.tick_params(labelsize=8)
        images.append(image)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    title = fig.suptitle(frame_titles[0], fontsize=11)

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height(physical=True)
    width, height = width - width % 2, height - height % 2  # yuv420p needs even sizes
    writer = imageio_ffmpeg.write_frames(
        str(path), (width, height), fps=fps, codec="libx264", quality=8, macro_block_size=1,
    )
    writer.send(None)
    try:
        indices = range(len(frame_titles))
        for index in progress(indices) if progress is not None else indices:
            for image, panel in zip(images, panels):
                if panel is None:
                    continue
                if panel.frames.ndim == 3:
                    image.set_data(np.where(panel.mask, panel.frames[index], np.nan))
                else:
                    image.set_data(_rgba(panel.frames[index], panel.mask))
            title.set_text(frame_titles[index])
            fig.canvas.draw()
            pixels = np.asarray(fig.canvas.buffer_rgba())[:height, :width, :3]
            writer.send(np.ascontiguousarray(pixels))
    finally:
        writer.close()
    return path


__all__ = ["BACKGROUND", "COMPOSITE_MODES", "Panel", "composite", "render_mp4", "stretch"]

"""Shared style of the micropattern paper figures.

Change the look of every figure here. Notebooks import these values instead of
defining their own colours, fonts or sizes:

    from Experiments.micropatterns.figures.style import paper_style, new_figure, save_figure

    with paper_style():
        fig = new_figure("single", aspect=0.8)
        ...
    save_figure(fig, "my_figure", folder="results")

- Sizes follow a journal page: 89 mm single and 183 mm double column, 7 pt text.
- ``paper.mplstyle`` holds the matplotlib settings; this file holds palettes, labels and helpers.
- Line colours were checked for colour-blind separation (dataviz palette validator).
"""

from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like, to_rgb
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Circle, Patch, Wedge

STYLE_FILE = Path(__file__).with_name("paper.mplstyle")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FIGURE_ROOT = REPOSITORY_ROOT / "figures"

# --- Page and text ---------------------------------------------------------

MM = 1 / 25.4
WIDTH = {"single": 89 * MM, "double": 183 * MM, "max_height": 247 * MM}

# Font sizes (pt) of each text role
FONT = {
    "letter": 8,  # panel letter, bold
    "title": 7,  # panel title, bold
    "subtitle": 7,  # after the title, regular, muted
    "label": 7,  # axis labels, row and block labels
    "tick": 6,
    "legend": 6,
    "annotation": 6,  # small notes inside panels, scale bars
    "min": 5,  # nothing smaller than this
}

# Slide-sized variant (for talk figures only)
TALK_FONT_SCALE = 11 / 7
TALK_WIDTH = 13.33

# Line widths (pt)
LINE = {"hairline": 0.3, "thin": 0.5, "axis": 0.6, "data": 1.0, "emphasis": 1.5}
BAND_ALPHA = 0.2
GROUP_LINESTYLES = ("-", "--", ":", "-.")

# --- Colours of the page, lines and schematics -----------------------------

INK = "#222222"  # text, axes, and measured data
MUTED = "#9a9a9a"  # secondary text, outlines, reference lines
PAGE = "white"
IMAGE_BG = "black"  # behind fluorescence images

DATA = INK
MODEL = "#0072B2"
ACCENT = "#D55E00"  # highlight: knockout, slice, probe, loss
NEIGHBOURHOOD = "#009E73"
UNMEASURED = {"face": "#d9d9d9", "edge": "#8c8c8c", "hatch": "////"}

# The colours above as one dict, for schematic code that passes them around together
THEME = {"bg": PAGE, "fg": INK, "muted": MUTED, "accent": ACCENT, "model": MODEL}

# Fixed order for any other set of lines (ablations, curricula, ...). Same as the mplstyle cycle.
SERIES = ("#0072B2", "#D55E00", "#009E73", "#C99A00", "#8E5BA8")
BASELINE = MUTED
# One colour per experiment group, where groups are told apart by colour (not by sector)
GROUP_COLOURS = {"cell_fate_s1": "#009E73", "cell_fate_s2": "#56B4E9", "rna_expression": "#C99A00",
                 "protein_response": "#8E5BA8"}
# Outlines of zoom boxes, visible on black colonies and on a white page
ZOOM_COLOURS = ("#D55E00", "#009E73", "#56B4E9", "#C99A00")

# Mesoderm (dark red) and endoderm (gold) differ in hue and lightness, so they stay
# apart with red-green colour blindness too
CELL_TYPE_COLOURS = {
    "Mesoderm": "#B2182B",
    "Notochord": "#009E73",
    "Endoderm": "#E6AB02",
    "Several": "#8E5BA8",
    "Other": MUTED,
}
UNKNOWN_CELL_TYPE = "#000000"

CMAP = {"intensity": "magma", "heatmap": "viridis", "diverging": "RdBu_r", "mask": "gray"}
BAD_COLOUR = "#d9d9d9"  # set_bad colour of masked image pixels

# --- Fluorescence channels --------------------------------------------------

# Channel colours, added like light on a black background
CHANNEL_COLOURS = {
    "off": None,
    "cyan": (0.0, 0.78, 0.9),
    "magenta": (0.92, 0.12, 0.72),
    "yellow": (1.0, 0.78, 0.0),
    "red": (0.9, 0.15, 0.1),
    "green": (0.15, 0.8, 0.2),
    "blue": (0.15, 0.35, 1.0),
    "grey": (0.6, 0.6, 0.6),
    "white": (1.0, 1.0, 1.0),
}
MARKER_COLOURS = {
    "SOX2": "cyan", "SOX17": "yellow", "TBXT": "magenta", "FOXA2": "cyan", "LMBR": "off",
    "CER1": "yellow", "LEFTY": "cyan", "NODAL": "magenta", "LEF1": "green", "SMAD23": "magenta",
}

# Channels shown in figures for each experiment group (others, such as LMBR, are left out)
FIGURE_CHANNELS = {
    "cell_fate_s1": ("SOX2", "TBXT", "SOX17"),
    "cell_fate_s2": ("FOXA2", "TBXT", "SOX17"),
    "rna_expression": ("CER1", "LEFTY", "NODAL"),
    "protein_response": ("LEF1", "SMAD23"),
}

# Cyan, magenta, yellow channels of each panel of the three-panel CMY composites
# (rendering.render_cmy_composite). These do not follow MARKER_COLOURS.
CMY_TRIPLETS = (
    ("SOX2", "TBXT", "SOX17"),
    ("CER1", "LEFTY", "NODAL"),
    ("FOXA2", "LEF1", "SMAD23"),
)

# --- Labels ------------------------------------------------------------------

EXPERIMENT_GROUPS = ("cell_fate_s1", "cell_fate_s2", "rna_expression", "protein_response")
GROUP_LABELS = {"cell_fate_s1": "Fate S1", "cell_fate_s2": "Fate S2", "rna_expression": "RNA",
                "protein_response": "Protein"}
GROUP_LABELS_LONG = {
    "cell_fate_s1": "Cell fate markers\n(staining set 1)",
    "cell_fate_s2": "Cell fate markers\n(staining set 2)",
    "rna_expression": "RNA expression",
    "protein_response": "Protein response",
}
CONDITION_LABELS = {"ctrl": "Control", "sl24": "NODAL KO from 24 h", "sl0": "NODAL KO from 0 h"}
# Telling conditions apart in line plots
CONDITION_COLOURS = {"ctrl": SERIES[0], "sl0": SERIES[1], "sl24": SERIES[2]}
# Row labels of measured data and model output
SOURCE_LABELS = {"target": "True data", "prediction": "NCA"}
SOURCE_COLOURS = {"target": DATA, "prediction": MODEL}
# Marker names of the channels of a raw stack file, in file order (see Common/dataloader/micropattern_schemas.py)
STAIN_NAMES = {
    "cell fate s1": ("SOX17", "SOX2", "TBXT", "LMBR"),
    "cell fate s2": ("SOX17", "FOXA2", "TBXT", "LMBR"),
}
# Telling projections apart in line plots (e.g. marker edges of measured data)
PROJECTION_COLOURS = {"max": INK, "sum": SERIES[4], "slice": SERIES[0]}
PROJECTION_LABELS = {
    "slice": ("Slice", r"$I(x, y, z_0)$"),
    "max": ("Max projection", r"$\max_z\, I(x, y, z)$"),
    "sum": ("Sum projection", r"$\sum_z\, I(x, y, z)$"),
}


# --- Helpers -------------------------------------------------------------------


# rc keys scaled by the talk preset, with their paper sizes
FONT_SIZE_RC = {
    "font.size": FONT["label"], "axes.titlesize": FONT["title"], "axes.labelsize": FONT["label"],
    "xtick.labelsize": FONT["tick"], "ytick.labelsize": FONT["tick"], "legend.fontsize": FONT["legend"],
}


@contextmanager
def paper_style(preset="paper"):
    """Matplotlib settings of the paper figures inside a ``with`` block.

    ``preset="talk"`` scales every font size for slide-sized figures.
    """
    rc = {}
    if preset == "talk":
        rc = {key: FONT_SIZE_RC[key] * TALK_FONT_SCALE for key in FONT_SIZE_RC}
    elif preset != "paper":
        raise ValueError(f"Unknown preset {preset!r}; use 'paper' or 'talk'")
    with plt.style.context(["default", STYLE_FILE]), plt.rc_context(rc):
        yield



def figure_width(width):
    """Width in inches of "single", "double" or a number of inches."""
    return WIDTH[width] if isinstance(width, str) else float(width)


def new_figure(width="double", height=None, aspect=None, **kwargs):
    """Figure ``width`` wide ("single", "double" or inches), ``height`` inches or ``aspect`` × width tall."""
    w = figure_width(width)
    if height is None:
        height = w * (0.6 if aspect is None else aspect)
    return plt.figure(figsize=(w, height), **kwargs)


def label_panel(ax, letter=None, title=None, subtitle=None, offset=(0, 4)):
    """Panel letter (bold), title (bold) and subtitle (muted) above the top-left corner of ``ax``."""
    anchor, x = None, offset[0]
    if letter:
        anchor = ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(x, offset[1]),
                             textcoords="offset points", ha="left", va="bottom", fontsize=FONT["letter"],
                             fontweight="bold", color=INK, annotation_clip=False)
    if title:
        xy, xycoords, xytext = ((1, 0), anchor, (4, 0)) if anchor else ((0, 1), "axes fraction", offset)
        anchor = ax.annotate(title, xy=xy, xycoords=xycoords, xytext=xytext, textcoords="offset points",
                             ha="left", va="bottom", fontsize=FONT["title"], fontweight="bold", color=INK,
                             annotation_clip=False)
    if subtitle:
        xy, xycoords, xytext = ((1, 0), anchor, (4, 0)) if anchor else ((0, 1), "axes fraction", offset)
        ax.annotate(subtitle, xy=xy, xycoords=xycoords, xytext=xytext, textcoords="offset points",
                    ha="left", va="bottom", fontsize=FONT["subtitle"], color=MUTED, annotation_clip=False)


def save_figure(fig, name, folder, formats=("pdf", "png"), svg=False, root=None):
    """Write ``fig`` to ``figures/<folder>/<name>.<format>`` and return the paths.

    Text stays text in svg files, and fonts are embedded as TrueType in pdf files.
    """
    directory = Path(root or FIGURE_ROOT) / folder
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        for extension in (*formats, *(("svg",) if svg else ())):
            path = directory / f"{name}.{extension}"
            fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
            paths.append(path)
    return paths


def colour_or(text, default):
    """``text`` if it is a matplotlib colour, else ``default``."""
    return text.strip() if is_color_like(text.strip()) else default


def marker_colour(name):
    """RGB (or None) of a channel ``group/MARKER`` or ``MARKER`` from MARKER_COLOURS."""
    return CHANNEL_COLOURS[MARKER_COLOURS.get(name.rsplit("/", 1)[-1], "grey")]


def group_colours(names, group):
    """Colour of each channel in ``names`` (``group/MARKER``) when showing ``group``.

    Channels of other groups, and channels not in FIGURE_CHANNELS, get None.
    """
    shown = FIGURE_CHANNELS.get(group)
    colours = []
    for name in names:
        name_group, marker = name.split("/")[0], name.rsplit("/", 1)[-1]
        visible = name_group == group and (shown is None or marker in shown)
        colours.append(marker_colour(name) if visible else None)
    return colours


def cell_type_colour(label):
    """Hex colour of a cell type label (black for labels without one)."""
    return CELL_TYPE_COLOURS.get(label, UNKNOWN_CELL_TYPE)


def cell_type_rgb(label):
    """RGB tuple of a cell type label."""
    return to_rgb(cell_type_colour(label))


# --- Legends -------------------------------------------------------------------


def sector_angles(index, count):
    """Start and end angle (degrees, anticlockwise from +x) of sector ``index`` of ``count``.

    The first sector starts at the top, so two sectors are the left and right halves.
    """
    start = 90.0 + index * 360.0 / count
    return start, start + 360.0 / count


class SectorHandle:
    """Legend handle for sector ``index`` of ``count`` (see sector_angles)."""

    def __init__(self, index, count):
        self.index, self.count = index, count


class SectorHandler(HandlerBase):
    """Draws a SectorHandle as a small disc with its sector filled."""

    def __init__(self, fill="#dddddd", edge="#999999", **kwargs):
        super().__init__(**kwargs)
        self.fill, self.edge = fill, edge

    def create_artists(self, legend, handle, xdescent, ydescent, width, height, fontsize, trans):
        centre = (width / 2 - xdescent, height / 2 - ydescent)
        radius = 0.75 * height
        start, stop = sector_angles(handle.index, handle.count)
        return [
            Circle(centre, radius, fill=False, edgecolor=self.edge, lw=LINE["axis"], transform=trans),
            Wedge(centre, radius, start, stop, facecolor=self.fill, edgecolor="none", transform=trans),
        ]


def legend_swatch(colour, kind="white"):
    """Legend handle of a channel colour: a boxed patch on a white legend, a line on a black one."""
    if kind == "white":
        return Patch(facecolor=colour, edgecolor=INK, lw=LINE["axis"])
    return plt.Line2D([], [], color=colour, lw=2)


def unmeasured_patch():
    """Legend handle of the hatched "not measured" fill."""
    return Patch(facecolor=UNMEASURED["face"], edgecolor=UNMEASURED["edge"], hatch=UNMEASURED["hatch"], lw=0)


def legend_options(kind="white", boxed=False):
    """Keyword arguments of ``legend``/``figure.legend`` for a white or black (image-like) legend.

    ``boxed`` draws a thin grey frame around a white legend (for legends floating over a figure).
    """
    white = kind == "white"
    return {
        "fontsize": FONT["legend"],
        "labelcolor": INK if white else "white",
        "frameon": boxed or not white,
        "fancybox": False,
        "facecolor": "white" if white else "black",
        "edgecolor": (MUTED if boxed else "white") if white else "black",
        "framealpha": 1.0,
        "handlelength": 1.2,
        "columnspacing": 0.8,
        "handler_map": {SectorHandle: (SectorHandler(fill="#666666", edge=INK) if white else SectorHandler())},
    }

"""Procedural RGBA tasks that need information to travel faster than a local NCA can carry it.

Every instance has tissue (alpha = 1) from the start, so nothing needs to
grow, and random cues, so the answer can't be memorised. A local NCA moves
information at most one cell per step, so with distances larger than run.t
it can only solve a task by using a signalling mechanism
(NCA/model/NCA_signalling.py).

- remote_switch: a bar with a red or blue cue at one end; the whole bar takes
  the cue colour. 1 bit over a distance.
- majority_vote: red and blue cues scattered over a disc; the disc takes the
  majority colour. Needs a global average.
- french_flag: a bar of random length with a red and a blue end marker; the
  bar splits into three equal bands. Needs relative position.
- mirror_copy: random dots on one side of a band are copied to their mirror
  image on the other side. Many bits, point to point.
- winding_relay: a cue at one end of a long zigzag path; the whole path takes
  the cue colour. The signal must follow the path without fading.
- connected_routing: two interleaved combs, a cue on one of them; only the
  comb with the cue takes its colour. Signals must follow the tissue, not
  straight-line distance.

remote_switch and winding_relay can also have a timed target (``front_step``):
the colour moves along the tissue by ``front_step`` cells per observed frame.

Each task function draws one instance from a numpy Generator and returns an
``Instance``. ``sample_batch`` stacks instances into ``[count, time, 4, H, W]``
trajectories. This module doesn't import JAX, so the config can check task
names and parameters.
"""

import inspect
from typing import NamedTuple

import numpy as np


TISSUE = (0.5, 0.5, 0.5, 1.0)
RED = (1.0, 0.15, 0.15, 1.0)
BLUE = (0.15, 0.15, 1.0, 1.0)
WHITE = (0.95, 0.95, 0.95, 1.0)
# Cells of a cue at the end of a bar or path
CUE_LENGTH = 3


class Instance(NamedTuple):
    # [initial, target 1, ..., target k], each [4, H, W]
    frames: list
    # How far the information has to travel (task specific, in cells)
    difficulty: float
    # [H, W] cells whose arrival time is measured (e.g. the far end of a bar)
    probe: np.ndarray


def paint(size, layers):
    """RGBA image [4, size, size] from (mask, colour) layers; later layers paint over earlier ones."""
    image = np.zeros((4, size, size), dtype=np.float32)
    for mask, colour in layers:
        image[:, mask] = np.asarray(colour, dtype=np.float32)[:, None]
    return image


def geodesic_distance(mask, sources):
    """Steps from ``sources`` to each cell of ``mask`` moving between 4-neighbours inside ``mask`` (inf if unreachable)."""
    distance = np.full(mask.shape, np.inf)
    reached = sources & mask
    step = 0
    while reached.any():
        distance[reached] = step
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        reached = grown & mask & np.isinf(distance)
        step += 1
    return distance


def _front_frames(initial, final, distance, front_step, frames):
    """Targets where the final colours have reached cells within ``front_step * k`` of the cue at frame k."""
    if front_step is None:
        return [final]
    return [np.where(distance <= front_step * k, final, initial) for k in range(1, frames + 1)]


def _far_end(tissue, distance):
    reachable = tissue & np.isfinite(distance)
    return reachable & (distance >= distance[reachable].max() - 2)


def _orient(rng, instance):
    """Rotate and flip an instance at random, so cues can come from any side."""
    turns, flip = rng.integers(4), rng.integers(2)

    def move(a):
        a = np.rot90(a, turns, axes=(-2, -1))
        return np.ascontiguousarray(np.flip(a, axis=-1) if flip else a)

    return Instance([move(f) for f in instance.frames], instance.difficulty, move(instance.probe))


def _cue_colour(rng):
    return RED if rng.integers(2) else BLUE


def _bar(size, row, column, length, width):
    mask = np.zeros((size, size), dtype=bool)
    mask[row:row + width, column:column + length] = True
    return mask


def remote_switch(rng, size, length=(10, 56), width=3, front_step=None, frames=1):
    """A bar of random length; a cue at one end sets the colour of the whole bar."""
    length = int(min(rng.integers(length[0], length[1] + 1), size - 4))
    row = rng.integers(2, size - width - 1)
    column = rng.integers(2, size - length - 1)
    tissue = _bar(size, row, column, length, width)
    cue = _bar(size, row, column, CUE_LENGTH, width)
    colour = _cue_colour(rng)
    initial = paint(size, [(tissue, TISSUE), (cue, colour)])
    final = paint(size, [(tissue, colour)])
    distance = geodesic_distance(tissue, cue)
    targets = _front_frames(initial, final, distance, front_step, frames)
    return _orient(rng, Instance([initial] + targets, float(length), _far_end(tissue, distance)))


def majority_vote(rng, size, radius=(12, 28), cues=(5, 15)):
    """A disc with scattered red and blue 2x2 cues; the rest of the disc takes the majority colour."""
    radius = int(min(rng.integers(radius[0], radius[1] + 1), size // 2 - 2))
    rows, columns = np.mgrid[:size, :size]
    centre = (size - 1) / 2
    tissue = (rows - centre) ** 2 + (columns - centre) ** 2 <= radius**2
    # Top-left corners of 2x2 cues inside the disc, on a grid of pitch 3 so cues never touch
    fits = np.zeros_like(tissue)
    fits[:-1, :-1] = tissue[:-1, :-1] & tissue[1:, 1:]
    fits[1::3, :] = fits[2::3, :] = False
    fits[:, 1::3] = fits[:, 2::3] = False
    corners = np.argwhere(fits)
    count = int(rng.integers(cues[0], cues[1] + 1)) | 1  # odd, so there is no tie
    count = min(count, len(corners) if len(corners) % 2 else len(corners) - 1)
    chosen = corners[rng.choice(len(corners), count, replace=False)]
    red_count = int(rng.integers(count + 1))
    red, blue = np.zeros_like(tissue), np.zeros_like(tissue)
    for index, (row, column) in enumerate(chosen):
        (red if index < red_count else blue)[row:row + 2, column:column + 2] = True
    majority = RED if 2 * red_count > count else BLUE
    initial = paint(size, [(tissue, TISSUE), (red, RED), (blue, BLUE)])
    final = paint(size, [(tissue, majority), (red, RED), (blue, BLUE)])
    return _orient(rng, Instance([initial, final], float(radius), tissue & ~red & ~blue))


def french_flag(rng, size, length=(24, 56), width=4):
    """A bar with a red marker at one end and a blue one at the other splits into red, white and blue thirds."""
    length = int(min(rng.integers(length[0], length[1] + 1), size - 4))
    row = rng.integers(2, size - width - 1)
    column = rng.integers(2, size - length - 1)
    tissue = _bar(size, row, column, length, width)
    left = _bar(size, row, column, CUE_LENGTH, width)
    right = _bar(size, row, column + length - CUE_LENGTH, CUE_LENGTH, width)
    band = np.zeros((size, size), dtype=int)
    band[:, column:column + length] = (3 * (np.arange(length) + 0.5) / length).astype(int)
    initial = paint(size, [(tissue, TISSUE), (left, RED), (right, BLUE)])
    final = paint(size, [(tissue & (band == b), colour) for b, colour in enumerate((RED, WHITE, BLUE))])
    # Probe: the two band boundaries, the cells furthest from both markers
    boundary = tissue & (np.abs(np.diff(band, axis=1, prepend=band[:, :1])) > 0)
    boundary |= np.roll(boundary, -1, axis=1)
    return _orient(rng, Instance([initial, final], float(length), boundary & tissue))


def mirror_copy(rng, size, dots=(2, 6), height=(8, 16)):
    """Random red dots on one half of a band are copied to their mirror image in the other half."""
    height = int(rng.integers(height[0], height[1] + 1))
    row = rng.integers(2, size - height - 1)
    tissue = _bar(size, row, 2, size - 4, height)
    centre = size // 2
    count = int(rng.integers(dots[0], dots[1] + 1))
    source, copy = np.zeros_like(tissue), np.zeros_like(tissue)
    distances = []
    for _ in range(count):
        r = rng.integers(row, row + height - 1)
        c = rng.integers(4, centre - 6)
        source[r:r + 2, c:c + 2] = True
        copy[r:r + 2, 2 * centre - c - 2:2 * centre - c] = True
        distances.append(2 * (centre - c) - 2)
    initial = paint(size, [(tissue, TISSUE), (source, RED)])
    final = paint(size, [(tissue, TISSUE), (source, RED), (copy, RED)])
    right = tissue.copy()
    right[:, :centre] = False
    return _orient(rng, Instance([initial, final], float(np.mean(distances)), right))


def zigzag(size, rng, turns, width, gap):
    """Mask of a random zigzag path and the cue cells at its start."""
    pitch = width + gap
    rows = turns + 1
    top = rng.integers(2, size - rows * pitch + gap - 1)
    left = rng.integers(2, size // 3)
    right = rng.integers(2 * size // 3, size - width - 2)
    path = np.zeros((size, size), dtype=bool)
    for index in range(rows):
        row = top + index * pitch
        path[row:row + width, left:right + width] = True
        if index < turns:
            column = right if index % 2 == 0 else left
            path[row:row + pitch + width, column:column + width] = True
    cue = _bar(size, top, left, CUE_LENGTH, width)
    return path, cue


def winding_relay(rng, size, turns=(2, 6), width=3, gap=3, front_step=None, frames=1):
    """A cue at the start of a long zigzag path sets the colour of the whole path."""
    turns = int(min(rng.integers(turns[0], turns[1] + 1), (size - 4) // (width + gap) - 1))
    path, cue = zigzag(size, rng, turns, width, gap)
    colour = _cue_colour(rng)
    initial = paint(size, [(path, TISSUE), (cue, colour)])
    final = paint(size, [(path, colour)])
    distance = geodesic_distance(path, cue)
    targets = _front_frames(initial, final, distance, front_step, frames)
    length = float(distance[np.isfinite(distance)].max())
    return _orient(rng, Instance([initial] + targets, length, _far_end(path, distance)))


def combs(size, rng, fingers, width, gap):
    """Masks of two interleaved combs that never touch (fingers of one lie between fingers of the
    other), and the (row, column) of the top of each spine."""
    pitch = 2 * (width + gap)
    height = fingers * pitch
    top = rng.integers(2, size - height - 1)
    left = rng.integers(2, size // 4)
    right = rng.integers(3 * size // 4 - width, size - width - 2)
    first, second = np.zeros((size, size), dtype=bool), np.zeros((size, size), dtype=bool)
    first[top:top + height, left:left + width] = True
    second[top:top + height, right:right + width] = True
    for index in range(fingers):
        row = top + index * pitch
        first[row:row + width, left:right - gap] = True
        row += width + gap
        second[row:row + width, left + width + gap:right + width] = True
    return first, second, (top, left), (top, right)


def connected_routing(rng, size, fingers=(3, 5), width=2, gap=2):
    """Two interleaved combs, a cue on one of them; only the comb with the cue takes the cue colour."""
    fingers = int(min(rng.integers(fingers[0], fingers[1] + 1), (size - 4) // (2 * (width + gap))))
    first, second, first_spine, second_spine = combs(size, rng, fingers, width, gap)
    if rng.integers(2):
        first, second, first_spine = second, first, second_spine
    # The cue sits at the top of the spine of the first comb
    cue = _bar(size, first_spine[0], first_spine[1], width, CUE_LENGTH)
    colour = _cue_colour(rng)
    initial = paint(size, [(first | second, TISSUE), (cue, colour)])
    final = paint(size, [(second, TISSUE), (first, colour)])
    distance = geodesic_distance(first, cue)
    length = float(distance[np.isfinite(distance)].max())
    return _orient(rng, Instance([initial, final], length, _far_end(first, distance)))


TASKS = {
    "remote_switch": remote_switch,
    "majority_vote": majority_vote,
    "french_flag": french_flag,
    "mirror_copy": mirror_copy,
    "winding_relay": winding_relay,
    "connected_routing": connected_routing,
}


def task_parameters(task):
    """Names of the settings a task takes (beyond ``rng`` and ``size``)."""
    return tuple(inspect.signature(TASKS[task]).parameters)[2:]


def sample_batch(task, seed, count, size=64, target_repeats=2, **params):
    """``count`` random instances of ``task``.

    Returns the trajectories [count, time, 4, size, size] (the last target is
    repeated ``target_repeats`` times, to train persistence), the difficulty
    [count] and the probe masks [count, size, size].
    """
    rng = np.random.default_rng(seed)
    instances = [TASKS[task](rng, size, **params) for _ in range(count)]
    data = np.stack([
        np.stack(i.frames + [i.frames[-1]] * (target_repeats - 1)) for i in instances
    ])
    difficulty = np.array([i.difficulty for i in instances], dtype=np.float32)
    probe = np.stack([i.probe for i in instances])
    return data.astype(np.float32), difficulty, probe

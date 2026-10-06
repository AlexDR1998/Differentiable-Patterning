"""A local disk cache for slow, deterministic data pre-processing.

Each set of settings gets its own folder in the cache directory:

    <cache_dir>/<name>--<key>/
        settings.yaml      the settings and when the folder was made
        <item>.npy         one array per cached item

The key is a hash of the settings and of the source code that does the
processing, so changing either starts a new folder instead of reusing stale
results. Old folders are never removed automatically; ``list_cache_folders``
shows them with their size, and they can be deleted by hand.

The cache is only meant for local notebooks and checks. Set ``DATA_CACHE_DIR``
(e.g. in ``.env``) to turn it on where a loader supports it.
"""

import hashlib
import inspect
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

CACHE_DIR_VARIABLE = "DATA_CACHE_DIR"


def cache_dir_from_environment():
    """``DATA_CACHE_DIR``, or None when it is unset or empty (no caching)."""
    return os.environ.get(CACHE_DIR_VARIABLE) or None


def _plain(value):
    """``value`` as plain YAML/JSON types (dicts, lists, numbers, strings)."""
    if isinstance(value, dict):
        return {key if isinstance(key, (str, int)) else str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def cache_key(settings, sources=()):
    """Short hash of ``settings`` and the source code of ``sources``.

    ``sources`` are functions or modules whose code produces the cached
    data; editing any of them changes the key.
    """
    digest = hashlib.sha256(json.dumps(_plain(settings), sort_keys=True).encode())
    for source in sources:
        digest.update(inspect.getsource(source).encode())
    return digest.hexdigest()[:12]


def cache_folder(cache_dir, name, settings, sources=()):
    """The folder of ``cache_dir`` for ``settings``, made with its settings.yaml if new."""
    folder = Path(cache_dir).expanduser() / f"{name}--{cache_key(settings, sources)}"
    settings_file = folder / "settings.yaml"
    if not settings_file.exists():
        folder.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(
            {"name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "settings": _plain(settings)},
            sort_keys=False,
        )
        _write_atomically(settings_file, lambda handle: handle.write(text.encode()))
    return folder


def _write_atomically(path, write):
    """Write ``path`` through a temporary file, so readers never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as file:
            write(file)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def cached_array(folder, item, compute):
    """The array saved as ``item`` in ``folder``, or ``compute()`` saved there first.

    ``item`` may contain slashes (e.g. an image path relative to the dataset
    root); it is stored at ``folder/<item>.npy``.
    """
    path = Path(folder) / f"{item}.npy"
    if path.exists():
        return np.load(path, allow_pickle=False)
    array = np.asarray(compute())
    _write_atomically(path, lambda file: np.save(file, array, allow_pickle=False))
    return array


def list_cache_folders(cache_dir):
    """One row per cache folder: its name, settings, number of items and size."""
    rows = []
    for settings_file in sorted(Path(cache_dir).expanduser().glob("*/settings.yaml")):
        folder = settings_file.parent
        items = list(folder.rglob("*.npy"))
        description = yaml.safe_load(settings_file.read_text()) or {}
        rows.append(
            {
                "folder": folder.name,
                "created": description.get("created", ""),
                "items": len(items),
                "size (MB)": round(sum(item.stat().st_size for item in items) / 1e6, 1),
                "settings": description.get("settings", {}),
            }
        )
    return rows

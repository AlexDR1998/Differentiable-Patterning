import numpy as np
import yaml

from Common.dataloader.disk_cache import cache_folder, cache_key, cached_array, list_cache_folders


def _double(x):
    return 2 * x


def _triple(x):
    return 3 * x


def test_key_depends_on_settings_and_code():
    assert cache_key({"a": 1}) == cache_key({"a": 1})
    assert cache_key({"a": 1}) != cache_key({"a": 2})
    assert cache_key({"a": 1}, [_double]) != cache_key({"a": 1}, [_triple])


def test_cached_array_is_computed_once(tmp_path):
    folder = cache_folder(tmp_path, "test", {"factor": 2, "by_hour": {0: 1.5}})
    settings = yaml.safe_load((folder / "settings.yaml").read_text())
    assert settings["settings"] == {"factor": 2, "by_hour": {0: 1.5}}
    calls = []

    def compute():
        calls.append(1)
        return np.arange(6, dtype=np.float32).reshape(2, 3)

    first = cached_array(folder, "group/image.ome.tif", compute)
    second = cached_array(folder, "group/image.ome.tif", compute)
    assert len(calls) == 1
    assert np.array_equal(first, second) and second.dtype == np.float32
    [row] = list_cache_folders(tmp_path)
    assert row["folder"] == folder.name and row["items"] == 1
    assert not list(folder.rglob("*.tmp"))

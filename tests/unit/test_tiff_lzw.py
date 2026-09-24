import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from Common.dataloader.tiff_lzw import lzw_decode, read_lzw_tiff, read_tiff


def _lzw_encode(data, clear_every=None):
    """Reference TIFF LZW encoder (MSB-first, early change), for round-trip tests."""
    codes, widths = [], []
    nbits = 9

    def emit(code):
        codes.append(code)
        widths.append(nbits)

    def reset():
        return {bytes((i,)): i for i in range(256)}

    table = reset()
    emit(256)
    current = b""
    emitted = 0
    for byte in data:
        candidate = current + bytes((byte,))
        if candidate in table:
            current = candidate
            continue
        emit(table[current])
        emitted += 1
        table[candidate] = len(table) + 2  # skip Clear (256) and EOI (257)
        if len(table) + 2 >= (1 << nbits) and nbits < 12:  # decoder lags one entry behind
            nbits += 1
        current = bytes((byte,))
        if len(table) + 2 >= 4093 or (clear_every and emitted % clear_every == 0):
            emit(table[current])
            emit(256)
            table, nbits, current = reset(), 9, b""
    if current:
        emit(table[current])
    emit(257)

    bits = "".join(format(code, f"0{width}b") for code, width in zip(codes, widths))
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


@pytest.mark.parametrize("clear_every", [None, 37])
def test_lzw_round_trip_including_table_resets(clear_every):
    rng = np.random.default_rng(0)
    # Low-entropy runs exercise long table entries and the KwKwK case; the size
    # forces the code width through 9..12 bits and a full-table Clear.
    payload = bytes(rng.choice([0, 1, 2, 255], size=40_000, p=[0.7, 0.1, 0.1, 0.1]).astype(np.uint8))
    payload += b"a" * 500 + bytes(range(256)) * 20

    assert lzw_decode(_lzw_encode(payload, clear_every)) == payload


def test_read_tiff_passes_through_uncompressed_files(tmp_path):
    values = np.arange(12, dtype="<f8").reshape(3, 4)
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, values)

    assert np.array_equal(read_tiff(path), values)
    with pytest.raises(ValueError, match="not LZW"):
        read_lzw_tiff(path)


SNOWMELT_ROOT = Path(os.environ.get("SNOWMELT_DATA_ROOT", Path.home() / "PhD" / "Data" / "snowmelt"))


@pytest.mark.skipif(not (SNOWMELT_ROOT / "S2_rawbands").is_dir(), reason="snowmelt data not available")
def test_read_lzw_tiff_matches_opencv_on_real_geotiff():
    cv2 = pytest.importorskip("cv2")
    path = sorted((SNOWMELT_ROOT / "S2_rawbands").glob("*.tif"))[0]

    decoded = read_lzw_tiff(path)

    assert np.array_equal(decoded, cv2.imread(str(path), cv2.IMREAD_UNCHANGED), equal_nan=True)

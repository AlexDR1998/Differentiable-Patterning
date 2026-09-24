"""Dependency-free reading of LZW-compressed TIFFs.

``tifffile`` parses every TIFF but delegates LZW decompression to the optional
``imagecodecs`` package. Where that is not installed (e.g. the training Docker
image), :func:`read_tiff` falls back to a pure-Python TIFF LZW decoder for
striped single-sample images without a predictor, which covers GDAL's default
LZW GeoTIFF output. Throughput is about 8 MB/s of decoded data.
"""

import numpy as np
import tifffile

_CLEAR, _EOI = 256, 257
_LZW = 5


def lzw_decode(data):
    """Decode one TIFF LZW strip (MSB-first codes, 9-12 bits, early change)."""
    out = bytearray()
    table = [bytes((i,)) for i in range(256)] + [b"", b""]
    padded = bytes(data) + b"\0\0\0"
    total_bits = len(data) * 8
    bitpos, nbits, prev = 0, 9, None
    while bitpos + nbits <= total_bits:
        byte = bitpos >> 3
        chunk = (padded[byte] << 16) | (padded[byte + 1] << 8) | padded[byte + 2]
        code = (chunk >> (24 - (bitpos & 7) - nbits)) & ((1 << nbits) - 1)
        bitpos += nbits
        if code == _CLEAR:
            del table[258:]
            nbits, prev = 9, None
            continue
        if code == _EOI:
            break
        if prev is None:
            entry = table[code]
        elif code < len(table):
            entry = table[code]
            table.append(prev + entry[:1])
        else:  # the KwKwK case: code refers to the entry being defined
            entry = prev + prev[:1]
            table.append(entry)
        out += entry
        prev = entry
        if len(table) >= (1 << nbits) - 1 and nbits < 12:
            nbits += 1
    return bytes(out)


def read_lzw_tiff(path):
    """Read the first page of a striped, single-sample, predictor-free LZW TIFF."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        if page.compression != _LZW:
            raise ValueError(f"{path} is not LZW-compressed")
        if page.is_tiled or page.samplesperpixel != 1 or page.predictor != 1:
            raise ValueError(
                f"{path}: only striped, single-sample LZW TIFFs without a predictor "
                "can be decoded without imagecodecs"
            )
        handle = tif.filehandle
        strips = []
        for offset, count in zip(page.dataoffsets, page.databytecounts):
            handle.seek(offset)
            strips.append(lzw_decode(handle.read(count)))
        dtype = page.dtype.newbyteorder(tif.byteorder)
        values = np.frombuffer(b"".join(strips), dtype=dtype)
        expected = int(np.prod(page.shape))
        if values.size < expected:
            raise ValueError(f"{path}: decoded {values.size} values, expected {expected}")
        return values[:expected].reshape(page.shape)


def read_tiff(path):
    """``tifffile.imread``, falling back to the built-in LZW decoder without imagecodecs."""
    try:
        return tifffile.imread(path)
    except ValueError as error:
        if "imagecodecs" not in str(error):
            raise
        return read_lzw_tiff(path)


__all__ = ["lzw_decode", "read_lzw_tiff", "read_tiff"]

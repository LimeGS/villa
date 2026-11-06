"""Native decoded snapshots and array reads agree with independent codec fixtures."""
import json

import numpy as np
import pytest
from numcodecs import Delta, Shuffle, Zlib

from vc_reader.reader import ZarrReader, writer


def fixture(tmp_path, array, order, filters, compressor=None):
    metadata = dict(zarr_format=2, shape=array.shape, chunks=array.shape,
                    dtype=array.dtype.str, fill_value=0, order=order,
                    compressor=compressor.get_config() if compressor else None,
                    filters=[codec.get_config() for codec in filters] or None)
    data = array.tobytes(order=order)
    for codec in filters:
        data = codec.encode(data)
    if compressor:
        data = compressor.encode(data)
    data = bytes(data)
    (tmp_path / '.zarray').write_text(json.dumps(metadata))
    (tmp_path / '0.0.0').write_bytes(data)
    return data, metadata


@pytest.mark.parametrize('dtype', ['u1', '<u2', '>u2', '<i2', '>i4', '<u8', '<i8', '<f4', '>f8'])
@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('filter_names', [[], ['delta'], ['shuffle'], ['delta', 'shuffle']])
def test_snapshot_and_array_decode_numeric_layouts(tmp_path, dtype, order, filter_names):
    array = (np.arange(24) * 3).astype(dtype).reshape(2, 3, 4)
    available = {'delta': Delta(dtype=dtype), 'shuffle': Shuffle(elementsize=array.dtype.itemsize)}
    data, metadata = fixture(tmp_path, array, order, [available[name] for name in filter_names], Zlib())
    decoded = writer.decode_chunk(data, metadata, max_decode_bytes=array.nbytes)
    assert decoded == array.tobytes()
    with ZarrReader.open(tmp_path, threads=1, memory_cache=0) as reader:
        np.testing.assert_array_equal(reader[1:, 1:, 1:], array[1:, 1:, 1:])


@pytest.mark.parametrize('dtype,astype', [('<i2', 'i1'), ('>i4', '<i2'), ('<f8', '<f4')])
def test_delta_encoded_dtype_can_differ(tmp_path, dtype, astype):
    array = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    data, metadata = fixture(tmp_path, array, 'F', [Delta(dtype=dtype, astype=astype)])
    assert writer.decode_chunk(data, metadata) == array.tobytes()


@pytest.mark.parametrize('dtype', ['<u8', '<i8'])
def test_delta_integer_values_never_pass_through_float(tmp_path, dtype):
    array = np.array([2**60, 2**60 + 3, 2**60 - 5, 2**60 + 1], dtype=dtype).reshape(1, 1, 4)
    data, metadata = fixture(tmp_path, array, 'C', [Delta(dtype=dtype)])
    assert writer.decode_chunk(data, metadata) == array.tobytes()


def test_decode_admission_and_truncation(tmp_path):
    array = np.arange(24, dtype='<u2').reshape(2, 3, 4)
    data, metadata = fixture(tmp_path, array, 'F', [])
    with pytest.raises((ValueError, RuntimeError), match='limit|budget'):
        writer.decode_chunk(data, metadata, max_decode_bytes=array.nbytes - 1)
    with pytest.raises((ValueError, RuntimeError), match='bytes|size|truncat'):
        writer.decode_chunk(data[:-1], metadata)
    metadata['chunks'] = [2**30] * 3
    with pytest.raises((ValueError, RuntimeError), match='limit|budget|overflow|large'):
        writer.decode_chunk(b'', metadata)


def test_writer_rejects_new_read_only_layout_before_mutation(tmp_path):
    array = np.arange(24, dtype='<u2').reshape(2, 3, 4)
    fixture(tmp_path, array, 'F', [Delta(dtype=array.dtype)])
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match='read-only|Fortran|filters'):
        writer.ZarrWriter.open(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize('dtype', ['i1', '<i2', '<i8', '<u8', '<f2', '<f4', '<f8'])
@pytest.mark.parametrize('astype', ['i1', '<i2', '<i8', '<u8', '<f2', '<f4', '<f8'])
def test_delta_accumulation_matches_numcodecs_precision(dtype, astype):
    codec = Delta(dtype=dtype, astype=astype)
    encoded = np.array([100, 1, 1, 1], dtype=astype)
    if np.dtype(dtype).kind == 'f' and np.dtype(astype).kind == 'f':
        encoded = np.array([2048, 1, -2048, .5], dtype=astype)
        if np.dtype(astype).itemsize >= 4:
            encoded = np.array([2**24, 1, -2**24, .5], dtype=astype)
    metadata = dict(zarr_format=2, shape=[1, 1, 4], chunks=[1, 1, 4], dtype=np.dtype(dtype).str,
                    fill_value=0, order='C', compressor=None, filters=[codec.get_config()])
    with np.errstate(over='ignore', invalid='ignore'):
        expected = codec.decode(encoded).tobytes()
    assert writer.decode_chunk(encoded, metadata) == expected


@pytest.mark.parametrize('dtype,astype,values', [
    ('<i2', '<f4', [-100, -1, 100, 1]),
    ('<f4', '<i4', [2**24, 1, -2**24, 1]),
    ('<i8', '<f4', [2**24, 1, -2**24, 1]),
    ('<i8', '<i8', [2**63 - 1, 1, -1, -1]),
    ('<u8', '<u8', [2**64 - 1, 1, 1, 1]),
])
def test_delta_mixed_types_and_overflow(dtype, astype, values):
    codec = Delta(dtype=dtype, astype=astype)
    encoded = np.array(values, dtype=astype)
    metadata = dict(zarr_format=2, shape=[1, 1, 4], chunks=[1, 1, 4], dtype=dtype,
                    fill_value=0, order='C', compressor=None, filters=[codec.get_config()])
    assert writer.decode_chunk(encoded, metadata) == codec.decode(encoded).tobytes()

"""Object reads enforce admission while streaming, before any cache publication."""
import gzip
import http.server
import pickle
import threading

import numpy as np
import pytest

from vc_reader.reader.writer import ObjectStore
from vc_reader.reader.disk_cache import disk_budget


@pytest.fixture
def endpoint():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/missing':
                self.send_error(404)
                return
            if self.path == '/denied':
                self.send_error(403)
                return
            body = b'x' * (33 if self.path == '/large' else 32)
            if self.path == '/gzip':
                body = gzip.compress(b'x' * 1024)
            self.send_response(200)
            if self.path != '/unsized':
                self.send_header('Content-Length', str(len(body)))
            if self.path == '/gzip':
                self.send_header('Content-Encoding', 'gzip')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_local_limit_survives_pickle(tmp_path):
    (tmp_path / 'exact').write_bytes(b'x' * 32)
    (tmp_path / 'large').write_bytes(b'x' * 33)
    store = pickle.loads(pickle.dumps(ObjectStore(tmp_path, auth=None, max_read_bytes=32)))
    assert store.read('exact') == b'x' * 32
    assert store.read('missing') is None
    with pytest.raises(RuntimeError, match='limit|budget'):
        store.read('large')


@pytest.mark.parametrize('name,limit', [('large', 32), ('unsized', 31), ('gzip', 32)])
def test_remote_limit_covers_headers_streams_and_expansion(endpoint, name, limit):
    store = ObjectStore(endpoint, auth=None, max_read_bytes=limit)
    with pytest.raises(RuntimeError, match='limit|budget'):
        store.read(name)


def test_remote_limit_preserves_exact_bytes_and_missingness(endpoint):
    store = ObjectStore(endpoint, auth=None, max_read_bytes=32)
    assert store.read('exact') == b'x' * 32
    assert store.read('missing') is None
    with pytest.raises(Exception, match='403|denied|Denied'):
        store.read('denied')


@pytest.mark.parametrize('budget', [0, 16, 64])
def test_streamed_bytes_do_not_require_retained_cache_space(endpoint, tmp_path, budget):
    store = ObjectStore(endpoint, auth=None, max_read_bytes=32)
    raw = store.read('exact')
    assert raw == b'x' * 32
    cache = disk_budget(str(tmp_path), budget)
    path = tmp_path / 'chunk'
    cache.write(str(path), np.frombuffer(raw, np.uint8))
    assert path.exists() == (budget >= len(raw))
    if path.exists():
        assert path.read_bytes() == raw


def test_zero_read_limit_allows_only_empty_local_object(tmp_path):
    (tmp_path / 'empty').write_bytes(b'')
    (tmp_path / 'one').write_bytes(b'x')
    store = ObjectStore(tmp_path, auth=None, max_read_bytes=0)
    assert store.read('empty') == b''
    with pytest.raises(RuntimeError, match='limit|budget'):
        store.read('one')

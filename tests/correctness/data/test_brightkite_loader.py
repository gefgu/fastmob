import gzip
import io

from fastmob.data import brightkite

ROWS = (
    "0\t2010-10-17T01:48:53Z\t39.747652\t-104.992510\tvenue-a\n"
    "0\t2010-10-16T06:02:04Z\t\t\tvenue-b\n"
    "1\t2010-10-15T06:02:04Z\t37.6\t-122.4\tvenue-c\n"
)


def _fake_urlopen(url):
    return io.BytesIO(gzip.compress(ROWS.encode()))


def test_load_brightkite_drops_rows_without_coordinates(monkeypatch, tmp_path):
    monkeypatch.setattr(brightkite.urllib.request, "urlopen", _fake_urlopen)
    traj = brightkite.load_brightkite(max_rows=None, cache=tmp_path)
    table = traj.to_native()
    assert table["uid"].to_pylist() == ["0", "1"]
    assert str(table.schema.field("datetime").type) == "timestamp[us]"


def test_load_brightkite_stops_after_max_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(brightkite.urllib.request, "urlopen", _fake_urlopen)
    assert len(brightkite.load_brightkite(max_rows=1, cache=tmp_path)) == 1

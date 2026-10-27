import io
import zipfile

import pytest
from fastmob.data import geolife

PLT = (
    "Geolife trajectory\nWGS 84\nAltitude is in Feet\nReserved 3\n0,2,255,Geolife Track\n0\n"
    "39.906631,116.385564,0,492,40097.58,2009-10-11,14:04:30\n"
    "39.906554,116.385625,0,492,40097.58,2009-10-11,14:04:35\n"
)


@pytest.fixture
def archive(monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for user in ("000", "001"):
            for trip in ("20091011140430", "20091012140430"):
                z.writestr(f"Geolife Trajectories 1.3/Data/{user}/Trajectory/{trip}.plt", PLT)
    monkeypatch.setattr(geolife, "HttpRangeFile", lambda url: io.BytesIO(buffer.getvalue()))


def test_load_geolife_reads_requested_users_and_trip_limit(archive, tmp_path):
    traj = geolife.load_geolife(users=["001"], max_trips_per_user=1, cache=tmp_path)
    assert len(traj) == 2
    assert set(traj.to_native()["uid"].to_pylist()) == {"001"}


def test_load_geolife_second_call_uses_cache(archive, monkeypatch, tmp_path):
    geolife.load_geolife(users=[0], cache=tmp_path)

    def fail(url):
        raise AssertionError("downloaded again")

    monkeypatch.setattr(geolife, "HttpRangeFile", fail)
    assert len(geolife.load_geolife(users=[0], cache=tmp_path)) == 4


def test_load_geolife_does_not_return_files_cached_for_another_selection(archive, tmp_path):
    assert len(geolife.load_geolife(users=["000"], max_trips_per_user=2, cache=tmp_path)) == 4
    assert len(geolife.load_geolife(users=["000"], max_trips_per_user=1, cache=tmp_path)) == 2

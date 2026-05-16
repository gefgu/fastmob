import skmob2
from skmob2.core import TrajDataFrame as CoreTrajDataFrame


def test_traj_dataframe_is_public_api():
    assert skmob2.TrajDataFrame is CoreTrajDataFrame
    assert "TrajDataFrame" in skmob2.__all__

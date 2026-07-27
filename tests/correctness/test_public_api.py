import fastmob
from fastmob.core import FlowDataFrame as CoreFlowDataFrame
from fastmob.core import TrajDataFrame as CoreTrajDataFrame


def test_dataframe_wrappers_are_public_api():
    assert fastmob.TrajDataFrame is CoreTrajDataFrame
    assert fastmob.FlowDataFrame is CoreFlowDataFrame
    assert "TrajDataFrame" in fastmob.__all__
    assert "FlowDataFrame" in fastmob.__all__


def test_measure_taxonomy_uses_canonical_public_packages():
    from fastmob.measures import collective, evaluation, individual

    assert individual.radius_of_gyration is fastmob.radius_of_gyration
    assert collective.visits_per_location is fastmob.visits_per_location
    assert evaluation.wasserstein_distance is fastmob.wasserstein_distance

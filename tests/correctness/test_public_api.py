import fkmob
from fkmob.core import FlowDataFrame as CoreFlowDataFrame
from fkmob.core import TrajDataFrame as CoreTrajDataFrame


def test_dataframe_wrappers_are_public_api():
    assert fkmob.TrajDataFrame is CoreTrajDataFrame
    assert fkmob.FlowDataFrame is CoreFlowDataFrame
    assert "TrajDataFrame" in fkmob.__all__
    assert "FlowDataFrame" in fkmob.__all__


def test_measure_taxonomy_uses_canonical_public_packages():
    import fkmob.measures.collective as collective
    import fkmob.measures.evaluation as evaluation
    import fkmob.measures.individual as individual

    assert individual.radius_of_gyration is fkmob.radius_of_gyration
    assert collective.visits_per_location is fkmob.visits_per_location
    assert evaluation.wasserstein_distance is fkmob.wasserstein_distance

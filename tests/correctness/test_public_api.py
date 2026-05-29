import skmob2
from skmob2.core import FlowDataFrame as CoreFlowDataFrame
from skmob2.core import TrajDataFrame as CoreTrajDataFrame


def test_dataframe_wrappers_are_public_api():
    assert skmob2.TrajDataFrame is CoreTrajDataFrame
    assert skmob2.FlowDataFrame is CoreFlowDataFrame
    assert "TrajDataFrame" in skmob2.__all__
    assert "FlowDataFrame" in skmob2.__all__


def test_measure_taxonomy_uses_canonical_public_packages():
    import skmob2.measures.collective as collective
    import skmob2.measures.evaluation as evaluation
    import skmob2.measures.individual as individual

    assert individual.radius_of_gyration is skmob2.radius_of_gyration
    assert collective.visits_per_location is skmob2.visits_per_location
    assert evaluation.wasserstein_distance is skmob2.wasserstein_distance

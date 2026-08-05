import pytest
from fastmob_vis import ecdf


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_ecdf_accepts_common_narwhals_backends(backend):
    module = pytest.importorskip(backend)
    if backend == "pandas":
        frame = module.DataFrame({"values": [[1.0, 2.0], [3.0]]})
    else:
        frame = module.DataFrame({"values": [[1.0, 2.0], [3.0]]})

    chart = ecdf(frame, value_col="values")

    assert chart.to_dict()["series"][0]["data"][0] == [1.0, 1 / 3]

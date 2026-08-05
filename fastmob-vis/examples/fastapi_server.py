import pyarrow as pa
from fastapi import FastAPI
from fastmob_vis import ecdf

app = FastAPI()

observed_jump_lengths = pa.table({"jump_lengths": [[0.4, 1.2, 1.2, 3.8, 10.0]]})
synthetic_jump_lengths = pa.table({"jump_lengths": [[0.6, 1.0, 2.0, 2.0, 7.5]]})


@app.get("/api/charts/jump-lengths-ecdf")
def jump_lengths_ecdf():
    chart = ecdf(
        observed_jump_lengths,
        value_col="jump_lengths",
        second=synthetic_jump_lengths,
        labels=("observed", "synthetic"),
    )
    return chart.to_dict()

from fastapi import FastAPI
from fastmob_vis import plot_jump_lengths_ecdf

app = FastAPI()

observed_jump_lengths = [0.4, 1.2, 1.2, 3.8, 10.0]
synthetic_jump_lengths = [0.6, 1.0, 2.0, 2.0, 7.5]


@app.get("/api/charts/jump-lengths-ecdf")
def jump_lengths_ecdf():
    figure = plot_jump_lengths_ecdf(
        observed_jump_lengths,
        synthetic_jump_lengths,
        labels=("observed", "synthetic"),
    )
    return figure.to_dict()

# Benchmark Results

These plots compare Fastkit-Mobility with every library that offers the same function, on the largest datasets we benchmark. Bars show average time (or peak extra memory) on a log axis, and the label on a Fastkit-Mobility bar is its advantage over the best other library. A bar reading *timeout* or *error* means the library did not finish.

The numbers are environment-specific, so they should be read as benchmark evidence for this run rather than as a universal performance guarantee.

## One function, every library

`jump_lengths` is the function the most libraries share.

![jump_lengths speed across eight libraries](../assets/benchmarks/jump-lengths-libraries.png)

## Measures

### Individual

![Individual measures, execution time](../assets/benchmarks/measures-individual-speed.png)

![Individual measures, peak extra memory](../assets/benchmarks/measures-individual-memory.png)

### Collective

![Collective measures, execution time](../assets/benchmarks/measures-collective-speed.png)

![Collective measures, peak extra memory](../assets/benchmarks/measures-collective-memory.png)

## Preprocessing

![Preprocessing, execution time](../assets/benchmarks/preprocessing-speed.png)

![Preprocessing, peak extra memory](../assets/benchmarks/preprocessing-memory.png)

## Distributed baseline: SparkMobility

![Execution time against SparkMobility](../assets/benchmarks/sparkmobility-speed.png)

![Peak extra memory against SparkMobility](../assets/benchmarks/sparkmobility-memory.png)

## Measures - Evaluation

*The evaluation and model plots below are earlier snapshots in the previous plotting style.*

#### Measures - Evaluation - (Speed) - NumPy and Rust kernels
![measures evaluation speed benchmark](../assets/benchmarks/fastmob_vs_skmob_evaluation_4M.png)
<!-- 
#### Measures - Evaluation - (Memory) (Still needs adjusts)
![measures evaluation memory benchmark](../assets/benchmarks/fastmob_vs_skmob_evaluation_pandas_4M_memory.png) -->


## Models

#### Models Agents - Speed
![Simple models speed benchmark](../assets/benchmarks/model_agent_based_100_agents.png)


#### Models Agents - Speed (No rust core yet...)
![Simple models speed benchmark](../assets/benchmarks/model_location_only_10k_locations.png)


## Privacy

![Privacy risk, execution time](../assets/benchmarks/privacy-speed.png)

## Reproduce the figures

The figures are drawn from the raw results of the sibling `fastmob_benchmarks` repository:

```bash
uv run --no-project --with matplotlib python scripts/make_benchmark_figures.py
```

Each figure is one entry of `figures()` in that script, so a new comparison needs one new entry.

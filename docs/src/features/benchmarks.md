# Benchmark Results

These plots collect the largest benchmark runs that directly compare skmob2 with scikit-mobility (`skmob`). They are useful as a quick visual snapshot of skmob2 behavior on 4M-row trajectory workloads.

The numbers are environment-specific, so they should be read as benchmark evidence for this run rather than as a universal performance guarantee.

## Preprocessing

#### Preprocessing - Pandas (Speed)
![4M-row pandas preprocessing speed benchmark](../assets/benchmarks/skmob2_vs_skmob_preprocessing_pandas_4M.png)

#### Preprocessing - Pandas (Memory) (Still needs adjusts)
![4M-row pandas preprocessing memory benchmark](../assets/benchmarks/skmob2_vs_skmob_preprocessing_pandas_4M_memory.png)


#### Preprocessing - Polars (Speed)
![4M-row polars preprocessing speed benchmark](../assets/benchmarks/skmob2_vs_skmob_preprocessing_polars_4M.png)

#### Preprocessing - Polars (Memory) (Still needs adjusts)
![4M-row polars preprocessing memory benchmark](../assets/benchmarks/skmob2_vs_skmob_preprocessing_polars_4M_memory.png)


## Measures - Individual

#### Measures - Individual - Pandas (Speed)
![4M-row pandas measures individual speed benchmark](../assets/benchmarks/skmob2_vs_skmob_individual_pandas_4M.png)

#### Measures - Individual - Pandas (Memory) (Still needs adjusts)
![4M-row pandas measures individual memory benchmark](../assets/benchmarks/skmob2_vs_skmob_individual_pandas_4M_memory.png)


#### Measures - Individual - Polars (Speed)
![4M-row polars measures individual speed benchmark](../assets/benchmarks/skmob2_vs_skmob_individual_polars_4M.png)

#### Measures - Individual - Polars (Memory) (Still needs adjusts)
![4M-row polars measures individual memory benchmark](../assets/benchmarks/skmob2_vs_skmob_individual_polars_4M_memory.png)


## Measures - Collective

#### Measures - Collective - Pandas (Speed)
![4M-row pandas measures collective speed benchmark](../assets/benchmarks/skmob2_vs_skmob_collective_pandas_4M.png)

#### Measures - Collective - Pandas (Memory) (Still needs adjusts)
![4M-row pandas measures collective memory benchmark](../assets/benchmarks/skmob2_vs_skmob_collective_pandas_4M_memory.png)


#### Measures - Collective - Polars (Speed)
![4M-row polars measures collective speed benchmark](../assets/benchmarks/skmob2_vs_skmob_collective_polars_4M.png)

<!-- #### Measures - Collective - Polars (Memory) (Still needs adjusts)
![4M-row polars measures collective memory benchmark](../assets/benchmarks/skmob2_vs_skmob_collective_polars_4M_memory.png) -->


## Measures - Evaluation

#### Measures - Evaluation - (Speed) - Mostly numpy/scipy wrappers
![measures evaluation speed benchmark](../assets/benchmarks/skmob2_vs_skmob_evaluation_4M.png)
<!-- 
#### Measures - Evaluation - (Memory) (Still needs adjusts)
![measures evaluation memory benchmark](../assets/benchmarks/skmob2_vs_skmob_evaluation_pandas_4M_memory.png) -->


## Models

#### Models Agents - Speed
![Simple models speed benchmark](../assets/benchmarks/model_agent_based_100_agents.png)


#### Models Agents - Speed (No rust core yet...)
![Simple models speed benchmark](../assets/benchmarks/model_location_only_10k_locations.png)


## Privacy

#### Privacy - Pandas (Speed)
![4M-row pandas privacy speed benchmark](../assets/benchmarks/skmob2_vs_skmob_privacy_pandas_all.png)

<!-- #### Privacy - Pandas (Memory) (Still needs adjusts)
![4M-row pandas privacy memory benchmark](../assets/benchmarks/skmob2_vs_skmob_privacy_pandas_all_memory.png) -->


#### Privacy - Polars (Speed)
![4M-row polars privacy speed benchmark](../assets/benchmarks/skmob2_vs_skmob_privacy_polars_all.png)

<!-- #### Privacy - Polars (Memory) (Still needs adjusts) -->
<!-- ![4M-row polars privacy memory benchmark](../assets/benchmarks/skmob2_vs_skmob_privacy_polars_all_memory.png) -->


## Pre-Sorted Benchmark Results

These refer to special metrics which benefit from sorted paths.


### Preprocessing

#### Preprocessing - Pandas (Speed)
![4M-row pandas preprocessing speed benchmark](../assets/benchmarks/sorted/skmob2_vs_skmob_preprocessing_pandas_4M.png)

<!-- #### Preprocessing - Pandas (Memory) (Still needs adjusts)
![4M-row pandas preprocessing memory benchmark](../assets/benchmarks/skmob2_vs_skmob_preprocessing_pandas_4M_memory.png) -->


#### Preprocessing - Polars (Speed)
![4M-row polars preprocessing speed benchmark](../assets/benchmarks/skmob2_vs_skmob_preprocessing_polars_4M.png)

<!-- #### Preprocessing - Polars (Memory) (Still needs adjusts)
![4M-row polars preprocessing memory benchmark](../assets/benchmarks/skmob2_vs_skmob_preprocessing_polars_4M_memory.png) -->


## Measures - Individual

#### Measures - Individual - Pandas (Speed)
![4M-row pandas measures individual speed benchmark](../assets/benchmarks/sorted/skmob2_vs_skmob_individual_pandas_4M.png)

<!-- #### Measures - Individual - Pandas (Memory) (Still needs adjusts)
![4M-row pandas measures individual memory benchmark](../assets/benchmarks/skmob2_vs_skmob_individual_pandas_4M_memory.png) -->


#### Measures - Individual - Polars (Speed)
![4M-row polars measures individual speed benchmark](../assets/benchmarks/sorted/skmob2_vs_skmob_individual_polars_4M.png)

<!-- #### Measures - Individual - Polars (Memory) (Still needs adjusts)
![4M-row polars measures individual memory benchmark](../assets/benchmarks/skmob2_vs_skmob_individual_polars_4M_memory.png) -->


## Measures - Collective

#### Measures - Collective - Pandas (Speed)
![4M-row pandas measures collective speed benchmark](../assets/benchmarks/skmob2_vs_skmob_collective_pandas_4M.png)

<!-- #### Measures - Collective - Pandas (Memory) (Still needs adjusts)
![4M-row pandas measures collective memory benchmark](../assets/benchmarks/skmob2_vs_skmob_collective_pandas_4M_memory.png) -->


#### Measures - Collective - Polars (Speed)
![4M-row polars measures collective speed benchmark](../assets/benchmarks/skmob2_vs_skmob_collective_polars_4M.png)

<!-- #### Measures - Collective - Polars (Memory) (Still needs adjusts)
![4M-row polars measures collective memory benchmark](../assets/benchmarks/skmob2_vs_skmob_collective_polars_4M_memory.png) -->


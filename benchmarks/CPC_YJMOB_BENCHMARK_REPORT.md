# YJMob CPC benchmark report

Run: 2026-09-15T16:40:13.948489+00:00  
CPU: Intel(R) Xeon(R) Silver 4316 CPU @ 2.30GHz (40 logical CPUs)  
Python: 3.12.12  
Data: `/mnt/raid5/gustavo/fastmob_benchmarks/data/raw/yjmob_wgs84_simple.parquet`

The 1M/10M labels are approximate raw-ping counts per side. Each comparison uses two disjoint cohorts with complete user traces. Peak RSS is sampled additional process RSS above the already loaded cohort frames; it is approximate.

| Tier | Input load/slice (s) | H3 | Direct Trips total (s) | OD DataFrame total (s) | OD / Direct time | Direct peak RSS delta (MB) | OD peak RSS delta (MB) | Trips / side | OD edges / side | CPC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1M | 0.149 | 7 | 0.320 | 0.215 | 0.67× | 415.3 | 34.7 | 535,517 / 531,791 | 84,500 / 82,031 | 0.520984 |
| 1M | 0.149 | 8 | 0.324 | 0.287 | 0.89× | 153.9 | 50.7 | 710,362 / 713,739 | 279,367 / 274,090 | 0.253110 |
| 1M | 0.149 | 9 | 0.311 | 0.327 | 1.05× | 97.5 | 15.5 | 783,278 / 779,611 | 392,626 / 387,813 | 0.146750 |
| 10M | 0.147 | 7 | 2.644 | 1.209 | 0.46× | 1852.2 | 85.1 | 5,368,858 / 5,398,030 | 264,098 / 260,768 | 0.788137 |
| 10M | 0.147 | 8 | 3.014 | 1.807 | 0.60× | 1145.0 | 261.3 | 7,147,083 / 7,190,667 | 1,462,532 / 1,449,799 | 0.563460 |
| 10M | 0.147 | 9 | 3.324 | 2.491 | 0.75× | 1223.1 | 558.3 | 7,870,230 / 7,912,372 | 2,579,427 / 2,572,576 | 0.418283 |

## Timing breakdown

### 1M pings per side, H3 7

| Path / phase | Median seconds |
|---|---:|
| Direct: trajectory_to_trips_left | 0.1579 |
| Direct: trajectory_to_trips_right | 0.1383 |
| Direct: cpc_wrapper | 0.0241 |
| OD path: trajectory_to_od_left | 0.1043 |
| OD path: trajectory_to_od_right | 0.0996 |
| OD path: flow_dataframe_left | 0.0002 |
| OD path: flow_dataframe_right | 0.0001 |
| OD path: cpc_wrapper | 0.0110 |
| OD path: standalone_od_matrix_aggregation_both_sides | 0.0332 |
| direct: endpoint factorization | 0.0167 |
| direct: Rust CPC kernel | 0.0213 |
| materialized: endpoint factorization | 0.0037 |
| materialized: Rust CPC kernel | 0.0083 |

### 1M pings per side, H3 8

| Path / phase | Median seconds |
|---|---:|
| Direct: trajectory_to_trips_left | 0.1369 |
| Direct: trajectory_to_trips_right | 0.1328 |
| Direct: cpc_wrapper | 0.0548 |
| OD path: trajectory_to_od_left | 0.1271 |
| OD path: trajectory_to_od_right | 0.1253 |
| OD path: flow_dataframe_left | 0.0002 |
| OD path: flow_dataframe_right | 0.0001 |
| OD path: cpc_wrapper | 0.0347 |
| OD path: standalone_od_matrix_aggregation_both_sides | 0.0547 |
| direct: endpoint factorization | 0.0176 |
| direct: Rust CPC kernel | 0.0509 |
| materialized: endpoint factorization | 0.0082 |
| materialized: Rust CPC kernel | 0.0296 |

### 1M pings per side, H3 9

| Path / phase | Median seconds |
|---|---:|
| Direct: trajectory_to_trips_left | 0.1305 |
| Direct: trajectory_to_trips_right | 0.1241 |
| Direct: cpc_wrapper | 0.0566 |
| OD path: trajectory_to_od_left | 0.1464 |
| OD path: trajectory_to_od_right | 0.1401 |
| OD path: flow_dataframe_left | 0.0002 |
| OD path: flow_dataframe_right | 0.0001 |
| OD path: cpc_wrapper | 0.0403 |
| OD path: standalone_od_matrix_aggregation_both_sides | 0.0748 |
| direct: endpoint factorization | 0.0165 |
| direct: Rust CPC kernel | 0.0559 |
| materialized: endpoint factorization | 0.0115 |
| materialized: Rust CPC kernel | 0.0340 |

### 10M pings per side, H3 7

| Path / phase | Median seconds |
|---|---:|
| Direct: trajectory_to_trips_left | 1.2293 |
| Direct: trajectory_to_trips_right | 1.2231 |
| Direct: cpc_wrapper | 0.1917 |
| OD path: trajectory_to_od_left | 0.6007 |
| OD path: trajectory_to_od_right | 0.5778 |
| OD path: flow_dataframe_left | 0.0002 |
| OD path: flow_dataframe_right | 0.0001 |
| OD path: cpc_wrapper | 0.0304 |
| OD path: standalone_od_matrix_aggregation_both_sides | 0.1502 |
| direct: endpoint factorization | 0.1189 |
| direct: Rust CPC kernel | 0.1774 |
| materialized: endpoint factorization | 0.0051 |
| materialized: Rust CPC kernel | 0.0247 |

### 10M pings per side, H3 8

| Path / phase | Median seconds |
|---|---:|
| Direct: trajectory_to_trips_left | 1.2622 |
| Direct: trajectory_to_trips_right | 1.2559 |
| Direct: cpc_wrapper | 0.4954 |
| OD path: trajectory_to_od_left | 0.7851 |
| OD path: trajectory_to_od_right | 0.8127 |
| OD path: flow_dataframe_left | 0.0002 |
| OD path: flow_dataframe_right | 0.0001 |
| OD path: cpc_wrapper | 0.2087 |
| OD path: standalone_od_matrix_aggregation_both_sides | 0.3450 |
| direct: endpoint factorization | 0.0876 |
| direct: Rust CPC kernel | 0.4735 |
| materialized: endpoint factorization | 0.0204 |
| materialized: Rust CPC kernel | 0.1773 |

### 10M pings per side, H3 9

| Path / phase | Median seconds |
|---|---:|
| Direct: trajectory_to_trips_left | 1.2732 |
| Direct: trajectory_to_trips_right | 1.2701 |
| Direct: cpc_wrapper | 0.7806 |
| OD path: trajectory_to_od_left | 1.0252 |
| OD path: trajectory_to_od_right | 1.0213 |
| OD path: flow_dataframe_left | 0.0002 |
| OD path: flow_dataframe_right | 0.0001 |
| OD path: cpc_wrapper | 0.4437 |
| OD path: standalone_od_matrix_aggregation_both_sides | 0.5065 |
| direct: endpoint factorization | 0.0950 |
| direct: Rust CPC kernel | 0.7123 |
| materialized: endpoint factorization | 0.0321 |
| materialized: Rust CPC kernel | 0.2900 |

## Interpretation

Direct total includes building paired Trips for both cohorts and the public CPC call. OD total includes `trajectory_to_od` for both cohorts, FlowDataFrame construction, and the public CPC call. Endpoint factorization and Rust kernel rows are diagnostic decompositions and are not added again to those totals.

In this run, the direct Trips path is slower and uses more additional RSS in all six tier/resolution cases. Avoiding `od_matrix` does not offset carrying and factorizing every paired trip through CPC; the materialized path's CPC sees only unique OD edges. The direct helper remains useful when callers need individual consecutive-ping pairs, but this benchmark does not show a performance or memory win for CPC as currently implemented.

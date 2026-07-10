# Benchmarks

Run commands from the repository root.

## STS-EPR, one day

This focused command reuses the large-scale model benchmark path, which runs
social trajectory models from `2020-01-01 08:00:00` to `2020-01-02 08:00:00`.
It benchmarks only STS-EPR with the selected agent count and writes the standard benchmark
JSON output under `benchmarks/results/...`.

```bash
N_AGENTS=500
uv run python benchmarks/speed_models_large_scale.py \
  --library fkmob \
  --mode trajectory \
  --metrics sts_epr_custom \
  --sts-epr-agents "$N_AGENTS" \
  --sizes 1000 \
  --iterations 3
```

Change `iterations` to `1` for a quick smoke run.

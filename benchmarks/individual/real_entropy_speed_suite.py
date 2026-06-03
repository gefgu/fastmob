"""Opt-in benchmark suite for the expensive individual real_entropy metric."""

from __future__ import annotations

from benchmarks.individual import speed_suite as _base

BenchmarkSpec = _base.BenchmarkSpec
EXPENSIVE_INDIVIDUAL_METRICS = _base.EXPENSIVE_INDIVIDUAL_METRICS


def parse_args(argv: list[str] | None = None):
    args = _base.parse_args(argv)
    args.expensive_metric = "real_entropy"
    return args


def run_suite(args, *, backend: str | None = None):
    old_metrics = _base.INDIVIDUAL_METRICS
    try:
        _base.INDIVIDUAL_METRICS = EXPENSIVE_INDIVIDUAL_METRICS
        payload = _base.run_suite(args, backend=backend)
    finally:
        _base.INDIVIDUAL_METRICS = old_metrics
    payload["metadata"]["suite"] = "individual_real_entropy"
    payload["metadata"]["expensive_metric"] = "real_entropy"
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for input_order in _base.concrete_input_orders(args):
        order_args = _base.argparse.Namespace(**vars(args))
        order_args.input_order = input_order
        for backend in _base.concrete_backends(order_args):
            payload = run_suite(order_args, backend=backend)
            output_path = _base.build_output_path(
                _base.Path(order_args.output_dir),
                order_args.library,
                order_args.timing_mode,
                backend,
                order_args.profile,
                order_args.input_order,
            )
            output_path = output_path.with_name(output_path.name.replace("individual_", "individual_real_entropy_"))
            _base.write_json(payload, output_path)
            print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

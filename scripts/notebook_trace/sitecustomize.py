"""Record which fastmob functions a notebook cell calls directly (loaded through PYTHONPATH).

Active only when ``FASTMOB_CALL_LOG`` is set. A profile hook notes every call of a fastmob Python
function whose caller is notebook cell code, together with the cell's execution count, and writes a
JSON list ``[[canonical_symbol, execution_count], ...]``, rewritten whenever a new call is seen (the
kernel is stopped without a clean exit, so there is no exit hook to rely on).
"""

import json
import os
import sys

OUT = os.environ.get("FASTMOB_CALL_LOG")


def _install():
    calls = set()
    code_to_symbol = {}
    seen = {"modules": -1}

    def rebuild():
        seen["modules"] = len(sys.modules)
        for name, module in list(sys.modules.items()):
            if not name.startswith("fastmob") or module is None:
                continue
            for attr in list(vars(module).values()):
                targets = []
                if isinstance(attr, type) and attr.__module__.startswith("fastmob"):
                    targets = [(m, f"{attr.__module__}.{attr.__qualname__}.{n}") for n, m in vars(attr).items()]
                elif callable(attr) and getattr(attr, "__module__", "").startswith("fastmob"):
                    targets = [(attr, f"{attr.__module__}.{attr.__qualname__}")]
                for fn, symbol in targets:
                    fn = getattr(fn, "__func__", fn)
                    fn = getattr(fn, "fget", fn)
                    code = getattr(fn, "__code__", None)
                    if code is not None:
                        code_to_symbol.setdefault(code, symbol)

    def in_cell(frame):
        name = frame.f_code.co_filename
        return name.startswith("<ipython-input") or "ipykernel" in name

    def profiler(frame, event, arg):
        if event != "call":
            return
        caller = frame.f_back
        if caller is None or not in_cell(caller) or "fastmob" not in frame.f_code.co_filename:
            return
        if len(sys.modules) != seen["modules"] and frame.f_code not in code_to_symbol:
            rebuild()
        symbol = code_to_symbol.get(frame.f_code)
        if symbol:
            try:
                from IPython import get_ipython

                # Newer IPython bumps ``execution_count`` before the cell runs, so count the input history.
                count = len(get_ipython().user_ns["In"]) - 1
            except (ImportError, AttributeError, KeyError):
                count = None
            if (symbol, count) not in calls:
                calls.add((symbol, count))
                dump()

    def dump():
        with open(OUT, "w") as fh:
            json.dump(sorted(calls, key=lambda c: (c[0], c[1] or 0)), fh)

    sys.setprofile(profiler)


if OUT and "ipykernel_launcher" in " ".join(getattr(sys, "orig_argv", [])):
    _install()

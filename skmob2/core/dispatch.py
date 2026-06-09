from typing import Any, Callable, Dict


class TrajectoryDispatcher:
    """Handles routing Narwhals DataFrames to backend-specific mathematical implementations."""
    
    def __init__(self, arrow_ops: Dict[str, Callable], numpy_ops: Dict[str, Callable]):
        # Base dictionary combining shared extraction logic with function-specific ops
        self.dispatch = {
            "arrow": {
                "extract_data": lambda s: s.to_arrow(),
                **arrow_ops
            },
            "numpy": {
                "extract_data": lambda s: s.to_numpy(),
                **numpy_ops
            }
        }

    def get_backend_key(self, df: Any) -> str:
        """Determines if the backend should route to arrow or numpy paths."""
        backend_name = df.implementation.value
        return "arrow" if backend_name in ("polars", "pyarrow") else "numpy"

    def get_ops(self, df: Any) -> dict:
        """Returns the fully merged operations dictionary safely."""
        key = self.get_backend_key(df)
        return self.dispatch.get(key, self.dispatch["numpy"])
"""The in-page formatter scripts must run against every generic chart's option (needs Node)."""

from __future__ import annotations

import re
import shutil
import subprocess

import pyarrow as pa
import pytest
from fastmob_vis import bar, boxplot, ecdf, heatmap, scatter

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")

RUN = """
const script = require("fs").readFileSync(0, "utf8");
const echarts = { init: () => ({ setOption() {}, resize() {} }) };
const element = {};
new Function("echarts", "document", "window", script)(
  echarts, { getElementById: () => element }, { addEventListener() {} });
"""

TABLE = pa.table({"c": ["a", "b"], "x": [1.0, 2.0], "y": [2.0, 3.0], "v": [1.0, 4.0]})
CHARTS = {
    "ecdf": lambda: ecdf(TABLE, value_col="v"),
    "ecdf_pair": lambda: ecdf(TABLE, value_col="v", second=TABLE),
    "bar": lambda: bar(TABLE, category_col="c", value_col="v"),
    "scatter": lambda: scatter(TABLE, x_col="x", y_col="y"),
    "heatmap": lambda: heatmap(TABLE, x_col="c", y_col="c", value_col="v"),
    "boxplot": lambda: boxplot(TABLE, category_col="c", value_col="v"),
}


@pytest.mark.parametrize("name", CHARTS)
def test_chart_script_runs_without_a_browser(name):
    snippet = CHARTS[name]().render()._to_inline_snippet()
    script = re.search(r"<script>(.*?)</script>", snippet, re.DOTALL).group(1)
    done = subprocess.run(["node", "-e", RUN], input=script, capture_output=True, text=True, check=False)

    assert done.returncode == 0, done.stderr[-400:]

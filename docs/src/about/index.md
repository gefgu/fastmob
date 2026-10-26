# About fastmob

fastmob is a Python library for human-mobility analysis. It pairs a
familiar dataframe-facing API with Rust kernels for the compute-heavy parts of
trajectory processing and measurement. See [Features](../features/index.md)
for what it can do.

[![PyPI](https://img.shields.io/pypi/v/fastmob)](https://pypi.org/project/fastmob/)
[![Python](https://img.shields.io/pypi/pyversions/fastmob)](https://pypi.org/project/fastmob/)
[![License](https://img.shields.io/pypi/l/fastmob)](https://github.com/gefgu/fastmob/blob/main/LICENSE)
[![GitHub](https://img.shields.io/github/stars/gefgu/fastmob?style=social)](https://github.com/gefgu/fastmob)

## Relationship to scikit-mobility

fastmob follows the measure API of
[scikit-mobility](https://github.com/scikit-mobility/scikit-mobility) and
reimplements it on a Rust core with Narwhals dataframe support. Comparison
tests against scikit-mobility run wherever the APIs overlap. Beyond that,
fastmob has its own trajectory hierarchy, models and network tools.

## License and credits

fastmob is released under the
[BSD 3-Clause license](https://github.com/gefgu/fastmob/blob/main/LICENSE).
Third-party code and algorithms, including MoveTK (Apache-2.0), are credited in
[THIRD_PARTY_NOTICES.md](https://github.com/gefgu/fastmob/blob/main/fastmob-core/THIRD_PARTY_NOTICES.md).

## Cite fastmob

```bibtex
@software{fastmob,
  author = {dos Santos, Gustavo Henrique Bruno},
  title  = {fastmob: fast human-mobility analysis in Python},
  year   = {2026},
  version = {0.3.2},
  url    = {https://github.com/gefgu/fastmob}
}
```

## Maintainer

Gustavo Henrique Bruno dos Santos ([@gefgu](https://github.com/gefgu)).

## Contribute and report issues

- [Source code and issue tracker](https://github.com/gefgu/fastmob/issues)
- [Development setup](development.md)

A useful bug report includes:

- the fastmob and Python versions
- the dataframe backend and its version
- a minimal input
- the expected and the actual output
- the complete traceback

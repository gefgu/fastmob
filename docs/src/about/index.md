# About Fastkit-Mobility

Fastkit-Mobility is a Python library for human-mobility analysis. It pairs a
familiar dataframe-facing API with Rust kernels for the compute-heavy parts of
trajectory processing and measurement. See [Features](../features/index.md)
for what it can do.

[![PyPI](https://img.shields.io/pypi/v/fastmob)](https://pypi.org/project/fastmob/)
[![Python](https://img.shields.io/pypi/pyversions/fastmob)](https://pypi.org/project/fastmob/)
[![License](https://img.shields.io/pypi/l/fastmob)](https://github.com/gefgu/fastmob/blob/main/LICENSE)
[![GitHub](https://img.shields.io/github/stars/gefgu/fastmob?style=social)](https://github.com/gefgu/fastmob)

Released 2026-10-28 ([release notes](../release_notes/index.md)) · Alpha · pandas and Polars via Narwhals

## Relationship to scikit-mobility

Fastkit-Mobility follows the measure API of
[scikit-mobility](https://github.com/scikit-mobility/scikit-mobility) and
reimplements it on a Rust core with Narwhals dataframe support. Comparison
tests against scikit-mobility run wherever the APIs overlap, and it is up to 2658x faster on 4M Brightkite rows (see [benchmarks](../features/benchmarks.md)). Beyond that,
Fastkit-Mobility has its own trajectory hierarchy, models and network tools.

## License and credits

Fastkit-Mobility is released under the
[BSD 3-Clause license](https://github.com/gefgu/fastmob/blob/main/LICENSE).
Third-party code and algorithms, including MoveTK (Apache-2.0), are credited in
[THIRD_PARTY_NOTICES.md](https://github.com/gefgu/fastmob/blob/main/fastmob-core/THIRD_PARTY_NOTICES.md).

## Cite Fastkit-Mobility

If you use Fastkit-Mobility in research, please cite the paper
([HAL hal-05755990](https://inria.hal.science/hal-05755990)):

```bibtex
@inproceedings{santos2026fastkit,
  author    = {Santos, Gustavo H. and
               Viana, Aline Carneiro and
               Pappalardo, Luca and
               Silva, Thiago H.},
  title     = {Fastkit-Mobility: A
               High-Performance,
               DataFrame-Agnostic Library
               for Mobility Analysis},
  booktitle = {NetMob 2026},
  address   = {Niter{\'o}i, Brazil},
  year      = {2026},
  month     = oct,
  url       = {https://inria.hal.science/
               hal-05755990}
}
```

To cite a specific release of the software, use:

```bibtex
@software{fastmob,
  author  = {Santos, Gustavo H.},
  title   = {Fastkit-Mobility},
  note    = {Fast human-mobility
             analysis in Python.
             PyPI package fastmob},
  year    = {2026},
  version = {0.3.4},
  url     = {https://github.com/
             gefgu/fastmob}
}
```

## Maintainer

Gustavo Henrique Bruno dos Santos ([@gefgu](https://github.com/gefgu)).

## Contribute and report issues

- [Source code and issue tracker](https://github.com/gefgu/fastmob/issues)
- [Development setup](development.md)

A useful bug report includes the Fastkit-Mobility and Python versions, the dataframe backend and its version, a minimal input, the expected and actual output, and the full traceback.

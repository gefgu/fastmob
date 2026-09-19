# Third-Party Notices and Provenance

`fastmob-core` is released under the BSD-3-Clause license in `LICENSE`.
The following source modules document an upstream implementation that informed
or was ported into this crate. This file records the provenance required for
release review; it must remain in source and binary distributions.

## MovingPandas

- Project: https://github.com/movingpandas/movingpandas
- License: BSD 3-Clause License
- Copyright: 2018-2026 MovingPandas developers
- Affected modules: trajectory smoothing; segmentation; selected trajectory
  simplification and Haversine helpers.
- Pinned version: `movingpandas==0.22.4` (per
  `fastmob_benchmarks/movingpandas/requirements.txt`).

MovingPandas' BSD-3-Clause notice requires its copyright notice, conditions,
and disclaimer to be retained in redistributions of derivative source or
binary code. The upstream license text is available at
https://github.com/movingpandas/movingpandas/blob/main/LICENSE.txt.

## Tracktable

- Project: https://github.com/sandialabs/tracktable
- License: BSD 3-Clause License
- Copyright: 2014-2023 National Technology and Engineering Solutions of
  Sandia, LLC; the U.S. Government retains certain rights under Contract
  DE-NA0003525.
- Affected module: `src/trajectory/shape_signature.rs`.
- Pinned version: `tracktable==1.7.3` (per
  `fastmob_benchmarks/tracktable/requirements.txt`).

Tracktable's BSD-3-Clause notice must be retained for any derivative code.
Its license text is available at
https://github.com/sandialabs/tracktable/blob/main/LICENSE.txt.

## MoveTK

- Project: https://github.com/movetk/movetk
- License: Apache License 2.0
- Copyright: 2018-2020 HERE Europe B.V.; Utrecht University (The
  Netherlands); TU/e (The Netherlands). Per the identical per-file header on
  `GreedyOutlierDetector.h`, `SmartGreedyOutlierDetector.h`, and
  `ZhengOutlierDetector.h` (`src/include/movetk/outlierdetection/`), authored
  by Mees van de Kerkhof, Bram Custers, and Kevin Verbeek, modified by Aniket
  Mitra. MoveTK's repository carries no separate `NOTICE` file.
- Referenced upstream implementation: `GreedyOutlierDetector.h`,
  `SmartGreedyOutlierDetector.h`, and `ZhengOutlierDetector.h`
  (`movetk::outlierdetection`), translated into Rust.
- Affected modules: `src/preprocessing/outliers/greedy.rs`,
  `src/preprocessing/outliers/zheng.rs`.
- Pinned revision: commit `facd6babd3f6915ccfb92ae405347da48ae5e98d`
  (`master`, 2023-03-14, `git describe`: `v0.7.0-beta-340-gfacd6ba`) — the
  checkout at `fastmob_benchmarks/movetk/src-repo`, remote
  `https://github.com/movetk/movetk.git`.

`greedy.rs` and `zheng.rs` are translations of this MoveTK C++ source, so
Apache-2.0 §4 applies in full: this crate's redistribution must (a) include a
copy of the Apache License, Version 2.0 itself (not just a link — a
`LICENSE-APACHE-MOVETK` file, or the full text appended here, needs adding
before publish), (b) carry a notice that `greedy.rs`/`zheng.rs` are modified
from the original C++, and (c) retain the copyright/attribution lines above.

## PTRAIL

- Project: https://github.com/YakshHaranwala/PTRAIL
- License: BSD 3-Clause License
- Copyright: 2021, PTRAIL Developers.
- Referenced upstream implementation: `Filters.hampel_outlier_detection`
  (which itself delegates to the `hampel` PyPI package, see below),
  `Interpolation.interpolate_position`, and the `kinematic_help` coefficient
  derivation, translated into Rust.
- Affected modules: `src/preprocessing/outliers/hampel.rs`,
  `src/trajectory/interpolate.rs`.
- Pinned version: `ptrail==1.0` (per
  `fastmob_benchmarks/ptrail/requirements.txt`; installed at
  `fastmob_benchmarks/ptrail/.venv`).

PTRAIL's BSD-3-Clause notice requires its copyright notice, conditions, and
disclaimer to be retained in redistributions of derivative source or binary
code.

## `hampel` (PyPI package)

- Project: https://github.com/MichaelisTrofficus/hampel_filter
- License: MIT License
- Copyright: 2018, The Python Packaging Authority (this is the copyright
  line actually committed in the upstream `LICENSE` file — apparently an
  unedited packaging-tutorial template rather than the author's own name;
  confirmed via two independent fetches of that file, so record it as-is
  rather than substitute a guessed holder).
- Referenced upstream implementation: the package's Cython kernel
  (`hampel.extension.hampel`), translated into Rust.
- Affected module: `src/preprocessing/outliers/hampel.rs`.
- Pinned version: `hampel==1.0.2` (per
  `fastmob_benchmarks/ptrail/requirements.txt`, where PTRAIL pulls it in as
  a dependency; installed at `fastmob_benchmarks/ptrail/.venv`).

The MIT license requires the above copyright notice and the permission
notice to be included in all copies or substantial portions of the
Software.

## HuMobi

- Project: https://github.com/SmolakK/HuMobi
- License: BSD 3-Clause License
- Copyright: 2021, HuMobi.
- Referenced upstream implementation: `predictors.markov.MarkovChain` and
  `predictors.wrapper.TopLoc`, translated into Rust.
- Affected module: `src/models/next_location.rs`.
- Pinned version: `humobi==0.1.17` (per
  `fastmob_benchmarks/humobi/requirements.txt`; installed at
  `fastmob_benchmarks/humobi/.venv`).

HuMobi's BSD-3-Clause notice requires its copyright notice, conditions, and
disclaimer to be retained in redistributions of derivative source or binary
code.

## Release review

Provenance is now confirmed for every `ported from`/`adapted from`/`ported to
match` statement under `src/`: all five sources above (MovingPandas,
Tracktable, MoveTK, PTRAIL, `hampel`, HuMobi) were translated from upstream
source rather than independently reimplemented, and each entry above records
the exact upstream revision (a pinned git commit for MoveTK, pinned package
versions for the rest) per the checkouts/installs in `fastmob_benchmarks`.
Remaining pre-publish checklist items:

- Add a full copy of the Apache License, Version 2.0 to this crate's
  distributed contents for the MoveTK entry — the only non-BSD/MIT license
  here, and the only one requiring the license text itself to be included
  rather than just its notice/disclaimer retained.
- If any further `ported from`/`adapted from` statement is added to `src/`
  later, add a matching entry here before the next release.

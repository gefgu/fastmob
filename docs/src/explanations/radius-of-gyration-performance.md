# Why Radius of Gyration Is 447× Faster

skmob2 computes `radius_of_gyration` on 4 million trajectory points in about **72 ms**. The original scikit-mobility takes **32 seconds** on the same data — a 447× difference. This page explains exactly where that gap comes from, layer by layer.

---

## The measure

Radius of gyration quantifies how far a person typically roams from their most-frequented location. For a user \(u\) with \(n\) recorded positions:

\[
r_g(u) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} d_{\text{Haversine}}(r_i,\, r_{\text{cm}})^2}
\]

where \(r_{\text{cm}}\) is the arithmetic mean of the user's latitude and longitude coordinates. The formula is order-independent: only which points belong to a user matters, not the sequence they were visited.

---

## The full pipeline at a glance

<div style="text-align:center; margin: 1.5rem 0;">
<svg viewBox="0 0 640 548" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="skmob2 data-flow pipeline for radius_of_gyration" style="width:100%;max-width:640px;font-family:Inter,sans-serif;">
<defs>
  <marker id="pipe-arr" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#546E7A"/>
  </marker>
</defs>

<!-- Python top zone -->
<rect x="8" y="4" width="624" height="200" rx="10" fill="#E3F2FD" stroke="#1565C0" stroke-width="1.5" stroke-dasharray="5,3"/>
<text x="20" y="21" font-size="11" fill="#1565C0" font-weight="600">Python</text>

<!-- Box A: Input DataFrame -->
<rect x="120" y="26" width="400" height="44" rx="6" fill="#1976D2"/>
<text x="320" y="44" text-anchor="middle" font-size="13" fill="white" font-weight="600">Input DataFrame</text>
<text x="320" y="62" text-anchor="middle" font-size="11" fill="#BBDEFB">pandas · Polars · any eager Narwhals-compatible backend</text>

<!-- Arrow A→B -->
<line x1="320" y1="70" x2="320" y2="90" stroke="#546E7A" stroke-width="2" marker-end="url(#pipe-arr)"/>

<!-- Box B: Narwhals + column detection -->
<rect x="120" y="92" width="400" height="44" rx="6" fill="#1565C0"/>
<text x="320" y="110" text-anchor="middle" font-size="13" fill="white" font-weight="600">Narwhals wrap · column detection</text>
<text x="320" y="127" text-anchor="middle" font-size="11" fill="#BBDEFB">detect datetime / lat / lng / uid from column-name priority list</text>

<!-- Arrow B→C -->
<line x1="320" y1="136" x2="320" y2="154" stroke="#546E7A" stroke-width="2" marker-end="url(#pipe-arr)"/>

<!-- Box C: Zero-copy extraction -->
<rect x="120" y="156" width="400" height="44" rx="6" fill="#0D47A1"/>
<text x="320" y="174" text-anchor="middle" font-size="13" fill="white" font-weight="600">Zero-copy lat/lng extraction</text>
<text x="320" y="191" text-anchor="middle" font-size="11" fill="#BBDEFB">NumPy buffer (pandas) · Arrow buffer (Polars) — no Python-side copy</text>

<!-- Arrow C→Rust (crosses FFI line) -->
<line x1="320" y1="200" x2="320" y2="250" stroke="#546E7A" stroke-width="2" marker-end="url(#pipe-arr)"/>

<!-- FFI boundary 1 -->
<line x1="48" y1="212" x2="592" y2="212" stroke="#78909C" stroke-width="1.5" stroke-dasharray="5,3"/>
<text x="320" y="224" text-anchor="middle" font-size="10" fill="#78909C">PyO3 FFI boundary</text>

<!-- Rust zone -->
<rect x="8" y="230" width="624" height="218" rx="10" fill="#FFF3E0" stroke="#E65100" stroke-width="1.5" stroke-dasharray="5,3"/>
<text x="20" y="247" font-size="11" fill="#E65100" font-weight="600">Rust kernel (compiled + parallel)</text>

<!-- Box D: Build valid user index ranges -->
<rect x="80" y="252" width="480" height="58" rx="6" fill="#E65100"/>
<text x="320" y="271" text-anchor="middle" font-size="13" fill="white" font-weight="600">Build valid user index ranges</text>
<text x="320" y="288" text-anchor="middle" font-size="11" fill="#FFE0B2">filter null lat/lng · sort index array (not the full DataFrame) · emit (start, end) per user</text>
<text x="320" y="303" text-anchor="middle" font-size="11" fill="#FFE0B2">O(n log n) index sort — avoids O(n) full-DataFrame materialization</text>

<!-- Arrow D→E -->
<line x1="320" y1="310" x2="320" y2="328" stroke="#546E7A" stroke-width="2" marker-end="url(#pipe-arr)"/>

<!-- Box E: Rayon parallel kernel -->
<rect x="80" y="330" width="480" height="110" rx="6" fill="#BF360C"/>
<text x="320" y="349" text-anchor="middle" font-size="13" fill="white" font-weight="600">Rayon par_iter — one thread per user</text>

<!-- Inner detail block -->
<rect x="104" y="357" width="432" height="76" rx="4" fill="rgba(255,255,255,0.07)" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>
<text x="320" y="374" text-anchor="middle" font-size="11" fill="#FFCCBC" font-family="'Fira Code',monospace">1. fold(0,0) over user's rows → (lat_sum, lng_sum)   ← one sequential pass</text>
<text x="320" y="392" text-anchor="middle" font-size="11" fill="#FFCCBC" font-family="'Fira Code',monospace">2. cm_lat = lat_sum / n,  cm_lng = lng_sum / n</text>
<text x="320" y="410" text-anchor="middle" font-size="11" fill="#FFCCBC" font-family="'Fira Code',monospace">3. cos_cm_lat = cos(cm_lat_rad)   ← cached once per user</text>
<text x="320" y="428" text-anchor="middle" font-size="11" fill="#FFCCBC" font-family="'Fira Code',monospace">4. map rows → Haversine(pt, cm)²  →  √(Σ d² / n)</text>

<!-- Arrow Rust→Python (crosses FFI line) -->
<line x1="320" y1="440" x2="320" y2="492" stroke="#546E7A" stroke-width="2" marker-end="url(#pipe-arr)"/>

<!-- FFI boundary 2 -->
<line x1="48" y1="452" x2="592" y2="452" stroke="#78909C" stroke-width="1.5" stroke-dasharray="5,3"/>
<text x="320" y="464" text-anchor="middle" font-size="10" fill="#78909C">PyO3 FFI boundary</text>

<!-- Python bottom zone -->
<rect x="8" y="470" width="624" height="70" rx="10" fill="#E3F2FD" stroke="#1565C0" stroke-width="1.5" stroke-dasharray="5,3"/>
<text x="20" y="487" font-size="11" fill="#1565C0" font-weight="600">Python</text>

<!-- Box F: Assemble result -->
<rect x="120" y="490" width="400" height="44" rx="6" fill="#1976D2"/>
<text x="320" y="508" text-anchor="middle" font-size="13" fill="white" font-weight="600">Assemble result DataFrame</text>
<text x="320" y="524" text-anchor="middle" font-size="11" fill="#BBDEFB">uid labels + radius_of_gyration values · returned in caller's original backend</text>
</svg>
</div>

The two FFI boundary crossings are cheap: PyO3 marshals array pointers, not data. The real work happens entirely inside the Rust kernel.

---

## How the original skmob worked

skmob's implementation follows the natural Python approach: iterate over users with `groupby`, compute the center of mass with NumPy `.mean()`, then call a Haversine helper per point:

```python
# skmob approach (representative pseudocode)
def radius_of_gyration(tdf):
    results = []
    for uid, group in tdf.groupby("uid"):       # ① GIL held throughout
        lats = group["lat"].values
        lngs = group["lng"].values
        cm_lat = lats.mean()                    # ② array pass 1
        cm_lng = lngs.mean()                    # ② array pass 2
        dists = np.array([                      # ③ Python loop, per-point calls
            haversine(lat, lng, cm_lat, cm_lng)
            for lat, lng in zip(lats, lngs)
        ])
        rog = np.sqrt((dists ** 2).mean())      # ④ two more array passes
        results.append((uid, rog))
    return results
```

Four costs compound here:

1. **The GIL holds across all users.** `groupby` iteration is sequential Python. Other threads cannot run numerical code in parallel.
2. **Multiple array passes per user.** Computing `lats.mean()`, then `lngs.mean()`, then squaring distances, then averaging them again allocates temporary arrays on each pass.
3. **Per-point Python function calls.** The Haversine function is called once per trajectory point from Python, paying interpreter overhead 4 million times at 4M rows.
4. **Full DataFrame sort as a precondition.** skmob requires a `TrajDataFrame` with rows pre-sorted by user and time. Sorting a 4M-row DataFrame of strings, timestamps, and floats is expensive before the measure even starts.

---

## How skmob2 works

### One-pass center of mass

The center of mass needs only the sum of latitudes and the sum of longitudes. Rust's `fold` accumulates both in a single sequential scan:

```rust
// src/radius_of_gyration.rs, rog_for_parallel_slices
let (lat_sum, lng_sum) = (start..end).fold((0.0f64, 0.0f64), |(ls, ns), idx| {
    (ls + latitudes[idx], ns + longitudes[idx])
});
let cm_lat = lat_sum / n as f64;
let cm_lng = lng_sum / n as f64;
```

Two values accumulate in one loop instead of two separate `mean()` calls. The CPU prefetches the next cache line while arithmetic finishes on the current one.

### Cached trigonometry

Every Haversine call for a given user shares the same center of mass. The latitude-derived trig values are computed once and reused:

```rust
let cm_lat_rad = cm_lat.to_radians();
let cm_lng_rad = cm_lng.to_radians();
let cos_cm_lat = cm_lat_rad.cos();          // computed once per user

let sum_sq: f64 = (start..end).map(|idx| {
    let lat_rad = latitudes[idx].to_radians();
    let dlat = lat_rad - cm_lat_rad;
    let dlng = longitudes[idx].to_radians() - cm_lng_rad;
    let a = (dlat / 2.0).sin().powi(2)
          + cos_cm_lat * lat_rad.cos() * (dlng / 2.0).sin().powi(2);
    let d = 2.0 * a.sqrt().asin() * 6371.0088;
    d * d
}).sum();
```

At 4M rows distributed across thousands of users, saving one `cos()` call per user is negligible. What matters is that the inner loop is entirely compiled arithmetic — no Python object allocations, no GIL, no interpreter dispatch.

### Rayon parallel dispatch

Once the user index ranges are built, all users are dispatched in one call:

```rust
// radius_of_gyration_batch_impl
let results: Vec<f64> = valid_ranges
    .par_iter()                             // Rayon work-stealing thread pool
    .map(|&(start, end)| {
        rog_for_parallel_slices(&valid_latitudes, &valid_longitudes, start, end)
    })
    .collect();
```

Each user slice is an independent unit of work. Rayon's work-stealing scheduler fills all available CPU cores. On a typical development machine with 8–16 cores, this alone multiplies throughput.

### Index-based grouping, not DataFrame sorting

skmob2 never sorts the input DataFrame. Instead, it builds a **sorted index array** — a `Vec<usize>` whose values point into the original coordinate arrays:

```rust
// Build sorted row-index vector in-place
indices.sort_by(|&l, &r| values[l].cmp(&values[r]).then(l.cmp(&r)));
```

Sorting 4M `usize` values costs roughly 4M × 32 bytes of data movement. Sorting a pandas DataFrame of the same size moves 4M rows of mixed types across multiple columns — an order of magnitude more data.

### Zero-copy array views

On the pandas path, `PyReadonlyArray1::as_slice()` returns a `&[f64]` reference directly into the NumPy buffer. No copy is made moving data into Rust. On the Polars path, the same happens via Arrow buffers. The only allocation before the kernel runs is the index array itself.

---

## The speedup in numbers

<div id="rog-chart-wrap" style="position:relative; width:100%; max-width:700px; margin:1.5rem auto 0.5rem;">
<svg id="rog-chart-svg" viewBox="0 0 700 380" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Bar chart comparing skmob and skmob2 radius of gyration benchmark times on a linear scale" style="width:100%;">

<!-- Background -->
<rect x="0" y="0" width="700" height="380" fill="transparent"/>

<!-- Y grid lines (linear: 5s, 10s, 15s, 20s, 25s, 30s, 35s) -->
<line x1="68" y1="287" x2="678" y2="287" stroke="#ECEFF1" stroke-width="1"/>
<line x1="68" y1="245" x2="678" y2="245" stroke="#ECEFF1" stroke-width="1"/>
<line x1="68" y1="204" x2="678" y2="204" stroke="#ECEFF1" stroke-width="1"/>
<line x1="68" y1="162" x2="678" y2="162" stroke="#ECEFF1" stroke-width="1"/>
<line x1="68" y1="121" x2="678" y2="121" stroke="#ECEFF1" stroke-width="1"/>
<line x1="68" y1="79" x2="678" y2="79" stroke="#ECEFF1" stroke-width="1"/>
<line x1="68" y1="38" x2="678" y2="38" stroke="#ECEFF1" stroke-width="1"/>

<!-- Axis lines -->
<line x1="68" y1="328" x2="678" y2="328" stroke="#B0BEC5" stroke-width="1.5"/>
<line x1="68" y1="38" x2="68" y2="328" stroke="#B0BEC5" stroke-width="1.5"/>

<!-- Y axis labels -->
<text x="63" y="332" text-anchor="end" font-size="11" fill="#78909C" font-family="Inter,sans-serif">0</text>
<text x="63" y="291" text-anchor="end" font-size="11" fill="#78909C" font-family="Inter,sans-serif">5 s</text>
<text x="63" y="249" text-anchor="end" font-size="11" fill="#78909C" font-family="Inter,sans-serif">10 s</text>
<text x="63" y="208" text-anchor="end" font-size="11" fill="#78909C" font-family="Inter,sans-serif">15 s</text>
<text x="63" y="166" text-anchor="end" font-size="11" fill="#78909C" font-family="Inter,sans-serif">20 s</text>
<text x="63" y="125" text-anchor="end" font-size="11" fill="#78909C" font-family="Inter,sans-serif">25 s</text>
<text x="63" y="83" text-anchor="end" font-size="11" fill="#78909C" font-family="Inter,sans-serif">30 s</text>
<text x="63" y="42" text-anchor="end" font-size="11" fill="#78909C" font-family="Inter,sans-serif">35 s</text>

<!-- Y axis title -->
<text transform="rotate(-90)" x="-183" y="14" text-anchor="middle" font-size="12" fill="#546E7A" font-family="Inter,sans-serif">average time (seconds)</text>

<!-- ══ GROUP 0: 1K rows ══ -->
<!-- skmob  5.6ms → linear → barH=2 (floor), barY=326 -->
<rect class="bar-skmob" x="82" y="326" width="44" height="2" fill="#EF5350" rx="1"
      data-label="1K — skmob" data-ms="5.6" data-note="At this scale the FFI setup cost offsets Rust's gains"/>
<!-- skmob2 6.3ms → linear → barH=2 (floor), barY=326 -->
<rect class="bar-skmob2" x="132" y="326" width="44" height="2" fill="#1976D2" rx="1"
      data-label="1K — skmob2" data-ms="6.3" data-speedup="~1&times;" data-note="FFI overhead dominates at tiny sizes; crossover comes at ~10K rows"/>
<!-- ~1× label (gray, parity zone) -->
<text x="129" y="316" text-anchor="middle" font-size="10" fill="#90A4AE" font-family="Inter,sans-serif">~1×</text>
<!-- X label -->
<text x="129" y="346" text-anchor="middle" font-size="12" fill="#546E7A" font-family="Inter,sans-serif">1K</text>

<!-- ══ GROUP 1: 10K rows ══ -->
<!-- skmob  32.5ms → linear → barH=2 (floor), barY=326 -->
<rect class="bar-skmob" x="204" y="326" width="44" height="2" fill="#EF5350" rx="1"
      data-label="10K — skmob" data-ms="32.5" data-note="Python groupby overhead scales linearly with rows"/>
<!-- skmob2 6.4ms → linear → barH=2 (floor), barY=326 -->
<rect class="bar-skmob2" x="254" y="326" width="44" height="2" fill="#1976D2" rx="1"
      data-label="10K — skmob2" data-ms="6.4" data-speedup="5&times;" data-note="Rust kernel already near its fixed overhead floor"/>
<text x="251" y="316" text-anchor="middle" font-size="11" fill="#2E7D32" font-weight="bold" font-family="Inter,sans-serif">5×</text>
<text x="251" y="346" text-anchor="middle" font-size="12" fill="#546E7A" font-family="Inter,sans-serif">10K</text>

<!-- ══ GROUP 2: 100K rows ══ -->
<!-- skmob  383ms → linear → barH=3, barY=325 -->
<rect class="bar-skmob" x="326" y="325" width="44" height="3" fill="#EF5350" rx="1"
      data-label="100K — skmob" data-ms="383" data-note="skmob time grows O(n) with the dataset"/>
<!-- skmob2 7.9ms → linear → barH=2 (floor), barY=326 -->
<rect class="bar-skmob2" x="376" y="326" width="44" height="2" fill="#1976D2" rx="1"
      data-label="100K — skmob2" data-ms="7.9" data-speedup="49&times;" data-note="Rayon parallelism keeps skmob2 near its floor"/>
<text x="373" y="315" text-anchor="middle" font-size="11" fill="#2E7D32" font-weight="bold" font-family="Inter,sans-serif">49×</text>
<text x="373" y="346" text-anchor="middle" font-size="12" fill="#546E7A" font-family="Inter,sans-serif">100K</text>

<!-- ══ GROUP 3: 1M rows ══ -->
<!-- skmob  4770ms → linear → barH=40, barY=288 -->
<rect class="bar-skmob" x="448" y="288" width="44" height="40" fill="#EF5350" rx="1"
      data-label="1M — skmob" data-ms="4770" data-note="4.77 seconds: 243× slower than skmob2 at this size"/>
<!-- skmob2 19.6ms → linear → barH=2 (floor), barY=326 -->
<rect class="bar-skmob2" x="498" y="326" width="44" height="2" fill="#1976D2" rx="1"
      data-label="1M — skmob2" data-ms="19.6" data-speedup="243&times;" data-note="Still under 20 ms with parallelism across thousands of users"/>
<text x="495" y="280" text-anchor="middle" font-size="11" fill="#2E7D32" font-weight="bold" font-family="Inter,sans-serif">243×</text>
<text x="495" y="346" text-anchor="middle" font-size="12" fill="#546E7A" font-family="Inter,sans-serif">1M</text>

<!-- ══ GROUP 4: 4M rows (Brightkite full dataset) ══ -->
<!-- skmob  32028ms → linear → barH=265, barY=63 -->
<rect class="bar-skmob" x="570" y="63" width="44" height="265" fill="#EF5350" rx="1"
      data-label="4M — skmob" data-ms="32028" data-note="32 seconds on the full Brightkite check-in dataset"/>
<!-- skmob2 71.6ms → linear → barH=2 (floor), barY=326 -->
<rect class="bar-skmob2" x="620" y="326" width="44" height="2" fill="#1976D2" rx="1"
      data-label="4M — skmob2" data-ms="71.6" data-speedup="447&times;" data-note="71 ms — 447× faster on the full Brightkite check-in dataset"/>
<text x="617" y="53" text-anchor="middle" font-size="12" fill="#2E7D32" font-weight="bold" font-family="Inter,sans-serif">447×</text>
<text x="617" y="346" text-anchor="middle" font-size="12" fill="#546E7A" font-family="Inter,sans-serif">4M</text>

<!-- X axis label -->
<text x="373" y="368" text-anchor="middle" font-size="12" fill="#546E7A" font-family="Inter,sans-serif">dataset size (rows)</text>

<!-- Legend -->
<rect x="68" y="12" width="14" height="14" fill="#EF5350" rx="2"/>
<text x="86" y="23" font-size="11" fill="#546E7A" font-family="Inter,sans-serif">skmob (original)</text>
<rect x="195" y="12" width="14" height="14" fill="#1976D2" rx="2"/>
<text x="213" y="23" font-size="11" fill="#546E7A" font-family="Inter,sans-serif">skmob2</text>
<text x="450" y="23" text-anchor="middle" font-size="10" fill="#90A4AE" font-family="Inter,sans-serif" font-style="italic">hover bars for details</text>

</svg>
<div id="rog-chart-tip" style="position:absolute;display:none;background:#263238;color:#ECEFF1;padding:8px 12px;border-radius:6px;font-size:12px;font-family:Inter,sans-serif;line-height:1.5;pointer-events:none;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,0.35);max-width:260px;white-space:normal;"></div>
</div>

<script>
(function () {
  function fmtMs(ms) {
    if (ms >= 1000) return (ms / 1000).toFixed(2) + ' s';
    if (ms >= 1) return ms.toFixed(1) + ' ms';
    return ms.toFixed(2) + ' ms';
  }
  var tip = document.getElementById('rog-chart-tip');
  var wrap = document.getElementById('rog-chart-wrap');
  if (!tip || !wrap) return;
  var bars = wrap.querySelectorAll('.bar-skmob, .bar-skmob2');
  bars.forEach(function (bar) {
    bar.style.cursor = 'default';
    bar.addEventListener('mouseenter', function () {
      var label = bar.getAttribute('data-label') || '';
      var ms = parseFloat(bar.getAttribute('data-ms') || '0');
      var speedup = bar.getAttribute('data-speedup') || '';
      var note = bar.getAttribute('data-note') || '';
      var isSkmob2 = bar.classList.contains('bar-skmob2');
      var html = '<strong>' + label + '</strong><br>avg time: ' + fmtMs(ms);
      if (speedup) html += '<br>speedup: <strong>' + speedup + '</strong>';
      if (note) html += '<br><em style="color:#90A4AE">' + note + '</em>';
      tip.innerHTML = html;
      tip.style.display = 'block';
    });
    bar.addEventListener('mousemove', function (e) {
      var rect = wrap.getBoundingClientRect();
      var x = e.clientX - rect.left + 14;
      var y = e.clientY - rect.top - 10;
      if (x + 260 > rect.width) x = e.clientX - rect.left - 280;
      tip.style.left = x + 'px';
      tip.style.top = y + 'px';
    });
    bar.addEventListener('mouseleave', function () {
      tip.style.display = 'none';
    });
  });
})();
</script>

| Dataset | skmob | skmob2 (pandas) | Speedup |
|---------|------:|----------------:|--------:|
| 1K rows | 5.6 ms | 6.3 ms | ~1× |
| 10K rows | 32.5 ms | 6.4 ms | **5×** |
| 100K rows | 383 ms | 7.9 ms | **49×** |
| 1M rows | 4.77 s | 19.6 ms | **243×** |
| 4M rows | 32.0 s | 71.6 ms | **447×** |

*Measured on the Brightkite check-in dataset. 5 iterations each, average reported. Machine: Linux 6.8, Intel CPU. skmob uses a pre-built `TrajDataFrame` to exclude I/O time.*

---

## Why the speedup grows with dataset size

At **1K rows** the speedup is roughly 1×. The Rust path has fixed overhead — PyO3 entry, building the index array, allocating result buffers. That overhead costs about the same 6 ms regardless of dataset size.

At **10K rows**, skmob's Python loop starts to dominate. skmob2 is still paying near its fixed cost.

Beyond **100K rows**, skmob scales linearly in both data size and the number of users. Each additional user is another iteration of the Python groupby loop, another set of temporary NumPy allocations, another batch of Python function calls into the Haversine helper. The cost grows with \(O(n)\).

skmob2 scales sub-linearly because:

- The compiled inner loop has no per-row Python overhead. Going from 100K to 4M rows (40×) takes only about 9× longer (7.9 ms → 71.6 ms), not 40×.
- Rayon dispatches all users in one call. Adding more users adds work units to the thread pool, which absorbs them in parallel until all CPU cores are saturated.
- Memory access is sequential. The lat/lng arrays are contiguous `f64` buffers. Hardware prefetch pipelines work efficiently on sequential scans.

---

## Why the optimizations multiply

Each optimization addresses a different bottleneck. They compose multiplicatively:

| Optimization | Approximate factor |
|---|---|
| Compiled inner loop (no Python interpreter per point) | ~10–50× |
| Single-pass fold (vs multi-pass mean + distance pass) | ~1.5–2× |
| Cached `cos(cm_lat)` (vs recomputed per point) | ~1.1–1.2× |
| Rayon parallelism across users | ~4–16× (core-count dependent) |
| Index-based grouping (vs full DataFrame sort) | ~2–5× |

Taken together the factors multiply rather than add, which is how a moderate improvement at each layer compounds into a 447× end-to-end gain at 4M rows.

The pattern generalises: the largest single-threaded gains come from eliminating Python interpreter overhead in tight loops. Parallelism then amplifies whatever per-core efficiency was already achieved.

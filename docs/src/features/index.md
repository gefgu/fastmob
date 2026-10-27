# Features

Fastkit-Mobility runs the scikit-mobility measure API on a Rust core, on whatever
dataframe you already use.

<div class="stat-strip"><span>623x median speedup on 16 measures</span></div>

<div class="feature-grid">
<a class="feature-card" href="../learn/analyze-a-simple-trajectory/">
<img class="feature-card__icon" src="../assets/images/icons/icon-code.svg" alt="">
<h3>Familiar API →</h3>
<p>Port existing code with an import change.</p>
</a>
<a class="feature-card" href="../learn/use-custom-columns-and-dataframe-backends/">
<img class="feature-card__icon" src="../assets/images/icons/icon-dataframe.svg" alt="">
<h3>Any dataframe&nbsp;→</h3>
<p>pandas, Polars and other Narwhals backends in; the same backend out.</p>
</a>
<div class="feature-card">
<img class="feature-card__icon" src="../assets/images/icons/icon-memory.svg" alt="">
<h3>Low memory</h3>
<p>Kernels read your columns in place instead of copying or sorting whole tables.</p>
</div>
<div class="feature-card">
<img class="feature-card__icon" src="../assets/images/icons/icon-shield.svg" alt="">
<h3>Validated</h3>
<p>Correctness tests on pandas and Polars, plus comparison tests against scikit-mobility. Run them yourself.</p>
</div>
<div class="feature-card">
<img class="feature-card__icon" src="../assets/images/icons/icon-branch.svg" alt="">
<h3>Open source</h3>
<p>BSD-3 licensed. Read the source, file issues, reproduce every benchmark.</p>
</div>
</div>

## Any dataframe in, same backend out

<div class="figure figure-scroll">
<img src="../assets/images/features/features-backend-flow.png" alt="Data moves from pandas or Polars through Narwhals and Arrow buffers into a Rust kernel, then returns as the same backend type">
</div>

## Switching is one import

```python
# before
import skmob.measures.individual as m
# after
import fastmob as m
```

## Speed at a glance

<div class="figure">
<picture>
<source media="(max-width: 600px)" srcset="../assets/images/home/home-benchmark-speedup-mobile.png">
<img src="../assets/images/home/home-benchmark-speedup.png" alt="Bar chart comparing scikit-mobility and fastmob execution time on 4M check-ins for five measures; fastmob is 423x to 2658x faster.">
</picture>
</div>

Average of 5 runs on 4 million Brightkite check-ins, pandas backend. Hardware
and methodology are on the [benchmarks page](benchmarks.md).

## Dig deeper

<div class="feature-grid">
<a class="feature-card" href="benchmarks/">
<img class="feature-card__icon" src="../assets/images/icons/icon-speed.svg" alt="">
<h3>Benchmarks →</h3>
<p>Numbers for every measure family, pandas and Polars.</p>
</a>
<a class="feature-card" href="optimizations/">
<img class="feature-card__icon" src="../assets/images/icons/icon-memory.svg" alt="">
<h3>Optimizations →</h3>
<p>How indexed ranges and batched kernels keep memory low.</p>
</a>
</div>

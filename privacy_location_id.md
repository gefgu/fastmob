# Privacy location IDs

## Goal

Privacy attacks should use a single, stable location key rather than repeatedly
comparing raw latitude/longitude pairs. The default key is an H3 cell ID at
resolution 12; this groups small GPS jitter into the same representative
location while keeping a high spatial granularity.

## Execution model

1. Validate latitude/longitude rows as today.
2. Convert all valid coordinate pairs once with the existing Rust-parallel
   `batch_latlng_to_cells` helper.
3. Pass the resulting H3 `u64` cells into the privacy core as the location
   keys. Do not factorize them: an H3 cell ID is already a stable integer key.
4. Build each user's candidate indexes once:
   - location multiset for location attacks;
   - `(location_id, time_bucket)` multiset for time attacks;
   - ordered location IDs for sequence attacks;
   - location sets and metric maps for the remaining attacks.
5. Compare each small target knowledge instance against those prepared
   candidate indexes, then derive risk as `1 / matching_candidates`.

This removes candidate-side map/set allocation from the pairwise comparison
loop that previously made repeated-user inputs impractical.

## Public API

All privacy functions should expose:

```python
h3_resolution: int = 12
```

- `12` is the default H3 location model.
- Values from `0` through `15` are accepted.
- Exact `(lat, lon)` matching is not supported.

Normal output remains `uid, risk`. In `force_instances=True` mode, return H3
cell-center latitude/longitude as the representative location coordinates.

## scikit-mobility parity

scikit-mobility compares exact coordinates. To compare it with H3-mode
Fastmob fairly, canonicalize the shared test/benchmark input to each H3 cell's
center coordinates before passing it to either implementation. Both libraries
then apply the same location-equivalence relation.

## Validation

- Unit-test H3 keys, time keys, sequences, metrics, null/invalid rows, and
  forced-instance cell-center output.
- Use H3-center-canonicalized scikit-mobility fixtures for the default path.
- Benchmark repeated users and `location_risk(..., knowledge_length=2)` to
  prevent reintroduction of per-candidate allocation in the inner loop.

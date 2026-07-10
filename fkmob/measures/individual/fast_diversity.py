"""Low-level suffix-array diversity primitive."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

try:
    from pydivsufsort import divsufsort as _divsufsort
    from pydivsufsort import kasai as _kasai
except ImportError:
    _divsufsort = None
    _kasai = None


def fast_diversity(sequence) -> float:
    """Compute the diversity of a sequence using suffix arrays.

    Maps elements to integer codes, builds a suffix array and LCP array via
    ``pydivsufsort``, then returns ``distinct_substrings / total_substrings``.

    Parameters
    ----------
    sequence : iterable
        An iterable of hashable elements.

    Returns
    -------
    float
        A value in ``[0, 1]`` where 0 means no diversity (constant sequence)
        and values approaching 1 mean high diversity.

    Raises
    ------
    ImportError
        When ``pydivsufsort`` is not installed.

    Examples
    --------
    >>> from fkmob.measures.individual import fast_diversity
    >>> sequence = ["home", "work", "home", "gym"]
    >>> print(round(fast_diversity(sequence), 3))
    0.9
    """
    if _divsufsort is None:
        raise ImportError("pydivsufsort is required: pip install fkmob[diversity]")

    items = list(sequence)
    n = len(items)
    if n <= 1:
        return 0.0

    # Map distinct elements to int32 codes
    unique_map = {x: i for i, x in enumerate(set(items))}
    int_seq = np.array([unique_map[x] for x in items], dtype=np.int32)

    sa = _divsufsort(int_seq)
    lcp = _kasai(int_seq, sa)

    # Standard formula: sum of (suffix_length - lcp) per suffix array entry
    # gives the total count of distinct substrings.
    distinct_substrings = int(np.sum((n - sa) - lcp))
    total_substrings = n * (n + 1) // 2

    if total_substrings == 0:
        return 0.0

    return distinct_substrings / total_substrings

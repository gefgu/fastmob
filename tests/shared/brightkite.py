from pathlib import Path

# Shared data path — used by both correctness and benchmark suites so the file
# is downloaded only once regardless of which suite runs first.
_BRIGHTKITE_PATH = (
    Path(__file__).parent
    / "data"
    / "loc-brightkite_totalCheckins.txt.gz"
)
_BRIGHTKITE_URL = (
    "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"
)
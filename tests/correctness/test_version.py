"""The public version follows installed distribution metadata."""

import importlib.metadata
import subprocess
import sys

import fastmob


def test_version_matches_installed_distribution():
    assert fastmob.__version__ == importlib.metadata.version("fastmob")


def test_version_fallback_without_distribution_metadata():
    code = '''
import importlib.metadata
real_version = importlib.metadata.version
def version(name):
    if name == "fastmob":
        raise importlib.metadata.PackageNotFoundError(name)
    return real_version(name)
importlib.metadata.version = version
import fastmob
assert fastmob.__version__ == "0+unknown"
'''
    subprocess.run([sys.executable, "-c", code], check=True)

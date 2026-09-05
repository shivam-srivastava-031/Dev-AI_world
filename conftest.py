import pathlib, sys

# Ensure the repo root is importable so `tests.support` resolves regardless of
# how pytest is invoked.
sys.path.insert(0, str(pathlib.Path(__file__).parent))

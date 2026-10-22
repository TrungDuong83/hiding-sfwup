import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts"), os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)
assert os.environ.get("PYTHONHASHSEED") == "0", "run: env PYTHONHASHSEED=0 python -m pytest"

# Pytest path bootstrap: keep imports stable without requiring editable installs during CI.
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

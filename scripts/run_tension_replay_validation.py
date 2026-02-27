from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ask_my_brain import _run_tension_replay_validation


if __name__ == "__main__":
    raise SystemExit(0 if _run_tension_replay_validation() else 1)

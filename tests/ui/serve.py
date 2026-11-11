"""Run the real UI server with a fixed token/port so the browser test can drive it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from strands_evals.cli._ui_server import build_server  # noqa: E402

root, port, token = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
read_only = "--read-only" in sys.argv
server = build_server(root, port=port, token=token, read_only=read_only)
print("ready", flush=True)  # noqa: T201 - the harness waits for this line
server.serve_forever()

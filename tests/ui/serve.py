"""Run real UI servers for the browser tests: each line read asks for one, and the line answered says where it is."""

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from strands_evals.cli._ui_server import build_server  # noqa: E402

for line in sys.stdin:
    asked = json.loads(line)
    server = build_server(Path(asked["root"]), token=asked["token"], read_only=asked.get("read_only", False))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(json.dumps({"port": server.server_address[1]}), flush=True)  # noqa: T201 - the harness reads this line

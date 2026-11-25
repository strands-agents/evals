"""`strands-evals ui`: browse and edit the experiments in a directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path

from .._common import EXIT_BAD_INPUT, EXIT_OK, resolve_format
from .._ui_discovery import DEFAULT_DEPTH, SCAN_SCHEMA_VERSION, discover
from .._ui_server import build_server, finite, is_loopback, server_url

TOKEN_ENV = "STRANDS_EVALS_UI_TOKEN"
# An optimizer run's `state` from the scan, in the words the page uses.
_RUN_STATES = {
    "running": "still running",
    "finished": "finished",
    "halted": "halted by a proposer error",
    "interrupted": "interrupted",
    "unknown": "not finished",
}


def _print_listing(root: Path, args: argparse.Namespace) -> None:
    result = discover(root, args.depth)
    if resolve_format(args) == "json":
        listing = finite({"schema_version": SCAN_SCHEMA_VERSION, "root": str(root.resolve()), **vars(result)})
        try:
            print(json.dumps(listing, indent=2))  # noqa: T201
        except RecursionError:  # a document nested past what laying it out follows: one line has no such limit
            print(json.dumps(listing))  # noqa: T201
        return

    scored = {r["path"]: f"{r['path']} ({r['overall_score']:.2f})" for r in result.reports}
    for experiment in result.experiments:
        made = ", ".join(scored[path] for path in experiment["reports"]) or "(not run)"
        print(f"{experiment['path']} ({experiment['cases']} cases)  ->  {made}")  # noqa: T201
    for report in result.reports:
        if report["experiment"] is None:
            # A report with experiments that fit equally says so: it is not one made in code.
            why = f"not paired: {report['link']['detail']}" if report["link"] else "no experiment"
            print(f"({why})  ->  {scored[report['path']]}")  # noqa: T201
    for run in result.optimizer_runs:
        search = f"{run['candidates']} candidates, {run['iterations']} iterations"
        champion = f"champion {run['champion_id']}" if run.get("champion_id") else "no champion"
        state = _RUN_STATES.get(run["state"], _RUN_STATES["unknown"])
        stopped = f", stopped: {run['stop_reason']}" if run.get("stop_reason") else ""
        print(f"optimizer run {run['dir']} ({search})  ->  {champion}, {state}{stopped}")  # noqa: T201
    for session in result.sessions:
        made = ", ".join(session["diagnoses"]) or "(not diagnosed)"
        print(f"session {session['path']} ({session['traces']} traces, {session['spans']} spans)  ->  {made}")  # noqa: T201
    for diagnosis in result.diagnoses:
        if diagnosis["session"] is None:
            print(f"(no session)  ->  {diagnosis['path']} ({diagnosis['failures']} failures)")  # noqa: T201
    for store in result.stores:
        print(f"task result store {store['dir']} ({store['files']} results)")  # noqa: T201
    for skip in result.skipped:
        print(f"skipped {skip['path']}: {skip['why']}", file=sys.stderr)  # noqa: T201
    if result.truncated:
        print("scan hit the file cap: the rest of DIRECTORY was not read", file=sys.stderr)  # noqa: T201


def _run(args: argparse.Namespace) -> int:
    root = Path(args.directory).expanduser()
    if not root.is_dir():
        print(f"strands-evals: error: {args.directory} is not a directory", file=sys.stderr)  # noqa: T201
        return EXIT_BAD_INPUT

    if args.list:
        _print_listing(root, args)
        return EXIT_OK

    # A supplied token keeps the URL stable across restarts. Opt-in: the default stays random.
    token = args.token or os.environ.get(TOKEN_ENV) or None

    server = build_server(
        root,
        host=args.host,
        port=args.port,
        depth=args.depth,
        read_only=args.read_only,
        token=token,
    )
    url = server_url(server)

    mode = " (read-only)" if args.read_only else ""
    print(f"strands-evals ui{mode}: serving {root.resolve()}", file=sys.stderr)  # noqa: T201
    print(f"  {url}", file=sys.stderr)  # noqa: T201
    print(  # noqa: T201
        "  without a browser: send that token as the X-Evals-Token header; /llms.txt says what is answered",
        file=sys.stderr,
    )
    if token:
        print(  # noqa: T201
            "  that token was supplied, not generated: any process on this machine that guesses"
            f" it can {'read' if args.read_only else 'read and write'} files under DIRECTORY",
            file=sys.stderr,
        )
    else:
        print("  the token in that link is what authorizes the page; Ctrl-C to stop", file=sys.stderr)  # noqa: T201
    if not is_loopback(args.host):
        print(  # noqa: T201
            f"  listening beyond this machine on {args.host}: anyone who can reach it and has that link can"
            f" {'read' if args.read_only else 'read and write'} files under DIRECTORY, over unencrypted HTTP",
            file=sys.stderr,
        )

    if args.browser:
        # A headless box has no browser to open; the URL is already printed either way.
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("", file=sys.stderr)  # noqa: T201
    finally:
        server.shutdown()
        server.server_close()
    return EXIT_OK


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "ui",
        parents=[parent],
        help="browse and edit the experiments in a directory",
        description=(
            "Serve a local page listing every experiment and report found under DIRECTORY, "
            "with a case editor that writes back to the file it loaded. Binds loopback and "
            "gates every request on a token printed with the URL."
        ),
    )
    parser.add_argument(
        "directory",
        metavar="DIRECTORY",
        nargs="?",
        default=".",
        help="directory to scan (default: the current directory)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="port to bind (default: an unused one chosen by the OS)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "address to bind (default: 127.0.0.1). This server reads and writes files in "
            "DIRECTORY, so binding anything other than loopback exposes them to your network. "
            "Use 0.0.0.0 inside a container, or a specific address to serve on one interface."
        ),
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help=f"how many directories deep to scan (default: {DEFAULT_DEPTH})",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="refuse writes; the editor loads cases but cannot save them",
    )
    parser.add_argument(
        "--token",
        help=(
            f"authorize with this token instead of a fresh random one (or set ${TOKEN_ENV}). "
            "With --port, keeps the printed link the same across restarts so that link can be bookmarked, "
            "at the cost of being a secret you chose rather than one nobody can guess. Use a long value, and "
            "prefer --read-only if anyone else can reach loopback on this machine."
        ),
    )
    parser.add_argument(
        "--no-browser",
        dest="browser",
        action="store_false",
        default=True,
        help="do not open a browser, just print the URL",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the discovered experiments and exit instead of serving",
    )
    parser.set_defaults(func=_run)
    return parser

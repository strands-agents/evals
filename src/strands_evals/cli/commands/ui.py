"""`strands-evals ui` — browse and edit the experiments in a directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path

from .._common import EXIT_BAD_INPUT, EXIT_OK, resolve_format
from .._ui_server import DEFAULT_DEPTH, build_server, discover, server_url

TOKEN_ENV = "STRANDS_EVALS_UI_TOKEN"


def _run(args: argparse.Namespace) -> int:
    root = Path(args.directory).expanduser()
    if not root.is_dir():
        print(f"strands-evals: error: {args.directory} is not a directory", file=sys.stderr)  # noqa: T201
        return EXIT_BAD_INPUT

    # --list is the non-interactive half of the same discovery the server exposes, so CI and
    # scripts can ask "what is in this tree" without starting a browser.
    if args.list:
        result = discover(root, args.depth)
        if resolve_format(args) == "json":
            print(json.dumps({"root": str(root.resolve()), "runs": result.runs, "skipped": result.skipped}, indent=2))  # noqa: T201
        else:
            for run in result.runs:
                experiment = run["experiment"]
                report = run["report"]
                left = f"{experiment['path']} ({experiment['cases']} cases)" if experiment else "(no experiment)"
                right = f"{report['path']} ({report['overall_score']:.2f})" if report else "(not run)"
                print(f"{left}  ->  {right}")  # noqa: T201
            for skip in result.skipped:
                print(f"skipped {skip['path']}: {skip['why']}", file=sys.stderr)  # noqa: T201
        return EXIT_OK

    # A fixed token makes the URL survive a restart, which is the difference between a bookmark
    # and copying a fresh link out of the terminal every time. Opt-in: the default stays random.
    token = args.token or os.environ.get(TOKEN_ENV) or None

    runs_dir = Path(args.runs_dir).expanduser() if args.runs_dir else None
    if runs_dir is not None and not runs_dir.is_dir():
        print(f"strands-evals: error: {args.runs_dir} is not a directory", file=sys.stderr)  # noqa: T201
        return EXIT_BAD_INPUT

    server = build_server(
        root,
        host=args.host,
        port=args.port,
        depth=args.depth,
        read_only=args.read_only,
        token=token,
        explorer_root=runs_dir,
    )
    url = server_url(server)

    mode = " (read-only)" if args.read_only else ""
    print(f"strands-evals ui{mode}: serving {root.resolve()}", file=sys.stderr)  # noqa: T201
    print(f"  {url}", file=sys.stderr)  # noqa: T201
    if runs_dir is not None:
        explorer = url.replace("/?t=", "/explorer/?t=")
        print(f"  run explorer over {runs_dir.resolve()} at {explorer}", file=sys.stderr)  # noqa: T201
    if token:
        # Said plainly, because a token you chose is one someone else can guess, and anything
        # that reaches loopback on this machine can then read and write files under DIRECTORY.
        print(  # noqa: T201
            "  that token was supplied, not generated: any process on this machine that guesses"
            f" it can {'read' if args.read_only else 'read and write'} files under DIRECTORY",
            file=sys.stderr,
        )
    else:
        print("  the token in that link is what authorizes the page; Ctrl-C to stop", file=sys.stderr)  # noqa: T201

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
            "DIRECTORY, so binding anything other than loopback exposes them to your network."
        ),
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help=f"how many directories deep to scan (default: {DEFAULT_DEPTH})",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="directory of optimizer run folders (each holding an events.jsonl); serves the run explorer at /explorer",
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
            "Keeps the URL the same across restarts, so it can be bookmarked — at the cost of "
            "being a secret you chose rather than one nobody can guess. Use a long value, and "
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

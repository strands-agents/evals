"""`strands-evals fetch` — pull traces from a provider and emit a Session JSON.

Each provider is a separate sub-subparser under `fetch`. CloudWatch is always
available (boto3 is a hard runtime dep); Langfuse and OpenSearch live behind
optional extras and are imported lazily so users without those extras can still
run the rest of the CLI.
"""

from __future__ import annotations

import argparse

from ...types.evaluation import TaskOutput
from ...types.trace import Session
from .._common import EXIT_BAD_INPUT, EXIT_OK, emit_error
from .._io import write_text_output


def _emit_session(task_output: TaskOutput, session_id: str, output: str | None) -> int:
    """Pull the Session out of a provider TaskOutput and write it as JSON.

    Output is always JSON — Rich rendering of a Session has no useful shape;
    its JSON form is the canonical representation.
    """
    trajectory = task_output.get("trajectory")
    if not isinstance(trajectory, Session):
        raise RuntimeError(
            f"provider returned no Session for session_id='{session_id}' (trajectory type: {type(trajectory).__name__})"
        )
    write_text_output(trajectory.model_dump_json(indent=2), output)
    return EXIT_OK


def _run_cloudwatch(args: argparse.Namespace) -> int:
    # Match OpenSearch's paired-creds check style: validate client-side and
    # exit 2 with a friendly message rather than letting CloudWatchProvider's
    # ProviderError bubble up as a generic exit-3 runtime error.
    if (args.log_group is None) == (args.agent_name is None):
        emit_error("exactly one of --log-group or --agent-name is required")
        return EXIT_BAD_INPUT

    from ...providers.cloudwatch_provider import CloudWatchProvider

    provider = CloudWatchProvider(
        region=args.region,
        log_group=args.log_group,
        agent_name=args.agent_name,
        lookback_days=args.lookback_days,
    )
    return _emit_session(provider.get_evaluation_data(args.session_id), args.session_id, args.output)


def _run_langfuse(args: argparse.Namespace) -> int:
    try:
        # optional extra: langfuse
        from ...providers.langfuse_provider import LangfuseProvider
    except ImportError:
        emit_error("langfuse extra not installed; install with `pip install strands-agents-evals[langfuse]`")
        return EXIT_BAD_INPUT

    provider = LangfuseProvider(host=args.host, timeout=args.timeout)
    return _emit_session(provider.get_evaluation_data(args.session_id), args.session_id, args.output)


def _run_opensearch(args: argparse.Namespace) -> int:
    auth: tuple[str, str] | None = None
    if args.username is not None or args.password is not None:
        if args.username is None or args.password is None:
            emit_error("--username and --password must be provided together")
            return EXIT_BAD_INPUT
        auth = (args.username, args.password)

    # `OpenSearchProvider` lazy-imports `opensearch_genai_observability_sdk_py`
    # inside its constructor (so the provider class itself is always importable).
    # Wrap both the module import and the constructor call in the ImportError
    # guard, scoped tightly so genuine inner ImportErrors aren't masked.
    try:
        # optional extra: opensearch
        from ...providers.opensearch_provider import OpenSearchProvider

        provider = OpenSearchProvider(
            host=args.host,
            index=args.index,
            auth=auth,
            verify_certs=args.verify_certs,
        )
    except ImportError:
        emit_error("opensearch extra not installed; install with `pip install strands-agents-evals[opensearch]`")
        return EXIT_BAD_INPUT

    return _emit_session(provider.get_evaluation_data(args.session_id), args.session_id, args.output)


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "fetch",
        parents=[parent],
        help="fetch traces from a provider and emit a Session JSON",
        description=(
            "Wraps a TraceProvider — one sub-subcommand per backend "
            "(cloudwatch, langfuse, opensearch). The fetched Session JSON "
            "goes to -o PATH or stdout, ready to pipe into "
            "`strands-evals diagnose -`. Credentials come from environment "
            "variables (AWS_REGION, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, "
            "etc.); the CLI does not reimplement auth."
        ),
    )
    sub = parser.add_subparsers(dest="provider", metavar="PROVIDER")
    sub.required = True

    _add_cloudwatch_parser(sub, parent)
    _add_langfuse_parser(sub, parent)
    _add_opensearch_parser(sub, parent)

    return parser


def _add_cloudwatch_parser(
    sub: argparse._SubParsersAction,
    parent: argparse.ArgumentParser,
) -> None:
    cw = sub.add_parser(
        "cloudwatch",
        parents=[parent],
        help="fetch a session from AWS CloudWatch Logs",
        description=(
            "Query CloudWatch Logs Insights for OTEL spans tagged with "
            "session.id and map them to a Session. Either --log-group or "
            "--agent-name must be supplied (agent-name discovers the runtime "
            "log group via describe_log_groups)."
        ),
    )
    cw.add_argument(
        "--session-id",
        required=True,
        metavar="ID",
        help="session id to fetch (matches attributes.session.id)",
    )
    cw.add_argument(
        "--region",
        default=None,
        help="AWS region (default: AWS_REGION / AWS_DEFAULT_REGION env vars, then us-east-1)",
    )
    cw.add_argument(
        "--log-group",
        default=None,
        help="full CloudWatch log group path",
    )
    cw.add_argument(
        "--agent-name",
        default=None,
        help="agent name used to discover the runtime log group via describe_log_groups",
    )
    cw.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="how many days back to search for traces (default: 30)",
    )
    cw.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="write the Session JSON to PATH instead of stdout",
    )
    cw.set_defaults(func=_run_cloudwatch)


def _add_langfuse_parser(
    sub: argparse._SubParsersAction,
    parent: argparse.ArgumentParser,
) -> None:
    lf = sub.add_parser(
        "langfuse",
        parents=[parent],
        help="fetch a session from Langfuse",
        description=(
            "Fetch all traces and observations for a session via the Langfuse "
            "SDK and map them to a Session. Requires the langfuse extra "
            "(`pip install strands-agents-evals[langfuse]`). Credentials read "
            "from LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY env vars."
        ),
    )
    lf.add_argument(
        "--session-id",
        required=True,
        metavar="ID",
        help="session id to fetch (Langfuse trace.list session_id filter)",
    )
    lf.add_argument(
        "--host",
        default=None,
        help="Langfuse API host (default: LANGFUSE_HOST env var, then https://us.cloud.langfuse.com)",
    )
    lf.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="request timeout in seconds (default: 120)",
    )
    lf.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="write the Session JSON to PATH instead of stdout",
    )
    lf.set_defaults(func=_run_langfuse)


def _add_opensearch_parser(
    sub: argparse._SubParsersAction,
    parent: argparse.ArgumentParser,
) -> None:
    os_ = sub.add_parser(
        "opensearch",
        parents=[parent],
        help="fetch a session from OpenSearch",
        description=(
            "Fetch spans for a session via OpenSearchTraceRetriever and map "
            "them to a Session. Requires the opensearch extra "
            "(`pip install strands-agents-evals[opensearch]`)."
        ),
    )
    os_.add_argument(
        "--session-id",
        required=True,
        metavar="ID",
        help="session id to fetch (conversation id)",
    )
    os_.add_argument(
        "--host",
        default="https://localhost:9200",
        help="OpenSearch endpoint URL (default: https://localhost:9200)",
    )
    os_.add_argument(
        "--index",
        default="otel-v1-apm-span-*",
        help="index pattern for span documents (default: otel-v1-apm-span-*)",
    )
    os_.add_argument(
        "--username",
        default=None,
        help="basic auth username (paired with --password); SigV4 auth is library-only",
    )
    os_.add_argument(
        "--password",
        default=None,
        help="basic auth password (paired with --username)",
    )
    verify_group = os_.add_mutually_exclusive_group()
    verify_group.add_argument(
        "--verify-certs",
        dest="verify_certs",
        action="store_true",
        default=True,
        help="verify TLS certificates (default)",
    )
    verify_group.add_argument(
        "--no-verify-certs",
        dest="verify_certs",
        action="store_false",
        help="skip TLS certificate verification (local/dev only)",
    )
    os_.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="write the Session JSON to PATH instead of stdout",
    )
    os_.set_defaults(func=_run_opensearch)

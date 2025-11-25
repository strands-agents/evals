import json
import logging
import os

from langfuse import Langfuse
from langfuse.api.resources.commons.types.trace_with_full_details import TraceWithFullDetails

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_from_json() -> TraceWithFullDetails:
    """Load trace from local JSON file and parse into TraceWithFullDetails."""
    file_path = os.path.join(os.path.dirname(__file__), "trace.json")
    with open(file_path, "r") as f:
        trace_data = json.load(f)
    return TraceWithFullDetails(**trace_data)


def get_langfuse_trace(trace_id: str) -> list[TraceWithFullDetails]:
    # Check if env vars are set, if not fallback to JSON
    if not all(os.environ.get(key) for key in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"]):
        logger.info("Langfuse environment variables not set. Loading from local trace.json")
        return [_load_from_json()]

    try:
        # Initialize Langfuse client
        langfuse = Langfuse(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            host=os.environ.get("LANGFUSE_BASE_URL", ""),
        )
        # Download trace
        trace = langfuse.api.trace.get(trace_id)
        return [trace]
    except Exception as e:
        logger.warning(f"Langfuse fetch failed: {e}. Loading from local trace.json")
        return [_load_from_json()]


def get_langfuse_session(session_id: str) -> list[TraceWithFullDetails]:
    # Check if env vars are set, if not fallback to JSON
    if not all(os.environ.get(key) for key in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"]):
        logger.info("Langfuse environment variables not set. Loading from local trace.json")
        return [_load_from_json()]

    try:
        # Initialize Langfuse client
        langfuse = Langfuse(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            host=os.environ.get("LANGFUSE_BASE_URL", ""),
        )
        session = langfuse.api.sessions.get(session_id)
        traces = []
        for trace in session.traces:
            traces.append(langfuse.api.trace.get(trace.id))
        traces.sort(key=lambda x: x.timestamp)
        return traces
    except Exception as e:
        logger.warning(f"Langfuse session fetch failed: {e}. Loading from local trace.json")
        return [_load_from_json()]

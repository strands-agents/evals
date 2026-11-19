"""Local HTTP server backing `strands-evals ui`.

Bound to loopback, but it reads and writes files under the served directory, so it is an authenticated surface:

- a per-process token, printed in the URL, gates the page and every `/api` route.
- the `Host` header must name a loopback address, which blocks DNS rebinding.
- every client path is resolved and must stay inside the root, so `..` and outbound symlinks are rejected.
- a write carries the `modified` stamp the client loaded, and a file changed on disk since then is a 409.
"""

from __future__ import annotations

import html
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import socket
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, TypeGuard
from urllib.parse import parse_qs, quote, unquote, urlparse

from ..case import Case
from ._ui_discovery import (
    DEFAULT_DEPTH,
    DIAGNOSIS,
    EXPERIMENT,
    NOT_JSON,
    REPORT,
    SESSION,
    classify,
    discover,
    fingerprint,
    is_inside,
    read_event_log,
    real_path,
    session_seconds,
)
from ._ui_evaluators import check_evaluators, evaluator_catalog
from ._ui_markdown import diagnosis_markdown, experiment_markdown, llms_text, report_markdown, session_markdown

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]"})
WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::"})

# Where the served page finds its token: the address bar carries it on the first request only.
TOKEN_SLOT = '<meta name="strands-evals-token" content="">'
COOKIE_PREFIX = "strands_evals_ui_"

# A refused request's body is read and dropped only within these bounds; past them the connection closes.
DRAIN_LIMIT = 1024 * 1024
DRAIN_TIMEOUT = 2.0

# Where the page keeps its views: `/reports/<file>`, `/experiments/<file>`, `/optimization/<directory>/overview`.
PAGE_ROOTS = ("/reports", "/experiments", "/sessions", "/optimization")
# A report, an experiment or a session is also served as data: its page address with one of these forms added.
DATA_ROOTS = {"/reports/": REPORT, "/experiments/": EXPERIMENT, "/sessions/": SESSION}
DATA_FORMS = (".md", ".json")
_PRINTABLE = re.compile(r"[!-~]+")

_LOCAL_ASSET = re.compile(r'<link rel="stylesheet" href="([\w.-]+)">|<script src="([\w.-]+)"></script>')
_ANY_ASSET = re.compile(r"<script[^>]*\ssrc=|<link[^>]*\srel=\"stylesheet\"")
_HOST_HEADER = re.compile(r"(\[[0-9A-Fa-f:.]+\]|[^:\[\]]+)(?::\d+)?")
_TOKEN_IN_URL = re.compile(r"([?&]t=)[^&\s\"]+")


def is_page_path(path: str) -> bool:
    """Whether a request path is one of the page's own addresses. They all serve the one document."""
    return path in ("/", "/index.html") or any(path == root or path.startswith(f"{root}/") for root in PAGE_ROOTS)


def data_form(path: str) -> tuple[str, str, str] | None:
    """`(kind, file, form)` when a request path is the address of a file of that kind with a form added."""
    for root, kind in DATA_ROOTS.items():
        for form in DATA_FORMS:
            if path.startswith(root) and path.lower().endswith(f".json{form}"):
                return kind, unquote(path[len(root) : -len(form)]), form
    return None


class PathOutsideRoot(ValueError):
    """A client-supplied path resolved outside the served directory."""


def resolve_within(root: Path, relative: str) -> Path:
    """Resolve a client path against the root, refusing anything that escapes it.

    Resolves before checking, so a symlink pointing out of the tree is caught as well as a literal `..`.
    """
    if not relative:
        raise PathOutsideRoot("no path given")
    if "\x00" in relative:
        raise PathOutsideRoot(f"{relative!r} is not a path")
    root = root.resolve()
    candidate = real_path(root / relative)
    if not is_inside(root, candidate):
        raise PathOutsideRoot(f"{relative} is outside {root}")
    return candidate


def _same_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address, bound: str) -> bool:
    try:
        return address == ipaddress.ip_address(bound)
    except ValueError:  # bound by name, e.g. a hostname: only loopback names are answered
        return False


def is_loopback(host: str) -> bool:
    """Whether binding `host` keeps the server on this machine."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_timestamp(value: Any) -> TypeGuard[float]:
    """Whether a client-sent `modified` can be compared with an mtime."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) < 2**53
    return isinstance(value, float) and math.isfinite(value)


def finite(value: Any) -> Any:
    """`value` with NaN and Infinity replaced by None, which is how JSON can carry them.

    Walks with its own stack, so a document nested deeper than the recursion limit is copied too.
    """
    out: list[Any] = [None]
    todo: list[tuple[Any, Any, Any]] = [(out, 0, value)]
    while todo:
        into, key, item = todo.pop()
        if isinstance(item, dict):
            copy: Any = dict.fromkeys(item)
            into[key] = copy
            todo.extend((copy, name, held) for name, held in item.items())
        elif isinstance(item, list):
            into[key] = copy = [None] * len(item)
            todo.extend((copy, index, held) for index, held in enumerate(item))
        else:
            into[key] = None if isinstance(item, float) and not math.isfinite(item) else item
    return out[0]


def validate_experiment_payload(payload: Any) -> dict:
    """Check an experiment document well enough to refuse to write a broken one.

    Cases go through `Case.model_validate`, the same gate `Experiment.from_file` uses. Evaluators are never
    instantiated, because that would reject custom evaluator subclasses; `check_evaluators` inspects them.
    """
    if not isinstance(payload, dict):
        raise ValueError("experiment must be a JSON object")
    if not isinstance(payload.get("cases"), list):
        raise ValueError("experiment is missing a 'cases' list")
    if not isinstance(payload.get("evaluators"), list):
        raise ValueError("experiment is missing an 'evaluators' list")

    for index, case in enumerate(payload["cases"]):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} is not a JSON object")
        try:
            Case.model_validate(case)
        except Exception as exc:  # pydantic ValidationError and anything it wraps
            name = case.get("name", f"index {index}")
            raise ValueError(f"case {name!r} is not valid: {exc}") from exc

    for index, evaluator in enumerate(payload["evaluators"]):
        if not isinstance(evaluator, dict) or "evaluator_type" not in evaluator:
            raise ValueError(f"evaluator {index} has no 'evaluator_type'")

    return payload


def write_json_atomic(target: Path, payload: dict) -> float:
    """Serialize, then replace the file in one step; returns the new mtime.

    Serializing first leaves the existing file untouched when the document is not valid JSON
    (NaN and Infinity included), matching `Experiment.to_file`.
    """
    data = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    try:
        mode = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        mode = None
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
    )
    try:
        with handle as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # The temp file is created 0600; without this a save would narrow the file's permissions.
        if mode is not None:
            os.chmod(handle.name, mode)
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target.stat().st_mtime


UNAUTHORIZED_PAGE = (
    b"<!doctype html><meta charset=utf-8><title>strands-evals</title>"
    b"<body style='font:14px \"Space Grotesk\",system-ui,sans-serif;padding:40px'>"
    b"<h3 style='font:600 18px \"JetBrains Mono\",ui-monospace,monospace'>Open the link the CLI printed</h3>"
    b"<p>This page needs the token in the URL that <code>strands-evals ui</code> printed.</p>"
)


@dataclass
class UIConfig:
    root: Path
    token: str
    depth: int = DEFAULT_DEPTH
    read_only: bool = False
    asset: str = ""
    # True when `asset` came from the shipped file rather than a caller, so it can be re-read.
    reload_asset: bool = False
    # The address the server was asked to bind, which decides which `Host` headers it answers.
    bound_host: str = "127.0.0.1"
    # Carried by the page's own <style> and <script> tags; markup injected from a file cannot know it.
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(16))

    def page(self) -> str:
        """The page to serve, re-read from disk on every request unless a caller pinned one."""
        if not self.reload_asset:
            return self.asset
        try:
            return load_asset(self.nonce)
        except OSError as exc:
            logger.warning("error=<%s> | could not re-read the ui page, serving the copy read at startup", exc)
            return self.asset

    def content_security_policy(self) -> str:
        own = f"'nonce-{self.nonce}'"
        return (
            f"default-src 'none'; script-src {own}; style-src {own}; style-src-attr 'unsafe-inline'; "
            "img-src data:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
        )


class UIHandler(BaseHTTPRequestHandler):
    """Routes for the UI. `config` is attached to the server by `build_server`."""

    server_version = "strands-evals-ui"
    protocol_version = "HTTP/1.1"
    timeout = 60  # an idle or stalled connection gives its thread back

    @property
    def config(self) -> UIConfig:
        return self.server.config  # type: ignore[attr-defined]

    # ── plumbing ──

    def log_message(self, fmt: str, *args: Any) -> None:
        # The base class writes to stderr; logging puts it under the CLI's -v/--debug flags.
        line = _TOKEN_IN_URL.sub(r"\1<redacted>", fmt % args)
        logger.debug("client=<%s> | %s", self.address_string(), line)

    def _send(
        self, status: HTTPStatus, body: bytes, content_type: str, *, headers: dict[str, str] | None = None
    ) -> None:
        self._answered = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        try:
            text = json.dumps(payload, allow_nan=False)
        except ValueError:  # a file on disk held NaN or Infinity, which Python reads and JSON forbids
            text = json.dumps(finite(payload), allow_nan=False)
        self._send(status, text.encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": message})

    def _host_is_admitted(self) -> bool:
        """Whether `Host` names this server, which is what stops DNS rebinding.

        A rebinding request carries the attacker's hostname. Loopback names are always answered; a server
        bound beyond loopback also answers to the address it was bound to, or to any IP literal when it was
        bound to every interface.
        """
        match = _HOST_HEADER.fullmatch((self.headers.get("Host") or "").strip())
        if match is None:
            return False
        name = match.group(1).lower()
        if name in LOOPBACK_HOSTS:
            return True
        try:
            address = ipaddress.ip_address(name.strip("[]"))
        except ValueError:
            return False
        bound = self.config.bound_host
        return bound in WILDCARD_HOSTS or _same_address(address, bound)

    def _is_token(self, supplied: str) -> bool:
        # Compared as bytes: compare_digest rejects non-ASCII str, and the value is attacker-chosen.
        return secrets.compare_digest(supplied.encode("utf-8", "replace"), self.config.token.encode("utf-8"))

    def _authorized(self, query: dict[str, list[str]]) -> bool:
        return self._is_token(self.headers.get("X-Evals-Token") or query.get("t", [""])[0])

    def _cookie_name(self) -> str:
        # Cookies ignore the port, so two servers on one machine need two names.
        return f"{COOKIE_PREFIX}{self.connection.getsockname()[1]}"

    def _has_page_cookie(self) -> bool:
        try:
            morsel = SimpleCookie(self.headers.get("Cookie") or "").get(self._cookie_name())
        except CookieError:
            return False
        return morsel is not None and self._is_token(unquote(morsel.value))

    def _refusal(self, query: dict[str, list[str]]) -> tuple[HTTPStatus, str] | None:
        """Why this request is not admitted, if it is not: the wrong Host or the wrong token."""
        if not self._host_is_admitted():
            return HTTPStatus.FORBIDDEN, "this server only answers requests addressed to the host it is serving on"
        if not self._authorized(query):
            return HTTPStatus.UNAUTHORIZED, "missing or wrong token"
        return None

    def _settings(self) -> dict:
        return {"root": str(self.config.root), "depth": self.config.depth, "read_only": self.config.read_only}

    def _requested_file(
        self, query: dict[str, list[str]], missing: str = "{rel} is not a file"
    ) -> tuple[str, Path] | None:
        """The `path` parameter resolved to a file inside the root, or None once the error is sent."""
        rel = query.get("path", [""])[0]
        try:
            target = resolve_within(self.config.root, rel)
        except PathOutsideRoot as exc:
            self._error(HTTPStatus.FORBIDDEN, str(exc))
            return None
        # `os.path.isfile`, because `Path.is_file` raises on a name too long for the filesystem.
        if not os.path.isfile(target):
            self._error(HTTPStatus.NOT_FOUND, missing.format(rel=rel))
            return None
        return rel, target

    def _body_length(self) -> int | None:
        """The declared body size, or None when it cannot be read by length.

        That is a missing or malformed `Content-Length`, a chunked body, or a size over the cap.
        """
        declared = self.headers.get("Content-Length")
        if declared is None or self.headers.get("Transfer-Encoding"):
            return None
        try:
            length = int(declared)
        except ValueError:
            return None
        return length if 0 <= length <= MAX_UPLOAD_BYTES else None

    def _reject_unread(self, status: HTTPStatus, message: str) -> None:
        """Refuse a request whose body has not been read.

        A small body is drained first: left unread, a kept-alive connection would parse it as the next
        request. Anything else closes the connection, so a client cannot hold a thread by withholding a body.
        """
        remaining = self._body_length()
        if remaining is None or remaining > DRAIN_LIMIT:
            self.close_connection = True
        else:
            self.connection.settimeout(DRAIN_TIMEOUT)
            try:
                while remaining:
                    chunk = self.rfile.read(min(remaining, 65536))
                    if not chunk:
                        break
                    remaining -= len(chunk)
            except OSError:  # the declared body never arrived
                self.close_connection = True
            finally:
                self.connection.settimeout(self.timeout)
        self._error(status, message)

    def _json_body(self) -> dict | None:
        """The request body as a JSON object, or None once the error is sent."""
        length = self._body_length()
        if length is None:
            declared = self.headers.get("Content-Length") or ""
            if declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
                self._reject_unread(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"body over {MAX_UPLOAD_BYTES} bytes")
            else:
                self._reject_unread(HTTPStatus.LENGTH_REQUIRED, "send the body with a Content-Length")
            return None
        if length == 0:
            self._error(HTTPStatus.BAD_REQUEST, "empty body")
            return None

        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except NOT_JSON as exc:
            self._error(HTTPStatus.BAD_REQUEST, f"body is not JSON: {exc}")
            return None
        if not isinstance(body, dict):
            self._error(HTTPStatus.BAD_REQUEST, "body is not a JSON object")
            return None
        return body

    # ── routes ──

    def _get_config(self, query: dict[str, list[str]]) -> None:
        """The served folder, the scan depth, whether saving is off, and the evaluator catalogue."""
        self._json(HTTPStatus.OK, {**self._settings(), "evaluators": evaluator_catalog()})

    def _get_scan(self, query: dict[str, list[str]]) -> None:
        """What a scan of the served folder finds, to its depth and its file cap, and a fingerprint of the folder."""
        # Taken before the scan, so a file written during it reads as a change afterwards.
        mark = fingerprint(self.config.root, self.config.depth)
        result = discover(self.config.root, self.config.depth)
        # `vars`, not `asdict`: that copies every listed document level by level, and one nested deeply stops it.
        self._json(HTTPStatus.OK, {**self._settings(), "fingerprint": mark, **vars(result)})

    def _get_changes(self, query: dict[str, list[str]]) -> None:
        """The fingerprint now. It differs from a scan's once something the scan listed has changed."""
        self._json(HTTPStatus.OK, {"fingerprint": fingerprint(self.config.root, self.config.depth)})

    def _get_events(self, query: dict[str, list[str]]) -> None:
        """With `?path=<events.jsonl>`: the events an optimizer run logged, one per line of the file."""
        requested = self._requested_file(query)
        if requested is None:
            return
        rel, target = requested
        log = read_event_log(target)
        self._json(
            HTTPStatus.OK,
            {
                "path": rel,
                "modified": target.stat().st_mtime,
                "events": log.events,
                "bad_lines": log.bad_lines,
                "truncated": log.truncated,
            },
        )

    def _get_file(self, query: dict[str, list[str]]) -> None:
        """With `?path=<file>`: one JSON file under the served folder, with the time it was last modified."""
        requested = self._requested_file(query)
        if requested is None:
            return
        rel, target = requested
        try:
            obj = json.loads(target.read_text(encoding="utf-8"))
        except NOT_JSON as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, f"{rel} could not be parsed: {exc}")
            return
        self._json(
            HTTPStatus.OK,
            {
                "path": rel,
                "modified": target.stat().st_mtime,
                "document": obj,
                "evaluator_notes": check_evaluators(obj.get("evaluators") if isinstance(obj, dict) else None),
            },
        )

    def _get_llms(self, query: dict[str, list[str]]) -> None:
        """This description of what the server answers."""
        handlers = {**self._GET_ROUTES, "PUT /api/experiment": UIHandler._put_experiment}
        routes = {route: " ".join((handler.__doc__ or "").split()) for route, handler in handlers.items()}
        text = llms_text(routes, PAGE_ROOTS, DATA_ROOTS, DATA_FORMS)
        self._send(HTTPStatus.OK, text.encode("utf-8"), "text/plain; charset=utf-8")

    def _get_data_form(self, wanted: str, rel: str, form: str) -> None:
        """A report, an experiment or a session as Markdown or as JSON, for a reader with no browser."""
        requested = self._requested_file({"path": [rel]})
        if requested is None:
            return
        target = requested[1]
        try:
            obj = json.loads(target.read_text(encoding="utf-8"))
        except NOT_JSON as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, f"{rel} could not be parsed: {exc}")
            return
        kind = classify(obj)
        # A session's address also names a diagnosis: the page opens one there when its session is not in the folder.
        if kind != wanted and (wanted, kind) != (SESSION, DIAGNOSIS):
            self._error(HTTPStatus.NOT_FOUND, f"{rel} is not {'an' if wanted == EXPERIMENT else 'a'} {wanted}")
            return
        if form == ".json":
            self._json(HTTPStatus.OK, obj)
            return
        obj = finite(obj)  # as the page is sent it, so both write the same Markdown
        # The path as the scan writes it, so the file is found in the listing however it was asked for.
        root = self.config.root.resolve()
        path = target.relative_to(root).as_posix()
        found = discover(self.config.root, self.config.depth)

        def listed(entries: list[dict]) -> dict:
            return next((entry for entry in entries if entry["path"] == path), {})

        try:
            if kind == REPORT:
                paired = listed(found.reports)
                text = report_markdown(path, obj, paired.get("experiment"), paired.get("link"))
            elif kind == EXPERIMENT:
                text = experiment_markdown(path, obj, listed(found.experiments).get("reports") or [])
            elif kind == DIAGNOSIS:
                text = diagnosis_markdown(path, obj, listed(found.diagnoses).get("session"))
            else:
                diagnoses = []
                for made in listed(found.sessions).get("diagnoses") or []:
                    try:
                        diagnoses.append((made, finite(json.loads((root / made).read_text(encoding="utf-8")))))
                    except NOT_JSON as exc:
                        self._error(HTTPStatus.UNPROCESSABLE_ENTITY, f"{made} could not be parsed: {exc}")
                        return
                text = session_markdown(path, obj, session_seconds(obj), diagnoses)
        except RecursionError:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, f"{rel} is nested too deeply to write as Markdown")
            return
        self._send(HTTPStatus.OK, text.encode("utf-8", "replace"), "text/markdown; charset=utf-8")

    def _put_experiment(self, query: dict[str, list[str]], body: dict) -> None:
        """With `?path=<experiment file>` and a body of `{"document": ..., "modified": ...}`: saves the experiment
        over the file it was read from. `modified` is the time `/api/file` gave; a file changed since is refused,
        and so is every save when the server was started with `--read-only`.
        """
        if "document" not in body:
            self._error(HTTPStatus.BAD_REQUEST, "expected {'document': ..., 'modified': ...}")
            return
        requested = self._requested_file(query, missing="{rel} is no longer on disk, so it cannot be saved in place")
        if requested is None:
            return
        rel, target = requested
        if target.suffix.lower() != ".json":
            self._error(HTTPStatus.BAD_REQUEST, "only .json files can be written")
            return

        try:
            document = validate_experiment_payload(body["document"])
        except ValueError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            return

        expected = body.get("modified")
        if not _is_timestamp(expected):
            self._error(HTTPStatus.BAD_REQUEST, "'modified' must be the number the file was loaded with")
            return
        # The page saves back to the experiment it loaded; it never creates a file or replaces another kind.
        try:
            current = json.loads(target.read_text(encoding="utf-8"))
        except NOT_JSON:
            current = None
        if classify(current) != EXPERIMENT:
            self._error(HTTPStatus.CONFLICT, f"{rel} is not an experiment file")
            return
        # Lost-update guard: a file whose mtime differs from the one the client loaded was written by someone else.
        if abs(target.stat().st_mtime - expected) > 1e-6:
            self._error(HTTPStatus.CONFLICT, f"{rel} changed on disk since it was loaded; reload before saving")
            return
        if not os.access(target, os.W_OK):
            self._error(HTTPStatus.FORBIDDEN, f"{rel} is read-only")
            return

        try:
            modified = write_json_atomic(target, document)
        except ValueError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, f"{rel} is not valid JSON: {exc}")
            return
        except RecursionError:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, f"{rel} is nested too deeply to write")
            return
        except OSError as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"could not write {rel}: {exc}")
            return

        logger.info("path=<%s>, cases=<%d> | wrote experiment", rel, len(document["cases"]))
        self._json(
            HTTPStatus.OK,
            {
                "path": rel,
                "modified": modified,
                "cases": len(document["cases"]),
                "evaluator_notes": check_evaluators(document.get("evaluators")),
            },
        )

    def _get_page(self, query: dict[str, list[str]], status: HTTPStatus = HTTPStatus.OK) -> None:
        """Serve the page to a request that has the token or the cookie.

        The printed link is answered with the cookie and a redirect to the same address without the token,
        so the token is not left in the address bar, the history or a shared link.
        """
        html_type = "text/html; charset=utf-8"
        if "t" in query and self._authorized(query):
            cookie = f"{self._cookie_name()}={quote(self.config.token, safe='')}; Path=/; HttpOnly; SameSite=Lax"
            # The same address without the token, so a link into the page lands where it pointed.
            link = urlparse(self.path)
            rest = "&".join(pair for pair in link.query.split("&") if pair and unquote(pair.split("=")[0]) != "t")
            path = link.path + (f"?{rest}" if rest else "")
            where = path if _PRINTABLE.fullmatch(path) else "/"
            self._send(HTTPStatus.SEE_OTHER, b"", html_type, headers={"Location": where, "Set-Cookie": cookie})
            return
        if not (self._authorized(query) or self._has_page_cookie()):
            self._send(HTTPStatus.UNAUTHORIZED, UNAUTHORIZED_PAGE, html_type)
            return
        slot = TOKEN_SLOT.replace('""', f'"{html.escape(self.config.token, quote=True)}"')
        page = self.config.page().replace(TOKEN_SLOT, slot, 1)
        policy = {"Content-Security-Policy": self.config.content_security_policy()}
        self._send(status, page.encode("utf-8"), html_type, headers=policy)

    _GET_ROUTES: dict[str, Callable[[UIHandler, dict[str, list[str]]], None]] = {
        "/api/config": _get_config,
        "/api/scan": _get_scan,
        "/api/changes": _get_changes,
        "/api/events": _get_events,
        "/api/file": _get_file,
        "/llms.txt": _get_llms,
    }

    def _answer(self, route: Callable[[], None]) -> None:
        """Run a request. One that raises is answered with an error, not dropped with nothing sent.

        The body names the kind of failure only: what was raised may quote a path, and the log has it.
        """
        self._answered = False
        try:
            route()
        except Exception as exc:
            if self._answered:  # part of an answer has gone out, so a second one cannot follow it
                raise
            logger.exception("path=<%s> | request failed", urlparse(self.path).path)
            self.close_connection = True
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"the server could not answer this request: {type(exc).__name__}"
            )

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        self._answer(self._get)

    def _get(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        form = data_form(parsed.path)
        is_page = form is None and is_page_path(parsed.path)

        if is_page and self._host_is_admitted():
            self._get_page(query)
            return
        # A browser holding the cookie, at an address the page does not have, gets the page, which says so.
        stray = form is None and parsed.path not in self._GET_ROUTES and not f"{parsed.path}/".startswith("/api/")
        if stray and self._has_page_cookie() and self._host_is_admitted():
            self._get_page({}, HTTPStatus.NOT_FOUND)
            return
        refusal = self._refusal(query)
        if refusal:
            self._error(*refusal)
            return
        if form:
            self._get_data_form(*form)
            return

        route = self._GET_ROUTES.get(parsed.path)
        if route is None:
            self._error(HTTPStatus.NOT_FOUND, f"no route for {parsed.path}")
            return
        route(self, query)

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_PUT(self) -> None:  # noqa: N802
        self._answer(self._put)

    def _put(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        refusal = self._refusal(query)
        if refusal is None and parsed.path != "/api/experiment":
            refusal = HTTPStatus.NOT_FOUND, f"no route for {parsed.path}"
        if refusal is None and self.config.read_only:
            refusal = HTTPStatus.FORBIDDEN, "server started with --read-only"
        if refusal:
            self._reject_unread(*refusal)
            return

        body = self._json_body()
        if body is not None:
            self._put_experiment(query, body)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], config: UIConfig):
        self.config = config
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, handler)


def load_asset(nonce: str = "") -> str:
    """The page shipped beside this module, with its stylesheet and scripts inlined.

    One document keeps every byte of the page behind the token and needs no static routes.
    `tests/ui/harness.mjs` assembles the page the same way for the browser tests.

    Raises:
        ValueError: If `ui.html` references an asset in a form this cannot inline.
    """
    assets = resources.files(__package__).joinpath("assets")
    source = assets.joinpath("ui.html").read_text(encoding="utf-8")
    if len(_ANY_ASSET.findall(source)) != len(_LOCAL_ASSET.findall(source)):
        raise ValueError(
            'ui.html may only reference assets as <link rel="stylesheet" href="NAME"> or <script src="NAME"></script>'
        )
    attribute = f' nonce="{nonce}"' if nonce else ""

    def inline(match: re.Match[str]) -> str:
        stylesheet, script = match.groups()
        if stylesheet:
            return f"<style{attribute}>\n{assets.joinpath(stylesheet).read_text(encoding='utf-8')}</style>"
        return f"<script{attribute}>\n{assets.joinpath(script).read_text(encoding='utf-8')}</script>"

    return _LOCAL_ASSET.sub(inline, source)


def build_server(
    root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    depth: int = DEFAULT_DEPTH,
    read_only: bool = False,
    token: str | None = None,
    asset: str | None = None,
) -> _Server:
    """Bind a server for `root`. Port 0 picks a free one, which the tests rely on."""
    config = UIConfig(
        root=root.resolve(),
        token=token or secrets.token_urlsafe(32),
        depth=depth,
        read_only=read_only,
        reload_asset=asset is None,
        bound_host=host,
    )
    config.asset = asset if asset is not None else load_asset(config.nonce)
    try:
        return _Server((host, port), UIHandler, config)
    except OSError as exc:
        raise OSError(f"could not bind {host}:{port}: {exc}") from exc


def server_url(server: _Server) -> str:
    """The link to print: the page's address with the token, on loopback when every interface was bound."""
    host, port = str(server.server_address[0]), server.server_address[1]
    shown = {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "::1"}.get(host, host)
    if ":" in shown:
        shown = f"[{shown}]"
    return f"http://{shown}:{port}/?t={server.config.token}"

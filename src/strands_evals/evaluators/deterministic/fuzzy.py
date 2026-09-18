from typing_extensions import Any

from ...types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..evaluator import Evaluator

# Sentinel so an explicitly-passed ``value=None`` is distinguishable from "compare against
# ``expected_output``". ``Equals`` uses a plain ``None`` default and therefore cannot check an
# expected value that is itself ``None``; ``FuzzyEquals`` keeps that door open.
_UNSET = object()


class FuzzyEquals(Evaluator[InputT, OutputT]):
    """Deterministic structural comparison with configurable tolerance.

    Exact matching (``Equals``) is too brittle for structured outputs such as SQL result sets,
    JSON objects, and tabular data, where row order, key order, numeric precision, or whitespace
    should not decide a pass. ``FuzzyEquals`` compares ``actual_output`` against an expected value
    recursively across lists and dicts, applying the knobs below at every level.

    Attributes:
        value: The expected value. When omitted, ``expected_output`` from the case is used. Pass
            ``value=None`` explicitly to assert against a literal ``None``.
        ignore_order: Treat lists as multisets so element order does not matter.
        ignore_key_order: When ``False``, dict comparison additionally requires the same key
            insertion order; when ``True`` (default) only the key/value pairs must match, matching
            ordinary Python dict-equality semantics.
        numeric_tolerance: Relative tolerance for numeric values, e.g. ``0.01`` accepts values
            within +/-1%. ``0.0`` (default) requires exact numeric equality.
        case_sensitive: When ``False``, strings are compared case-insensitively.
        ignore_whitespace: When ``True``, leading/trailing whitespace is stripped and internal
            runs of whitespace collapse to a single space before string comparison.
        subset_mode: When ``True``, pass if the expected value is *contained* in the actual value
            (``actual`` superset of ``expected``): expected dict keys must be present with matching
            values (extra actual keys are ignored) and every expected list element must match a
            distinct actual element.
        type_coercion: When ``True``, numeric strings compare equal to numbers (``"42" == 42``)
            and boolean strings compare equal to booleans (``"true" == True``).
    """

    def __init__(
        self,
        value: Any = _UNSET,
        *,
        ignore_order: bool = False,
        ignore_key_order: bool = True,
        numeric_tolerance: float = 0.0,
        case_sensitive: bool = True,
        ignore_whitespace: bool = False,
        subset_mode: bool = False,
        type_coercion: bool = False,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.value = value
        self.ignore_order = ignore_order
        self.ignore_key_order = ignore_key_order
        self.numeric_tolerance = numeric_tolerance
        self.case_sensitive = case_sensitive
        self.ignore_whitespace = ignore_whitespace
        self.subset_mode = subset_mode
        self.type_coercion = type_coercion

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        expected = self.value if self.value is not _UNSET else evaluation_case.expected_output
        match, reason = self._compare(expected, evaluation_case.actual_output, path="$")
        return [
            EvaluationOutput(
                score=1.0 if match else 0.0,
                test_pass=match,
                reason="actual_output matches expected value" if match else reason,
            )
        ]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        return self.evaluate(evaluation_case)

    def to_dict(self) -> dict:
        # ``value`` may be the private ``_UNSET`` sentinel; strip it so serialization stays clean.
        _dict = super().to_dict()
        if _dict.get("value") is _UNSET:
            _dict.pop("value", None)
        return _dict

    # --- comparison core ---

    def _compare(self, expected: Any, actual: Any, path: str) -> tuple[bool, str]:
        """Recursively compare ``expected`` and ``actual``, returning (match, reason_on_fail)."""
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False, f"at {path}: expected an object, got {type(actual).__name__}"
            return self._compare_dicts(expected, actual, path)

        # Strings are sequences but must be compared as scalars, never element-wise.
        if isinstance(expected, (list, tuple)) and not isinstance(expected, (str, bytes)):
            if not isinstance(actual, (list, tuple)) or isinstance(actual, (str, bytes)):
                return False, f"at {path}: expected a list, got {type(actual).__name__}"
            return self._compare_lists(list(expected), list(actual), path)

        return self._compare_scalars(expected, actual, path)

    def _compare_dicts(self, expected: dict, actual: dict, path: str) -> tuple[bool, str]:
        if self.subset_mode:
            missing = [key for key in expected if key not in actual]
            if missing:
                return False, f"at {path}: missing expected key(s) {sorted(map(str, missing))}"
        else:
            if set(expected.keys()) != set(actual.keys()):
                return False, (
                    f"at {path}: key set differs "
                    f"(expected {sorted(map(str, expected))}, got {sorted(map(str, actual))})"
                )
            if not self.ignore_key_order and list(expected.keys()) != list(actual.keys()):
                return False, f"at {path}: key order differs"

        for key, expected_value in expected.items():
            match, reason = self._compare(expected_value, actual[key], f"{path}.{key}")
            if not match:
                return False, reason
        return True, ""

    def _compare_lists(self, expected: list, actual: list, path: str) -> tuple[bool, str]:
        if self.ignore_order:
            return self._compare_lists_unordered(expected, actual, path)

        if self.subset_mode:
            # Ordered containment: expected must appear as an in-order subsequence of actual.
            actual_index = 0
            for expected_index, expected_value in enumerate(expected):
                found = False
                while actual_index < len(actual):
                    match, _ = self._compare(expected_value, actual[actual_index], f"{path}[{expected_index}]")
                    actual_index += 1
                    if match:
                        found = True
                        break
                if not found:
                    return False, f"at {path}: expected element at index {expected_index} not found in order"
            return True, ""

        if len(expected) != len(actual):
            return False, f"at {path}: length differs (expected {len(expected)}, got {len(actual)})"
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual, strict=False)):
            match, reason = self._compare(expected_value, actual_value, f"{path}[{index}]")
            if not match:
                return False, reason
        return True, ""

    def _compare_lists_unordered(self, expected: list, actual: list, path: str) -> tuple[bool, str]:
        remaining = list(range(len(actual)))
        for expected_index, expected_value in enumerate(expected):
            matched_at = None
            for pos, actual_index in enumerate(remaining):
                match, _ = self._compare(expected_value, actual[actual_index], f"{path}[{expected_index}]")
                if match:
                    matched_at = pos
                    break
            if matched_at is None:
                return False, f"at {path}: no match found for expected element at index {expected_index}"
            remaining.pop(matched_at)

        if not self.subset_mode and remaining:
            return False, f"at {path}: {len(remaining)} unexpected element(s) in actual"
        return True, ""

    def _compare_scalars(self, expected: Any, actual: Any, path: str) -> tuple[bool, str]:
        left, right = expected, actual
        if self.type_coercion:
            left = _coerce_scalar(left)
            right = _coerce_scalar(right)

        # bool is an int subclass; compare booleans exactly and never numerically. A boolean only
        # matches another boolean, so True != 1 even though Python's == says otherwise.
        if isinstance(left, bool) or isinstance(right, bool):
            match = isinstance(left, bool) and isinstance(right, bool) and left == right
            return match, "" if match else f"at {path}: expected {expected!r}, got {actual!r}"

        if _is_number(left) and _is_number(right):
            match = _numbers_close(float(left), float(right), self.numeric_tolerance)
            return match, "" if match else f"at {path}: expected {expected!r}, got {actual!r}"

        if isinstance(left, str) and isinstance(right, str):
            left_norm = self._normalize_string(left)
            right_norm = self._normalize_string(right)
            match = left_norm == right_norm
            return match, "" if match else f"at {path}: expected {expected!r}, got {actual!r}"

        match = left == right
        return match, "" if match else f"at {path}: expected {expected!r}, got {actual!r}"

    def _normalize_string(self, value: str) -> str:
        result = value
        if self.ignore_whitespace:
            result = " ".join(result.split())
        if not self.case_sensitive:
            result = result.lower()
        return result


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numbers_close(left: float, right: float, tolerance: float) -> bool:
    if left == right:
        return True
    if tolerance <= 0.0:
        return False
    scale = max(abs(left), abs(right))
    return abs(left - right) <= tolerance * scale


def _coerce_scalar(value: Any) -> Any:
    """Coerce numeric/boolean strings into their scalar types for cross-type comparison."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return value

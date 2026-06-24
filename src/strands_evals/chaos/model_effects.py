"""Model output corruption effect classes.

Effects are polymorphic — they accept both ``str`` and ``list[dict]`` content.
The caller (ModelChaosPlugin) is responsible for locating the content within
the framework-specific response; these effects only corrupt it.

Effect hierarchy:
    ChaosEffect (from .effects) → ModelEffect → concrete effects
"""

import logging
import random
import re
from typing import Any

from .effects import ChaosEffect
from .model_types import ModelOutputCorruptionConfig, ModelOutputCorruptionType, ModelOutputHallucinationType
from .model_utils import (
    _CONFABULATION_TEMPLATES,
    _CONTRADICTION_PHRASES,
    _DATE_PATTERNS,
    _LOCATION_SUBSTITUTIONS,
    _NUMBER_PATTERNS,
    _PRICE_PATTERNS,
    _REFUSAL_TEMPLATES,
    _TOOL_NARRATIVES,
    _map_text_in_blocks,
)

logger = logging.getLogger(__name__)

_GARBAGE_STRINGS = [
    "\x00\xff\xfe GARBAGE DATA \x00\x01",
    "asdkjh2398yr2h3f CORRUPTED asjkdhf98",
    "<!DOCTYPE html><html><ERROR>NOT_A_VALID_RESPONSE</ERROR>",
    "NaN undefined null [object Object]",
]

# Set of hallucination types for dispatch
HALLUCINATION_TYPES = set(ModelOutputHallucinationType)


class FormatCorruptionEffect(ChaosEffect):
    """Corrupts model output content format.

    Supports: EMPTY_RESPONSE, TRUNCATED_RESPONSE, MALFORMED_JSON,
    SCHEMA_VIOLATION, GARBAGE_OUTPUT.
    """

    hook = "post"
    effect_type: str = "format_corruption"

    def __init__(self, config: ModelOutputCorruptionConfig):
        super().__init__()
        self._config = config

    def apply(self, content: Any = None) -> Any:
        """Corrupt *content* and return the same shape."""
        if content is None:
            raise ValueError("FormatCorruptionEffect.apply() requires content")

        ct = self._config.corruption_type

        if isinstance(content, str):
            return self._apply_to_str(content, ct)
        elif isinstance(content, list):
            return self._apply_to_blocks(content, ct)
        else:
            raise ValueError(
                f"FormatCorruptionEffect.apply() received unsupported content type "
                f"{type(content).__name__}; expected str or list[dict]."
            )

    def _truncate(self, text: str) -> str:
        if not text:
            return text
        end = max(1, round(len(text) * (1 - self._config.truncate_ratio)))
        return text[:end]

    def _apply_to_str(self, text: str, ct: Any) -> str:
        match ct:
            case ModelOutputCorruptionType.EMPTY_RESPONSE:
                return ""
            case ModelOutputCorruptionType.TRUNCATED_RESPONSE:
                return self._truncate(text)
            case ModelOutputCorruptionType.GARBAGE_OUTPUT:
                return random.choice(_GARBAGE_STRINGS)
            case ModelOutputCorruptionType.MALFORMED_JSON:
                return self._malform_text(text)
            case _:
                return text

    def _apply_to_blocks(self, blocks: list, ct: Any) -> list:
        match ct:
            case ModelOutputCorruptionType.EMPTY_RESPONSE:
                return []
            case ModelOutputCorruptionType.TRUNCATED_RESPONSE:
                return _map_text_in_blocks(blocks, self._truncate)
            case ModelOutputCorruptionType.GARBAGE_OUTPUT:
                return _map_text_in_blocks(blocks, lambda _: random.choice(_GARBAGE_STRINGS))
            case ModelOutputCorruptionType.MALFORMED_JSON:
                return self._malform_json(blocks)
            case ModelOutputCorruptionType.SCHEMA_VIOLATION:
                return self._violate_schema(blocks)
            case _:
                return list(blocks)

    @staticmethod
    def _malform_text(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return stripped[: len(stripped) // 2]
        return text

    @staticmethod
    def _malform_json(blocks: list) -> list:
        import json as json_mod

        def corrupt_tool_use(tool_use: dict) -> dict:
            raw = json_mod.dumps(tool_use.get("input", {}))
            tool_use["input"] = raw[:-1] if raw.endswith("}") else raw + "{{{"
            return tool_use

        result = []
        for block in blocks:
            block = dict(block)
            if "toolUse" in block:
                block["toolUse"] = corrupt_tool_use(dict(block["toolUse"]))
            elif "text" in block and isinstance(block["text"], str):
                block["text"] = FormatCorruptionEffect._malform_text(block["text"])
            result.append(block)
        return result

    @staticmethod
    def _violate_schema(blocks: list) -> list:
        type_swaps: list[Any] = [None, 99999, True, [], "WRONG_TYPE"]

        def corrupt_tool_use(tool_use: dict) -> dict:
            inp = tool_use.get("input", {})
            if isinstance(inp, dict) and inp:
                corrupted = {}
                for k, v in inp.items():
                    candidates = [s for s in type_swaps if not isinstance(v, type(s))]
                    corrupted[k] = random.choice(candidates) if candidates else "WRONG_TYPE"
                tool_use["input"] = corrupted
            return tool_use

        result = []
        for block in blocks:
            block = dict(block)
            if "toolUse" in block:
                block["toolUse"] = corrupt_tool_use(dict(block["toolUse"]))
            result.append(block)
        return result


class HallucinationEffect(ChaosEffect):
    """Corrupts model output text with hallucination mutations.

    Supports: FACTUAL_ERROR, CONFABULATION, CONTEXT_UNFAITHFULNESS,
    TOOL_CLAIM_FABRICATION.
    """

    hook = "post"
    effect_type: str = "hallucination"

    def __init__(self, config: ModelOutputCorruptionConfig) -> None:
        super().__init__()
        self._config = config

    def apply(self, content: Any = None) -> Any:
        """Mutate *content* and return the same shape."""
        if content is None:
            raise ValueError("HallucinationEffect.apply() requires content")

        ct = self._config.corruption_type

        if isinstance(content, str):
            return self._apply_to_str(content, ct)
        elif isinstance(content, list):
            return self._apply_to_blocks(content, ct)
        else:
            raise ValueError(
                f"HallucinationEffect.apply() received unsupported content type "
                f"{type(content).__name__}; expected str or list[dict]."
            )

    def _apply_to_str(self, text: str, ct: Any) -> str:
        match ct:
            case ModelOutputHallucinationType.FACTUAL_ERROR:
                return self._factual_error(text)
            case ModelOutputHallucinationType.CONFABULATION:
                return self._confabulation(text)
            case ModelOutputHallucinationType.CONTEXT_UNFAITHFULNESS:
                return self._context_unfaithfulness(text)
            case ModelOutputHallucinationType.TOOL_CLAIM_FABRICATION:
                return self._tool_claim_fabrication(text)
            case _:
                return text

    def _apply_to_blocks(self, blocks: list, ct: Any) -> list:
        match ct:
            case ModelOutputHallucinationType.FACTUAL_ERROR:
                return _map_text_in_blocks(blocks, self._factual_error)
            case ModelOutputHallucinationType.CONFABULATION:
                return _map_text_in_blocks(blocks, self._confabulation)
            case ModelOutputHallucinationType.CONTEXT_UNFAITHFULNESS:
                return _map_text_in_blocks(blocks, self._context_unfaithfulness)
            case ModelOutputHallucinationType.TOOL_CLAIM_FABRICATION:
                return _map_text_in_blocks(blocks, self._tool_claim_fabrication)
            case _:
                return list(blocks)

    def _factual_error(self, text: str) -> str:
        target_types = self._config.target_entity_types
        result = text

        entity_patterns: list[tuple] = []
        if target_types is None or "date" in target_types:
            entity_patterns.extend(_DATE_PATTERNS)
        if target_types is None or "number" in target_types:
            entity_patterns.extend(_NUMBER_PATTERNS)
        if target_types is None or "price" in target_types:
            entity_patterns.extend(_PRICE_PATTERNS)

        for pattern, replacer in entity_patterns:
            try:
                result = re.sub(pattern, replacer, result)
            except Exception:
                pass

        if target_types is None or "location" in target_types:
            for old_loc, new_loc in _LOCATION_SUBSTITUTIONS.items():
                result = result.replace(old_loc, new_loc)

        return result

    def _confabulation(self, text: str) -> str:
        if not text:
            return text
        template = random.choice(_CONFABULATION_TEMPLATES)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 1:
            return template + text
        insert_pos = random.randint(1, len(sentences) - 1)
        sentences.insert(insert_pos, template)
        return " ".join(sentences)

    def _context_unfaithfulness(self, text: str) -> str:
        if not text:
            return text
        phrase = random.choice(_CONTRADICTION_PHRASES)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 1:
            return text + phrase
        insert_pos = random.randint(1, len(sentences) - 1)
        sentences.insert(insert_pos, phrase.strip())
        return " ".join(sentences)

    def _tool_claim_fabrication(self, text: str) -> str:
        if not text:
            return text
        narrative = random.choice(_TOOL_NARRATIVES)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 1:
            return text + " " + narrative
        insert_pos = random.randint(1, len(sentences) - 1)
        sentences.insert(insert_pos, narrative.strip())
        return " ".join(sentences)


class RefusalEffect(ChaosEffect):
    """Replaces model output content with a refusal message.

    Supports: FULL_REFUSAL only.
    """

    hook = "post"
    effect_type: str = "refusal"

    def __init__(self, config: ModelOutputCorruptionConfig) -> None:
        super().__init__()
        self._config = config

    def apply(self, content: Any = None) -> Any:
        """Replace *content* with a refusal message."""
        if content is None:
            raise ValueError("RefusalEffect.apply() requires content")

        template = random.choice(_REFUSAL_TEMPLATES)

        if isinstance(content, str):
            return template
        elif isinstance(content, list):
            return [{"text": template}]
        else:
            raise ValueError(
                f"RefusalEffect.apply() received unsupported content type "
                f"{type(content).__name__}; expected str or list[dict]."
            )

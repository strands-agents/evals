"""Shared helpers for the red team module."""

from __future__ import annotations

import logging

from strands.models.model import Model

logger = logging.getLogger(__name__)


def _serialize_model(model: Model | str | None) -> str | None:
    """Coerce a `model` value into a JSON-safe string id, or `None`.

    `None` stays `None` so a strategy that defers to the experiment-level
    model survives a round-trip. A `Model` without a dict config exposing
    `model_id` logs a warning and returns `None`.
    """
    if model is None or isinstance(model, str):
        return model
    if isinstance(model, Model):
        config = model.get_config()
        if isinstance(config, dict):
            model_id = config.get("model_id")
            if model_id is not None:
                return str(model_id)
    logger.warning(
        "type=<%s> | non-coercible Model: missing dict config with 'model_id'; dropping field",
        type(model).__name__,
    )
    return None


def _put_model_field(out: dict, model: Model | str | None) -> None:
    """Coerce `model` via `_serialize_model` and write it onto `out["model"]` if non-None.

    Centralizes the `to_dict` idiom shared by the experiment and every model-aware
    strategy: when `_serialize_model` drops the field (None / non-coercible Model),
    the key stays absent so reload falls back to the experiment-level model.
    """
    model_id = _serialize_model(model)
    if model_id is not None:
        out["model"] = model_id

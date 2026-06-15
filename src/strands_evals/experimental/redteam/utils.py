"""Shared helpers for the red team module."""

from __future__ import annotations

import logging

from strands.models.model import Model

logger = logging.getLogger(__name__)


def _serialize_model(model: Model | str | None) -> str | None:
    """Coerce a `model` value into a JSON-safe string id, or `None`.

    Used by `RedTeamExperiment.to_dict` and individual strategy `to_dict`
    methods so they all serialize their `model` field the same way. Unlike
    `Evaluator._get_model_id`, `None` stays `None`: a strategy with
    `model=None` defers to the experiment-level model, and that semantic
    must survive a round-trip.

    When a non-`None` `Model` instance does not expose a dict config with
    a `model_id`, a warning is logged and `None` is returned. Returning `None`
    means the caller's `to_dict` will drop the field, so reload will fall
    back to the experiment-level model -- a visible behavior change, but the
    warning makes it diagnosable rather than silent.
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

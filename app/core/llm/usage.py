"""
Token usage tracking and cost estimation helpers.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def normalize_usage(raw: Any) -> Optional[Dict[str, Any]]:
    if not raw:
        return None

    if isinstance(raw, dict):
        prompt_tokens = raw.get("prompt_tokens", raw.get("input_tokens"))
        completion_tokens = raw.get("completion_tokens", raw.get("output_tokens"))
        total_tokens = raw.get("total_tokens")
        model = raw.get("model")
        provider = raw.get("provider")
        details = raw.get("details")
    else:
        prompt_tokens = getattr(raw, "prompt_tokens", None) or getattr(raw, "input_tokens", None)
        completion_tokens = getattr(raw, "completion_tokens", None) or getattr(raw, "output_tokens", None)
        total_tokens = getattr(raw, "total_tokens", None)
        model = getattr(raw, "model", None)
        provider = getattr(raw, "provider", None)
        details = None

    prompt_tokens = _coerce_int(prompt_tokens, 0)
    completion_tokens = _coerce_int(completion_tokens, 0)
    total_tokens = _coerce_int(total_tokens, prompt_tokens + completion_tokens)

    if prompt_tokens <= 0 and completion_tokens <= 0 and total_tokens <= 0:
        return None

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    if model:
        usage["model"] = str(model)
    if provider:
        usage["provider"] = str(provider)
    if details:
        usage["details"] = details
    return usage


def usage_from_openai_callback(
    callback: Any,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not callback:
        return None

    prompt_tokens = _coerce_int(getattr(callback, "prompt_tokens", 0), 0)
    completion_tokens = _coerce_int(getattr(callback, "completion_tokens", 0), 0)
    total_tokens = _coerce_int(getattr(callback, "total_tokens", 0), prompt_tokens + completion_tokens)

    if prompt_tokens <= 0 and completion_tokens <= 0 and total_tokens <= 0:
        return None

    details = {}
    cached_tokens = _coerce_int(getattr(callback, "prompt_tokens_cached", 0), 0)
    reasoning_tokens = _coerce_int(getattr(callback, "reasoning_tokens", 0), 0)
    if cached_tokens > 0:
        details["prompt_tokens_cached"] = cached_tokens
    if reasoning_tokens > 0:
        details["reasoning_tokens"] = reasoning_tokens

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    if model:
        usage["model"] = str(model)
    if provider:
        usage["provider"] = str(provider)
    if details:
        usage["details"] = details
    return usage


class TokenUsageTracker:
    def __init__(self) -> None:
        self._items: list[Dict[str, Any]] = []
        self._by_model: Dict[str, Dict[str, int]] = {}
        self._total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def add(
        self,
        usage: Optional[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        normalized = normalize_usage(usage)
        if not normalized:
            return

        if model and not normalized.get("model"):
            normalized["model"] = str(model)
        if provider and not normalized.get("provider"):
            normalized["provider"] = str(provider)

        record = {
            "prompt_tokens": normalized["prompt_tokens"],
            "completion_tokens": normalized["completion_tokens"],
            "total_tokens": normalized["total_tokens"],
        }
        if normalized.get("model"):
            record["model"] = normalized["model"]
        if normalized.get("provider"):
            record["provider"] = normalized["provider"]
        if source:
            record["source"] = str(source)
        if normalized.get("details"):
            record["details"] = normalized["details"]

        self._items.append(record)
        self._total["prompt_tokens"] += normalized["prompt_tokens"]
        self._total["completion_tokens"] += normalized["completion_tokens"]
        self._total["total_tokens"] += normalized["total_tokens"]

        model_key = normalized.get("model")
        if model_key:
            entry = self._by_model.setdefault(
                model_key,
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
            entry["prompt_tokens"] += normalized["prompt_tokens"]
            entry["completion_tokens"] += normalized["completion_tokens"]
            entry["total_tokens"] += normalized["total_tokens"]

    def summary(self, *, include_items: bool = True) -> Dict[str, Any]:
        if self._total["total_tokens"] <= 0:
            return {}
        data = dict(self._total)
        if self._by_model:
            data["by_model"] = dict(self._by_model)
        if include_items and self._items:
            data["items"] = list(self._items)
        return data

    def build_meta(self, *, include_items: bool = True) -> Dict[str, Any]:
        usage = self.summary(include_items=include_items)
        if not usage:
            return {}

        meta: Dict[str, Any] = {"token_usage": usage}
        cost = self._estimate_cost()
        if cost:
            meta["token_cost"] = cost
        return meta

    def _estimate_cost(self) -> Dict[str, Any]:
        from app.config import settings

        if not getattr(settings, "TOKEN_BILLING_ENABLED", True):
            return {}

        default_in = _coerce_float(getattr(settings, "TOKEN_PRICE_INPUT_PER_1K", 0.0), 0.0)
        default_out = _coerce_float(getattr(settings, "TOKEN_PRICE_OUTPUT_PER_1K", 0.0), 0.0)
        currency = str(getattr(settings, "TOKEN_PRICE_CURRENCY", "USD") or "USD")
        raw_table = getattr(settings, "TOKEN_PRICE_TABLE", None)

        pricing_table = _parse_price_table(raw_table)
        has_pricing = bool(pricing_table) or default_in > 0 or default_out > 0
        if not has_pricing:
            return {}

        by_model_tokens = dict(self._by_model)
        if not by_model_tokens and self._total["total_tokens"] > 0:
            by_model_tokens = {
                "unknown": dict(self._total)
            }

        prompt_cost = 0.0
        completion_cost = 0.0
        total_cost = 0.0
        estimated = False
        pricing_source = None
        by_model_cost: Dict[str, Any] = {}
        unknown_models = []

        for model_name, tokens in by_model_tokens.items():
            price = _match_price(model_name, pricing_table)
            source = None
            if price:
                input_price, output_price = price
                source = "model_table"
            else:
                input_price, output_price = default_in, default_out
                source = "default" if input_price > 0 or output_price > 0 else None
                if not source:
                    estimated = True
                    unknown_models.append(model_name)
                    continue

            model_prompt = _coerce_int(tokens.get("prompt_tokens"), 0)
            model_completion = _coerce_int(tokens.get("completion_tokens"), 0)
            model_prompt_cost = model_prompt * input_price / 1000.0
            model_completion_cost = model_completion * output_price / 1000.0
            model_total_cost = model_prompt_cost + model_completion_cost

            prompt_cost += model_prompt_cost
            completion_cost += model_completion_cost
            total_cost += model_total_cost

            by_model_cost[model_name] = {
                "prompt_cost": round(model_prompt_cost, 6),
                "completion_cost": round(model_completion_cost, 6),
                "total_cost": round(model_total_cost, 6),
                "pricing_source": source,
                "input_per_1k": input_price,
                "output_per_1k": output_price,
            }

            if pricing_source is None:
                pricing_source = source
            elif pricing_source != source:
                pricing_source = "mixed"

        return {
            "currency": currency,
            "prompt_cost": round(prompt_cost, 6),
            "completion_cost": round(completion_cost, 6),
            "total_cost": round(total_cost, 6),
            "pricing_source": pricing_source or "unknown",
            "estimated": estimated,
            "unknown_models": unknown_models,
            "by_model": by_model_cost,
        }


def _parse_price_table(raw_table: Any) -> Dict[str, Any]:
    if not raw_table:
        return {}
    if isinstance(raw_table, dict):
        table = raw_table
    elif isinstance(raw_table, str):
        raw = raw_table.strip()
        if not raw:
            return {}
        try:
            table = json.loads(raw)
        except Exception:
            logger.warning("Failed to parse TOKEN_PRICE_TABLE JSON.")
            return {}
    else:
        return {}

    normalized = {}
    for key, value in table.items():
        if not key:
            continue
        normalized[str(key).strip().lower()] = value
    return normalized


def _match_price(model_name: Optional[str], table: Dict[str, Any]) -> Optional[tuple[float, float]]:
    if not model_name or not table:
        return None
    model = str(model_name).strip().lower()

    if model in table:
        return _coerce_price(table[model])

    for key, value in table.items():
        if key.endswith("*") and model.startswith(key[:-1]):
            return _coerce_price(value)
    return None


def _coerce_price(value: Any) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        input_price = value.get("input", value.get("prompt", value.get("in")))
        output_price = value.get("output", value.get("completion", value.get("out")))
        input_price = _coerce_float(input_price, 0.0)
        output_price = _coerce_float(output_price, 0.0)
        return (input_price, output_price)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        input_price = _coerce_float(value[0], 0.0)
        output_price = _coerce_float(value[1], 0.0)
        return (input_price, output_price)
    if isinstance(value, (int, float)):
        price = _coerce_float(value, 0.0)
        return (price, price)
    return None

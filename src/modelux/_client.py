from __future__ import annotations

from typing import Any, Dict, Optional

import openai
from openai import APIStatusError

from ._errors import BudgetExceededError
from ._types import BudgetInfo, ModeluxMetadata

DEFAULT_BASE_URL = "https://api.modelux.ai/v1"


def _build_extra_headers(
    *,
    user_id: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    trace_id: Optional[str] = None,
    no_cache: bool = False,
    dry_run: bool = False,
) -> Dict[str, str]:
    """Convert Modelux params into X-Modelux-* request headers."""
    headers: Dict[str, str] = {}
    if user_id:
        headers["X-Modelux-User-Id"] = user_id
    if tags:
        headers["X-Modelux-User-Tags"] = ",".join(
            f"{k}={v}" for k, v in tags.items()
        )
    if trace_id:
        headers["X-Modelux-Trace-Id"] = trace_id
    if no_cache:
        headers["Cache-Control"] = "no-cache"
    if dry_run:
        headers["X-Modelux-Dry-Run"] = "true"
    return headers


def _extract_metadata(headers: Any) -> ModeluxMetadata:
    """Extract ModeluxMetadata from httpx response headers."""
    cache_status = headers.get("x-modelux-cache", "")
    cache_hit = cache_status == "HIT"
    similarity = headers.get("x-modelux-cache-similarity")
    budget_remaining = headers.get("x-modelux-budget-remaining")

    return ModeluxMetadata(
        request_id=headers.get("x-modelux-request-id", ""),
        provider_used=headers.get("x-modelux-provider-used", ""),
        model_used=headers.get("x-modelux-model-used", ""),
        cache_hit=cache_hit,
        cache_similarity=float(similarity) if cache_hit and similarity else None,
        ab_variant=headers.get("x-modelux-ab-variant"),
        budget_name=headers.get("x-modelux-budget-name"),
        budget_remaining=float(budget_remaining) if budget_remaining else None,
        budget_action=headers.get("x-modelux-budget-action"),
        budget_reset=headers.get("x-modelux-budget-reset"),
    )


def _handle_budget_error(err: APIStatusError) -> None:
    """Re-raise 402 errors as BudgetExceededError if budget info is present."""
    if err.status_code != 402:
        return
    body = err.body
    if isinstance(body, dict):
        budget_data = body.get("budget")
        if budget_data and isinstance(budget_data, dict):
            retry_after_str = err.response.headers.get("retry-after")
            retry_after = int(retry_after_str) if retry_after_str else None
            raise BudgetExceededError(
                message=body.get("message", "Budget exceeded"),
                budget=BudgetInfo(
                    name=budget_data["name"],
                    spend_usd=budget_data["spend_usd"],
                    cap_usd=budget_data["cap_usd"],
                    period=budget_data["period"],
                    period_resets_at=budget_data["period_resets_at"],
                ),
                retry_after=retry_after,
            ) from err


# ---------------------------------------------------------------------------
# Streaming wrapper
# ---------------------------------------------------------------------------


class ModeluxStream:
    """Wraps an OpenAI streaming response with Modelux metadata."""

    def __init__(self, stream: openai.Stream, metadata: ModeluxMetadata) -> None:  # type: ignore[type-arg]
        self._stream = stream
        self.modelux = metadata

    def __iter__(self):  # type: ignore[override]
        return iter(self._stream)

    def __next__(self):  # type: ignore[override]
        return next(self._stream)

    def close(self) -> None:
        self._stream.close()

    def __enter__(self):  # type: ignore[override]
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncModeluxStream:
    """Wraps an OpenAI async streaming response with Modelux metadata."""

    def __init__(self, stream: openai.AsyncStream, metadata: ModeluxMetadata) -> None:  # type: ignore[type-arg]
        self._stream = stream
        self.modelux = metadata

    def __aiter__(self):  # type: ignore[override]
        return self._stream.__aiter__()

    async def close(self) -> None:
        await self._stream.close()

    async def __aenter__(self):  # type: ignore[override]
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Chat completions namespace
# ---------------------------------------------------------------------------


class ChatCompletions:
    """Sync chat completions, wrapping OpenAI with Modelux extensions."""

    def __init__(self, client: openai.OpenAI) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: Any,
        stream: bool = False,
        user_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        trace_id: Optional[str] = None,
        no_cache: bool = False,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> Any:
        extra_headers = _build_extra_headers(
            user_id=user_id,
            tags=tags,
            trace_id=trace_id,
            no_cache=no_cache,
            dry_run=dry_run,
        )

        try:
            if stream:
                response = self._client.chat.completions.with_raw_response.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    extra_headers=extra_headers,
                    **kwargs,
                )
                metadata = _extract_metadata(response.headers)
                parsed_stream = response.parse()
                return ModeluxStream(parsed_stream, metadata)
            else:
                response = self._client.chat.completions.with_raw_response.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    extra_headers=extra_headers,
                    **kwargs,
                )
                metadata = _extract_metadata(response.headers)
                completion = response.parse()
                completion.modelux = metadata  # type: ignore[attr-defined]
                return completion
        except APIStatusError as err:
            _handle_budget_error(err)
            raise


class AsyncChatCompletions:
    """Async chat completions, wrapping AsyncOpenAI with Modelux extensions."""

    def __init__(self, client: openai.AsyncOpenAI) -> None:
        self._client = client

    async def create(
        self,
        *,
        model: str,
        messages: Any,
        stream: bool = False,
        user_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        trace_id: Optional[str] = None,
        no_cache: bool = False,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> Any:
        extra_headers = _build_extra_headers(
            user_id=user_id,
            tags=tags,
            trace_id=trace_id,
            no_cache=no_cache,
            dry_run=dry_run,
        )

        try:
            if stream:
                response = await self._client.chat.completions.with_raw_response.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    extra_headers=extra_headers,
                    **kwargs,
                )
                metadata = _extract_metadata(response.headers)
                parsed_stream = response.parse()
                return AsyncModeluxStream(parsed_stream, metadata)
            else:
                response = await self._client.chat.completions.with_raw_response.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    extra_headers=extra_headers,
                    **kwargs,
                )
                metadata = _extract_metadata(response.headers)
                completion = response.parse()
                completion.modelux = metadata  # type: ignore[attr-defined]
                return completion
        except APIStatusError as err:
            _handle_budget_error(err)
            raise


class Chat:
    def __init__(self, client: openai.OpenAI) -> None:
        self.completions = ChatCompletions(client)


class AsyncChat:
    def __init__(self, client: openai.AsyncOpenAI) -> None:
        self.completions = AsyncChatCompletions(client)


# ---------------------------------------------------------------------------
# Main clients
# ---------------------------------------------------------------------------


class Modelux:
    """Sync Modelux client. Wraps the OpenAI SDK with Modelux extensions."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers={"X-Modelux-SDK": "python/0.1.0"},
        )
        self.chat = Chat(self._client)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Modelux":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncModelux:
    """Async Modelux client. Wraps the AsyncOpenAI SDK with Modelux extensions."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers={"X-Modelux-SDK": "python/0.1.0"},
        )
        self.chat = AsyncChat(self._client)

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> "AsyncModelux":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

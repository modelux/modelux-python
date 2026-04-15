import json

import httpx
import pytest

from modelux import BudgetExceededError
from modelux._client import _build_extra_headers, _extract_metadata
from conftest import (
    SAMPLE_COMPLETION,
    SAMPLE_MODELUX_HEADERS,
    make_httpx_response,
)


# ---------------------------------------------------------------------------
# Unit tests for header building
# ---------------------------------------------------------------------------


class TestBuildExtraHeaders:
    def test_empty_when_no_params(self):
        assert _build_extra_headers() == {}

    def test_user_id(self):
        h = _build_extra_headers(user_id="user_42")
        assert h["X-Modelux-User-Id"] == "user_42"

    def test_tags(self):
        h = _build_extra_headers(tags={"tier": "premium", "cohort": "beta"})
        assert "tier=premium" in h["X-Modelux-User-Tags"]
        assert "cohort=beta" in h["X-Modelux-User-Tags"]

    def test_trace_id(self):
        h = _build_extra_headers(trace_id="trace-xyz")
        assert h["X-Modelux-Trace-Id"] == "trace-xyz"

    def test_no_cache(self):
        h = _build_extra_headers(no_cache=True)
        assert h["Cache-Control"] == "no-cache"

    def test_dry_run(self):
        h = _build_extra_headers(dry_run=True)
        assert h["X-Modelux-Dry-Run"] == "true"

    def test_all_params(self):
        h = _build_extra_headers(
            user_id="u1",
            tags={"a": "b"},
            trace_id="t1",
            no_cache=True,
            dry_run=True,
        )
        assert len(h) == 5


# ---------------------------------------------------------------------------
# Unit tests for metadata extraction
# ---------------------------------------------------------------------------


class TestExtractMetadata:
    def test_basic_metadata(self):
        headers = httpx.Headers(SAMPLE_MODELUX_HEADERS)
        meta = _extract_metadata(headers)
        assert meta.request_id == "req-abc-123"
        assert meta.provider_used == "openai"
        assert meta.model_used == "gpt-4o"
        assert meta.cache_hit is False
        assert meta.cache_similarity is None

    def test_cache_hit(self):
        headers = httpx.Headers({
            **SAMPLE_MODELUX_HEADERS,
            "x-modelux-cache": "HIT",
            "x-modelux-cache-similarity": "0.9734",
        })
        meta = _extract_metadata(headers)
        assert meta.cache_hit is True
        assert meta.cache_similarity == pytest.approx(0.9734)

    def test_ab_variant(self):
        headers = httpx.Headers({
            **SAMPLE_MODELUX_HEADERS,
            "x-modelux-ab-variant": "control",
        })
        meta = _extract_metadata(headers)
        assert meta.ab_variant == "control"

    def test_budget_metadata(self):
        headers = httpx.Headers({
            **SAMPLE_MODELUX_HEADERS,
            "x-modelux-budget-name": "team-ml",
            "x-modelux-budget-remaining": "42.1500",
            "x-modelux-budget-action": "warn_only",
        })
        meta = _extract_metadata(headers)
        assert meta.budget_name == "team-ml"
        assert meta.budget_remaining == pytest.approx(42.15)
        assert meta.budget_action == "warn_only"


# ---------------------------------------------------------------------------
# Integration tests using mock transport
# ---------------------------------------------------------------------------


class TestChatCompletions:
    def test_non_streaming_returns_completion_with_metadata(
        self, client_with_transport
    ):
        client, transport = client_with_transport
        transport.handle_request.return_value = make_httpx_response(
            SAMPLE_COMPLETION,
            200,
            SAMPLE_MODELUX_HEADERS,
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert response.id == "chatcmpl-123"
        assert response.choices[0].message.content == "Hello!"
        assert response.modelux.request_id == "req-abc-123"
        assert response.modelux.provider_used == "openai"

    def test_modelux_params_not_in_request_body(self, client_with_transport):
        client, transport = client_with_transport
        transport.handle_request.return_value = make_httpx_response(
            SAMPLE_COMPLETION,
            200,
            SAMPLE_MODELUX_HEADERS,
        )

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            user_id="user_42",
            tags={"tier": "premium"},
        )

        # Check the request body doesn't contain Modelux params
        call_args = transport.handle_request.call_args
        request = call_args[0][0]
        body = json.loads(request.content)
        assert "user_id" not in body
        assert "tags" not in body
        assert body["model"] == "gpt-4o"

    def test_modelux_params_sent_as_headers(self, client_with_transport):
        client, transport = client_with_transport
        transport.handle_request.return_value = make_httpx_response(
            SAMPLE_COMPLETION,
            200,
            SAMPLE_MODELUX_HEADERS,
        )

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            user_id="user_42",
            trace_id="trace-xyz",
            no_cache=True,
        )

        call_args = transport.handle_request.call_args
        request = call_args[0][0]
        assert request.headers.get("x-modelux-user-id") == "user_42"
        assert request.headers.get("x-modelux-trace-id") == "trace-xyz"
        assert request.headers.get("cache-control") == "no-cache"


class TestBudgetError:
    def test_402_raises_budget_exceeded_error(self, client_with_transport):
        client, transport = client_with_transport
        transport.handle_request.return_value = make_httpx_response(
            {
                "error": {
                    "type": "budget_exceeded",
                    "message": 'budget "team-ml" exceeded ($105.00 / $100.00)',
                    "code": 402,
                    "budget": {
                        "name": "team-ml",
                        "spend_usd": 105.0,
                        "cap_usd": 100.0,
                        "period": "monthly",
                        "period_resets_at": "2026-05-01T00:00:00Z",
                    },
                }
            },
            402,
            {
                "retry-after": "3600",
                "x-modelux-budget-name": "team-ml",
                "x-modelux-budget-action": "block",
            },
        )

        with pytest.raises(BudgetExceededError) as exc_info:
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )

        err = exc_info.value
        assert err.budget.name == "team-ml"
        assert err.budget.spend_usd == 105.0
        assert err.budget.cap_usd == 100.0
        assert err.budget.period == "monthly"
        assert err.retry_after == 3600

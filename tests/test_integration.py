"""
Integration tests — hit a real Modelux proxy.

Required env vars:
  - MODELUX_API_KEY           Modelux API key (e.g. mlx_sk_...)
  - MODELUX_INTEGRATION=1     set to enable these tests

Optional:
  - MODELUX_BASE_URL          defaults to https://api.modelux.ai/v1
  - MODELUX_MODEL             defaults to "@default"

Run:
  MODELUX_INTEGRATION=1 MODELUX_API_KEY=mlx_sk_... pytest tests/test_integration.py -v
"""

import os

import pytest

from modelux import Modelux, AsyncModelux


SKIP = not os.environ.get("MODELUX_INTEGRATION")
API_KEY = os.environ.get("MODELUX_API_KEY", "")
BASE_URL = os.environ.get("MODELUX_BASE_URL", "https://api.modelux.ai/v1")
MODEL = os.environ.get("MODELUX_MODEL", "@default")

pytestmark = pytest.mark.skipif(SKIP, reason="MODELUX_INTEGRATION not set")


@pytest.fixture
def client():
    c = Modelux(api_key=API_KEY, base_url=BASE_URL)
    yield c
    c.close()


@pytest.fixture
def async_client():
    return AsyncModelux(api_key=API_KEY, base_url=BASE_URL)


class TestSync:
    def test_non_streaming_completion(self, client):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say hello in exactly 3 words"}],
            max_tokens=20,
        )

        assert len(response.choices) > 0
        assert response.choices[0].message.content
        assert response.modelux.request_id
        assert response.modelux.provider_used
        assert response.modelux.model_used
        assert isinstance(response.modelux.cache_hit, bool)

    def test_streaming_completion(self, client):
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Count from 1 to 3"}],
            max_tokens=30,
            stream=True,
        )

        # Metadata available immediately from response headers
        assert stream.modelux.request_id
        assert stream.modelux.provider_used
        assert stream.modelux.model_used

        chunks = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                chunks.append(delta)

        assert len(chunks) > 0
        text = "".join(chunks)
        assert len(text) > 0

    def test_modelux_params_accepted(self, client):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say ok"}],
            max_tokens=5,
            user_id="integration-test-user",
            tags={"source": "integration_test", "tier": "test"},
            trace_id="integration-trace-001",
            no_cache=True,
        )

        assert response.choices[0].message.content
        assert response.modelux.cache_hit is False  # no_cache=True

    def test_routing_config_selector(self, client):
        response = client.chat.completions.create(
            model="@default",
            messages=[{"role": "user", "content": "Say yes"}],
            max_tokens=5,
        )

        assert response.choices[0].message.content
        assert response.modelux.provider_used

    def test_invalid_api_key(self):
        bad_client = Modelux(
            api_key="mlx_sk_invalid",
            base_url=BASE_URL,
        )
        with pytest.raises(Exception):
            bad_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
        bad_client.close()


class TestAsync:
    @pytest.mark.asyncio
    async def test_async_completion(self, async_client):
        async with async_client as client:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=10,
            )

            assert response.choices[0].message.content
            assert response.modelux.request_id
            assert response.modelux.provider_used

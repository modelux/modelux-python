import json
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import httpx
import pytest

from modelux import Modelux


SAMPLE_COMPLETION = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

SAMPLE_MODELUX_HEADERS = {
    "x-modelux-request-id": "req-abc-123",
    "x-modelux-provider-used": "openai",
    "x-modelux-model-used": "gpt-4o",
    "x-modelux-cache": "MISS",
}


def make_httpx_response(
    body: Any,
    status: int = 200,
    headers: Optional[Dict[str, str]] = None,
) -> httpx.Response:
    """Build a mock httpx.Response."""
    all_headers = {"content-type": "application/json"}
    if headers:
        all_headers.update(headers)
    return httpx.Response(
        status_code=status,
        headers=all_headers,
        content=json.dumps(body).encode(),
        request=httpx.Request("POST", "https://test.modelux.ai/v1/chat/completions"),
    )


@pytest.fixture
def mock_transport():
    """Returns a mock httpx transport and the Modelux client using it."""
    transport = MagicMock(spec=httpx.BaseTransport)
    return transport


@pytest.fixture
def client_with_transport(mock_transport):
    """Create a Modelux client with a mock transport for intercepting requests."""
    import openai

    http_client = httpx.Client(transport=mock_transport)
    oai = openai.OpenAI(
        api_key="mlx_sk_test",
        base_url="https://test.modelux.ai/v1",
        http_client=http_client,
        default_headers={"X-Modelux-SDK": "python/0.1.0"},
    )
    client = Modelux.__new__(Modelux)
    from modelux._client import Chat
    client._client = oai
    client.chat = Chat(oai)
    return client, mock_transport

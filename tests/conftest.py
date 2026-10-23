"""测试共享的假对象：假响应、假 client、响应工厂。

**这里不需要 mock 框架**——因为 achat 接受一个 client 参数，
这是依赖注入带来的天然可测性。
"""

from typing import Any

import httpx
import pytest


def http_status_error(status_code: int) -> httpx.HTTPStatusError:
    """构造一个「真实」的 httpx 异常。

    为什么不能用普通 Exception 代替：
    achat_with_retry 捕获的是 httpx.HTTPStatusError，
    异常类型不匹配的话，重试逻辑根本不会触发。
    """
    request = httpx.Request("POST", "https://fake.local/chat/completions")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}", request=request, response=response
    )


class FakeResponse:
    """假的 httpx.Response。"""

    def __init__(self, item: dict[str, Any] | int) -> None:
        if isinstance(item, int):
            self._status_code = item
            self._payload: dict[str, Any] = {"error": "fake error"}
        else:
            self._status_code = 200
            self._payload = item

    @property
    def status_code(self) -> int:
        return self._status_code

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self._status_code >= 400:
            raise http_status_error(self._status_code)


class FakeAsyncClient:
    """假的 httpx.AsyncClient。

    script 里每一项要么是响应体（dict），要么是 HTTP 状态码（int，表示失败）。
    按顺序返回，同时把收到的请求体记到 calls 里，方便断言「发出去的请求长什么样」。
    """

    def __init__(self, script: list[dict[str, Any] | int]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def post(
        self, url: str, json: dict[str, Any], headers: dict[str, str]
    ) -> FakeResponse:
        self.calls.append(json)
        if not self._script:
            raise AssertionError(
                f"脚本用完了，但收到了第 {len(self.calls)} 次请求"
                "——说明代码发的请求比预期多"
            )
        return FakeResponse(self._script.pop(0))

    async def aclose(self) -> None:
        """为了匹配 ChatClient 协议而存在。

        实际不会被调用——因为假对象总是从外面注入的（owns_client 为 False），
        而 achat 只会在「自己创建了 client」时才去关闭它。
        """


# ---------------- 响应工厂 ----------------


def text_response(text: str) -> dict[str, Any]:
    """伪造一个「正常回答」的响应。"""
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def tool_call_response(
    name: str, arguments: str, call_id: str = "call_test_1"
) -> dict[str, Any]:
    """伪造一个「要求调用工具」的响应。"""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 200, "completion_tokens": 30},
    }

class AlwaysTimeoutClient:
    """模拟「每次调用都超时」的客户端。"""

    def __init__(self) -> None:
        self.calls = 0

    async def post(self, *args: object, **kwargs: object) -> None:
        self.calls += 1
        raise TimeoutError("simulated timeout")

    async def aclose(self) -> None:
        pass

@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")

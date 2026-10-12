"""LLM 客户端的测试——全部用假 client，不花一分钱。

注意：这里的测试函数是 async def，但**没有加任何装饰器**——
因为 pyproject.toml 里配了 asyncio_mode = "auto"。
"""

import httpx
import pytest
from conftest import FakeAsyncClient, text_response, tool_call_response

from agent_playground.config import Settings
from agent_playground.llm_client import achat, achat_with_retry


async def test_achat_returns_answer() -> None:
    """正常响应：拿到 answer，tool_calls 为空。"""
    client = FakeAsyncClient([text_response("你好")])

    result = await achat([{"role": "user", "content": "hi"}], Settings(), client=client)

    assert result["answer"] == "你好"
    assert result["tool_calls"] == []
    assert result["usage"]["prompt_tokens"] == 100


async def test_achat_normalizes_tool_calls() -> None:
    """★ tool_calls 应该被归一化成 [{id, name, arguments}]。

    归一化是刻意的设计：调用方（arun_agent）不需要知道
    API 原始格式是嵌套的 function.name / function.arguments。
    """
    client = FakeAsyncClient(
        [tool_call_response("calculator", '{"expression": "1+1"}')]
    )

    result = await achat(
        [{"role": "user", "content": "1+1"}], Settings(), client=client
    )

    assert result["tool_calls"] == [
        {
            "id": "call_test_1",
            "name": "calculator",
            "arguments": '{"expression": "1+1"}',
        }
    ]


async def test_achat_content_none_becomes_empty_string() -> None:
    """★ 模型调用工具时 content 是 None——必须转成空串，不能返回 None。

    这个坑很隐蔽：下游如果用 .strip() 或字符串拼接，None 会直接崩。
    """
    client = FakeAsyncClient([tool_call_response("calculator", "{}")])

    result = await achat([{"role": "user", "content": "x"}], Settings(), client=client)

    assert result["answer"] == ""
    assert isinstance(result["answer"], str)


async def test_achat_sends_tools_when_provided() -> None:
    """传了 tools 时，请求体里应该带上。"""
    client = FakeAsyncClient([text_response("ok")])
    tools = [{"type": "function", "function": {"name": "calc"}}]

    await achat(
        [{"role": "user", "content": "x"}],
        Settings(),
        tools=tools,
        client=client,
    )

    assert client.calls[0]["tools"] == tools


async def test_achat_omits_tools_when_not_provided() -> None:
    """★ 没传 tools 时，请求体里**不该有**这个字段。

    有些服务端对 tools: null 会报错，所以必须是「不存在」而不是「null」。
    """
    client = FakeAsyncClient([text_response("ok")])

    await achat([{"role": "user", "content": "x"}], Settings(), client=client)

    assert "tools" not in client.calls[0]


async def test_achat_raises_on_http_error() -> None:
    """★ 4xx/5xx 应该抛异常——由上层决定怎么处理。"""
    client = FakeAsyncClient([500])

    with pytest.raises(httpx.HTTPStatusError):
        await achat([{"role": "user", "content": "x"}], Settings(), client=client)


# ---------------- 重试逻辑 ----------------


async def test_retry_succeeds_after_transient_failures() -> None:
    """★ 前两次 500，第三次成功——重试应该救回来。

    base_backoff=0 让退避不真的等待，测试瞬间跑完。
    （这也是「让时间可注入」的收益——Day 2 讲过的原则）
    """
    client = FakeAsyncClient([500, 500, text_response("终于成功了")])

    result = await achat_with_retry(
        [{"role": "user", "content": "x"}],
        Settings(),
        client=client,
        base_backoff=0.0,
    )

    assert result["answer"] == "终于成功了"
    assert len(client.calls) == 3  # 确实试了三次


async def test_retry_gives_up_on_client_error() -> None:
    """★ 400 不该重试——参数错了，重试一万次也一样。"""
    client = FakeAsyncClient([400, text_response("不该走到这里")])

    with pytest.raises(httpx.HTTPStatusError):
        await achat_with_retry(
            [{"role": "user", "content": "x"}],
            Settings(),
            client=client,
            base_backoff=0.0,
        )

    assert len(client.calls) == 1  # 只试了一次就放弃


async def test_retry_exhausts_and_raises() -> None:
    """★ 一直失败时，重试次数用尽后抛异常。"""
    client = FakeAsyncClient([500, 500, 500])

    with pytest.raises(httpx.HTTPStatusError):
        await achat_with_retry(
            [{"role": "user", "content": "x"}],
            Settings(),
            client=client,
            max_attempts=3,
            base_backoff=0.0,
        )

    assert len(client.calls) == 3

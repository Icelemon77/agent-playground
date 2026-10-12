"""Agent 主循环的测试——今天最重要的一组。

用假 client 模拟「模型先要工具、再给答案」的完整过程。
重点验证 Day 3 踩过的坑：工具结果必须以 role=tool + tool_call_id 回传。
"""

from typing import Any

from conftest import FakeAsyncClient, text_response, tool_call_response

from agent_playground.agent import MAX_STEPS, arun_agent
from agent_playground.config import Settings

# ---------------- 正常链路 ----------------


async def test_agent_executes_tool_then_answers() -> None:
    """★ 核心测试：完整链路走通。

    第一次响应要求调用工具 → 执行 → 结果回传 → 第二次响应给出答案。
    """
    client = FakeAsyncClient(
        [
            tool_call_response("calculator", '{"expression": "3 ** 15"}'),
            text_response("3 的 15 次方是 14348907。"),
        ]
    )

    result = await arun_agent("3 的 15 次方是多少？", Settings(), client=client)

    assert result["steps"] == 2
    assert result["error"] is None
    assert "14348907" in result["answer"]
    # token 用量应该是两次响应之和
    assert result["prompt_tokens"] == 300  # 200 + 100
    assert result["completion_tokens"] == 50  # 30 + 20


async def test_agent_passes_tool_result_back_as_tool_message() -> None:
    """★★ 验证工具结果是以 role=tool + tool_call_id 回传的。

    这是 Day 3 踩过的坑：漏了 tool_call_id 接口会返回 400。
    这个测试把「正确的消息格式」锁住了。
    """
    client = FakeAsyncClient(
        [
            tool_call_response("calculator", '{"expression": "2+2"}'),
            text_response("等于 4"),
        ]
    )

    await arun_agent("2+2 等于几？", Settings(), client=client)

    second_request = client.calls[1]["messages"]

    # ① 有一条 role=tool 的结果消息，带着正确的 id
    tool_messages = [m for m in second_request if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "4"
    assert tool_messages[0]["tool_call_id"] == "call_test_1"

    # ② assistant 那条消息带着 tool_calls（模型要看到自己请求过什么）
    assistant_messages = [m for m in second_request if m["role"] == "assistant"]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["tool_calls"][0]["function"]["name"] == "calculator"


async def test_agent_answers_directly_without_tools() -> None:
    """★ 模型不需要工具时，一步就结束（不该白等一轮）。"""
    client = FakeAsyncClient([text_response("我是助手")])

    result = await arun_agent("你是谁？", Settings(), client=client)

    assert result["steps"] == 1
    assert result["answer"] == "我是助手"
    assert len(client.calls) == 1


# ---------------- 兜底与错误路径 ----------------


async def test_agent_stops_at_max_steps() -> None:
    """★ 模型一直要工具时，MAX_STEPS 必须兜住。

    脚本里塞满 MAX_STEPS 个「要工具」的响应，
    验证：循环停在 MAX_STEPS，不会多发一次请求。
    """
    script: list[dict[str, Any] | int] = [
        tool_call_response("calculator", '{"expression": "1+1"}')
    ] * MAX_STEPS
    script.append(text_response("不该走到这里"))
    client = FakeAsyncClient(script)

    result = await arun_agent("死循环测试", Settings(), client=client)

    assert result["steps"] == MAX_STEPS
    assert "最大步数" in result["answer"]
    assert len(client.calls) == MAX_STEPS  # 没有多发第 MAX_STEPS+1 次


async def test_agent_reports_error_on_max_steps() -> None:
    """★ 「达到最大步数」也是一种失败，error 字段必须有值。

    为什么重要：batch.py 用 r["error"] 判断成败——
    如果这条返回里没有 error 字段，批量任务会直接 KeyError 崩掉。
    """
    script: list[dict[str, Any] | int] = [
        tool_call_response("calculator", '{"expression": "1+1"}')
    ] * MAX_STEPS
    client = FakeAsyncClient(script)

    result = await arun_agent("死循环测试", Settings(), client=client)

    assert result["error"] == "MaxStepsReached"


async def test_agent_catches_http_error() -> None:
    """★ HTTP 错误被兜住，返回带 error 的结果而不是抛异常。

    用 400 触发（不可重试）——避免走重试退避，测试才能秒完成。
    重试路径的测试在 test_llm_client.py 里。
    """
    client = FakeAsyncClient([400])

    result = await arun_agent("网络错误测试", Settings(), client=client)

    assert result["error"] == "HTTPStatusError"
    assert result["steps"] == 0
    assert "网络错误" in result["answer"]

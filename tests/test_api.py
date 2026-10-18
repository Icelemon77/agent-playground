"""HTTP 接口层的测试。

为什么值得测：
  1. **接口是用户唯一会碰的地方**——CLI 出了 bug 只有你用，接口出了 bug 所有人都中
  2. **证明「没有第二份实现」**：/ask 必须调用 rag.answer，不能自己再拼一遍 prompt
  3. **错误路径最容易坏**：异常处理里少个 import，正常请求全过，模型一挂就 500

所有用例都用探针（把 answer / generate / get_index 换成假的），
所以**不联网、不花钱、0.1 秒跑完**。
"""

import json

import pytest
from fastapi.testclient import TestClient

import agent_playground.api as api_module
from agent_playground.api import app
from agent_playground.rag import TfidfIndex


@pytest.fixture()
def fake_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TfidfIndex, tuple[str, ...]]:
    """把知识库换成一个只有两个块的假索引。

    不依赖真实的 docs/：文档一改，测试结果就跟着变，那种测试没法当回归用。
    """
    index = TfidfIndex().fit(
        ["### 一、压缩\n不做压缩时输入 2175 tokens", "### 二、并发\n并发加速 4.1x"]
    )
    sources = ("docs/a.md", "docs/b.md")
    monkeypatch.setattr(api_module, "get_index", lambda: (index, sources))
    return index, sources


def test_root_serves_landing_page() -> None:
    """根路径是陌生人点开链接看到的第一屏，不能是 404。"""
    r = TestClient(app).get("/")

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "试试问这些" in r.text
    assert "/docs" in r.text


def test_health_reports_corpus_without_leaking_key(
    fake_corpus: tuple[TfidfIndex, tuple[str, ...]],
) -> None:
    """健康检查要报出语料状态，但绝不能把配置泄露出去。"""
    index, _sources = fake_corpus
    r = TestClient(app).get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["chunks"] == 2
    assert body["files"] == 2
    assert body["fingerprint"] == index.fingerprint()
    # 健康检查是最容易泄露配置的地方
    assert "api_key" not in r.text.lower()


def test_health_degraded_when_corpus_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """空知识库是「进程活着但答不了」，要能被区分出来。"""
    empty = TfidfIndex().fit([])
    monkeypatch.setattr(api_module, "get_index", lambda: (empty, ()))

    body = TestClient(app).get("/health").json()

    assert body["status"] == "degraded"
    assert body["chunks"] == 0


def test_search_returns_hits_with_source(
    fake_corpus: tuple[TfidfIndex, tuple[str, ...]],
) -> None:
    """/search 只做检索，返回的每一块都要带来源文件。"""
    _index, sources = fake_corpus
    r = TestClient(app).post("/search", json={"query": "压缩 token", "top_k": 1})

    assert r.status_code == 200
    hits = r.json()["hits"]
    assert len(hits) == 1
    assert hits[0]["source"] in sources
    assert 0 < hits[0]["score"] <= 1


def test_ask_delegates_to_rag_answer(
    monkeypatch: pytest.MonkeyPatch,
    fake_corpus: tuple[TfidfIndex, tuple[str, ...]],
) -> None:
    """/ask 不能有第二份实现——它必须调用 rag.answer。

    这是「API 和 CLI 同一条链路」的证据：把 answer 换成探针，
    如果接口自己拼了一遍 prompt，这个测试会失败。
    """
    seen: dict[str, object] = {}

    def fake_answer(
        question: str,
        index: TfidfIndex,
        settings: object,
        *,
        top_k: int = 3,
    ) -> dict:
        seen["question"] = question
        seen["top_k"] = top_k
        return {
            "answer": "探针答案",
            "hits": [(0.9, 0, "块一"), (0.8, 0, "块一"), (0.5, 1, "块二")],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            "elapsed": 0.5,
        }

    monkeypatch.setattr(api_module, "answer", fake_answer)
    r = TestClient(app).post("/ask", json={"question": "探针问题", "top_k": 2})

    assert r.status_code == 200
    assert seen == {"question": "探针问题", "top_k": 2}
    body = r.json()
    assert body["answer"] == "探针答案"
    # 两个命中来自同一个文件 → 去重后只剩一个来源，并且按名字排序
    assert body["sources"] == ["docs/a.md", "docs/b.md"]
    assert body["prompt_tokens"] == 12
    assert body["completion_tokens"] == 3


def test_ask_returns_502_with_error_id_and_no_stack(
    monkeypatch: pytest.MonkeyPatch,
    fake_corpus: tuple[TfidfIndex, tuple[str, ...]],
) -> None:
    """模型挂了要返 502 + 错误编号，而且不能把内部细节泄露给客户端。"""

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("上游炸了，这句话不该出现在响应里")

    monkeypatch.setattr(api_module, "answer", boom)
    r = TestClient(app).post("/ask", json={"question": "探针"})

    assert r.status_code == 502
    detail = r.json()["detail"]
    assert "错误编号" in detail
    # 「不暴露」的三条断言 —— 少一条都可能泄露
    assert "RuntimeError" not in detail
    assert "Traceback" not in detail
    assert "上游炸了" not in detail


@pytest.mark.parametrize("top_k", [0, 99])
def test_top_k_out_of_range_is_422(
    fake_corpus: tuple[TfidfIndex, tuple[str, ...]],
    top_k: int,
) -> None:
    """越界参数由 Pydantic 在进入函数前拦下，返回结构化错误。"""
    r = TestClient(app).post("/search", json={"query": "x", "top_k": top_k})

    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "top_k"]


def test_stream_pushes_sources_then_deltas(
    monkeypatch: pytest.MonkeyPatch,
    fake_corpus: tuple[TfidfIndex, tuple[str, ...]],
) -> None:
    """/ask/stream 的 SSE 顺序：先给来源，再逐片给正文，最后 [DONE]。"""

    def fake_generate(
        question: str,
        hits: list[tuple[float, int, str]],
        settings: object,
        *,
        on_chunk: object = None,
    ) -> dict:
        for piece in ["你好", "，世界"]:
            assert callable(on_chunk)
            on_chunk(piece)
        return {"answer": "你好，世界", "usage": {}, "elapsed": 0.0}

    monkeypatch.setattr(api_module, "generate", fake_generate)

    with TestClient(app).stream(
        # 查询词要能在假语料里命中，否则会走「检索没命中」那条分支
        "POST",
        "/ask/stream",
        json={"question": "压缩 token"},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        payloads = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]

    assert "sources" in payloads[0]
    assert payloads[-1] == "[DONE]"
    text = "".join(json.loads(p)["delta"] for p in payloads[1:-1])
    assert text == "你好，世界"

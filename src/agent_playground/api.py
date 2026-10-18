"""HTTP 接口层：把已有的检索与生成能力暴露成服务。

这一层只做协议转换——接收请求、调用 rag/agent 的现有函数、返回结构化响应。
不在这里重写检索或生成逻辑，保证 CLI、Agent、API 三条入口行为一致。
"""
import json
import queue
import threading
from collections.abc import Iterator
from fastapi import FastAPI,HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


from agent_playground.config import Settings
from agent_playground.rag import (
    get_index,
    DEFAULT_TOP_K,
    answer,
    generate,
    retrieve,
)


settings = Settings()

app = FastAPI(
    title="agent-playground",
    description="从零手写的 AI Agent 系统的 HTTP 接口",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    """健康检查的响应。"""

    status: str = Field(description="ok=可正常工作；degraded=进程活着但知识库是空的")
    chunks: int = Field(description="知识库切块数量")
    files: int = Field(description="知识库文件数量")
    fingerprint: str = Field(description="知识库内容指纹，用来确认线上跑的是哪一版语料")
    model: str = Field(description="当前使用的大模型")


@app.get("/health", response_model=HealthResponse, summary="健康检查")
def health() -> HealthResponse:
    index, sources = get_index()
    return HealthResponse(
        status="ok" if index.size > 0 else "degraded",
        chunks=index.size,
        files=len(set(sources)),
        fingerprint=index.fingerprint(),
        model=settings.llm_model,
    )

class Hit(BaseModel):
    """一个命中片段"""
    score: float = Field(description="余弦相似度，越大越相关")
    chunk: str = Field(description="命中的文本块")
    source: str = Field(description="来自哪个文件")

class SearchRequest(BaseModel):
    query: str = Field(min_length=1,max_length=200,description="检索关键词")
    top_k: int = Field(default=DEFAULT_TOP_K,ge=1,le=10,description="返回几条")

class SearchResponse(BaseModel):
    query: str
    hits: list[Hit]

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="要问的问题")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10)

class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(description="答案引用的文件，去重后按名字排序")
    hits: list[Hit]
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float

def to_hits(raw: list[tuple[float,int,str]],sources: tuple[str,...]) -> list[Hit]:
    return [
        Hit(score=round(score,4),chunk=chunk,source=sources[index])
        for score, index, chunk in raw
    ]

@app.post("/search", response_model=SearchResponse, summary="只检索，不生成")
def search(req: SearchRequest) -> SearchResponse:
    index, sources = get_index()
    raw = retrieve(req.query, index, top_k=req.top_k)
    return SearchResponse(query=req.query, hits=to_hits(raw, sources))

@app.post("/ask", response_model=AskResponse, summary="问答（非流式）")
def ask(req: AskRequest) -> AskResponse:
    index, sources = get_index()
    try:
        result = answer(req.question, index, settings, top_k=req.top_k)
    except Exception as exc:      # 见下面「为什么这里可以宽」
        raise HTTPException(status_code=502, detail="大模型调用失败，请稍后重试") from exc

    usage = result.get("usage") or {}
    raw = result.get("hits", [])
    return AskResponse(
        answer=result["answer"],
        sources=sorted({sources[i] for _score, i, _chunk in raw}),
        hits=to_hits(raw, sources),
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        elapsed_s=round(float(result.get("elapsed", 0.0)), 3),
    )

def sse(payload: dict) -> str:
    """拼一条 SSE 消息。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/ask/stream", summary="流式问答（SSE）")
def ask_stream(req: AskRequest) -> StreamingResponse:
    index, sources = get_index()
    raw = retrieve(req.question, index, top_k=req.top_k)

    def events() -> Iterator[str]:
        # 先把引用来源推给前端，让它在等待答案时就能显示出处
        yield sse({"sources": sorted({sources[i] for _s, i, _c in raw})})

        if not raw:
            yield sse({"delta": "资料中没有相关信息（检索没有命中任何文档块）"})
            yield "data: [DONE]\n\n"
            return

        chunks: queue.Queue[str | None] = queue.Queue()

        def worker() -> None:
            try:
                generate(req.question, raw, settings, on_chunk=chunks.put)
            finally:
                chunks.put(None)          # 哨兵：告诉消费者「结束了」

        threading.Thread(target=worker, daemon=True).start()

        while (piece := chunks.get()) is not None:
            yield sse({"delta": piece})
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
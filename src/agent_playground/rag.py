"""最小 RAG：切块 → 稀疏向量索引 → 余弦相似度检索 → 拼 prompt 交给模型。

设计上有两个刻意的选择：

1. **向量化独立成函数**（`TfidfIndex._vectorize`）。今天用 TF-IDF 稀疏向量，
   是因为当前服务商没有 embedding 接口。将来换成稠密向量只需替换这一个方法。
2. **`fit` 和 `search` 分开**。建索引是离线的一次性工作，检索是在线的反复调用。
   把两件事混在一起，就没法在启动时建一次、之后反复查。
"""

import hashlib
import math
import re
import sys
from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Self

from agent_playground.config import Settings
from agent_playground.llm_client import stream_chat

DEFAULT_CHUNK_SIZE = 300
DEFAULT_OVERLAP = 50
DEFAULT_TOP_K = 3
DOCS_DIR = Path("docs")
EXTRA_DOCS = (Path("README.md"), Path("NEEDS.md"))
# 评测报告不进检索语料。它逐题讨论了评测集的问题、示例里还复制了 expect 的取值，
# 进了语料就同时造成两件事：答案多了一份副本（假阳性），
# 以及模型会读到「第 N 题…没有给出具体数值」这类失败分析、照着拒答。
# 实测：加入这份报告后端到端从 0.93 掉到 0.83，模型开始逐题讨论「第 14 题」。
NON_RETRIEVABLE_DOCS = ("rag_eval.md",)

PROMPT_TEMPLATE = """【背景资料】
{context}

【问题】{question}

请只依据上面的背景资料回答。如果资料里没有相关信息，直接回答「资料中没有相关信息」，
不要用自己的知识补充。回答时用 [1][2] 这样的编号标注引用了哪段资料。"""

_CJK = r"\u4e00-\u9fff"
_TOKEN_RE = re.compile(rf"[{_CJK}]+|[A-Za-z0-9_]+")
_HEADING_RE = re.compile(r"^#{1,6}\s+\S")


def tokenize(text: str) -> list[str]:
    """把文本切成检索用的「词」。

    中文没有空格，又不引入分词依赖，所以用**字符 bigram**（两个字一组）：
    「上下文压缩」→ 上下 / 下文 / 文压 / 压缩。
    英文和数字按整词处理并转小写。
    """
    tokens: list[str] = []
    for part in _TOKEN_RE.findall(text):
        if part.isascii():
            tokens.append(part.lower())
        elif len(part) == 1:
            tokens.append(part)
        else:
            tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
    return tokens


def split_sections(text: str) -> list[tuple[str, str]]:
    """按 Markdown 标题把文档切成 [(标题, 正文), ...]。

    没有标题的文档会返回 [("", 全文)]，下游逻辑不用分两种情况。

    要跳过围栏代码块里的行——`# 这是一句 bash 注释` 长得和标题一样，
    但它不是章节边界，当成标题会把一段代码从中间劈开。
    """
    sections: list[tuple[str, list[str]]] = []
    title = ""
    body: list[str] = []
    in_fence = False

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            body.append(line)
            continue
        if not in_fence and _HEADING_RE.match(line):
            if title or any(part.strip() for part in body):
                sections.append((title, body))
            title = line.strip()
            body = []
        else:
            body.append(line)

    if title or any(part.strip() for part in body):
        sections.append((title, body))
    return [(head, "\n".join(lines)) for head, lines in sections]


def _split_oversized(para: str, size: int) -> list[str]:
    """超长段落按行切成不超过 size 的片段；表格片段会重复表头。

    按字符硬切（`para[i:i+size]`）会把表格的一行劈成两半，切出来的碎块
    既没有列名也没有版本号，检索时几乎不可能命中。实测 ReAct 范式对照表
    就被这样切开过，模型因此回答「表格在此处被截断，无法计算」。

    表格按行切、并在每段重复表头，碎片就仍然是自解释的。
    """
    if len(para) <= size:
        return [para]

    lines = para.splitlines()
    is_table = len(lines) >= 3 and all(
        line.strip().startswith("|") for line in lines if line.strip()
    )
    header = lines[:2] if is_table else []
    body = lines[2:] if is_table else lines

    parts: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            parts.append("\n".join([*header, *buf]))
            buf.clear()

    for line in body:
        if len(line) > size:  # 单行本身超长，只能按字符切
            flush()
            parts.extend(line[i : i + size] for i in range(0, len(line), size))
            continue
        if buf and len("\n".join([*header, *buf, line])) > size:
            flush()
        buf.append(line)
    flush()

    if not parts:
        return ["\n".join(header)] if header else [para[:size]]
    return parts


def _overlap_tail(text: str, max_chars: int) -> str:
    """截取用于重叠的尾巴，从行边界开始。

    按字符数硬截会在任意位置下刀，实测切出过以 `tion` 开头的块
    （`Observation` 被切剩的），以及以 `| **1253** | **5/5** |` 这种
    残缺表格行开头的块——丢了列名和版本号，检索几乎不可能命中。
    从行边界开始取，就不会把一整行劈成两半。
    """
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    newline = tail.find("\n")
    if newline == -1:
        return tail
    return tail[newline + 1 :]


def _chunk_plain(text: str, size: int, overlap: int) -> list[str]:
    """按空行分段，再把段落聚合成接近 size 字符的块，块间留 overlap 重叠。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    units: list[str] = []
    for para in paragraphs:
        units.extend(_split_oversized(para, size))

    chunks: list[str] = []
    buf = ""
    for unit in units:
        if buf and len(buf) + len(unit) + 1 > size:
            chunks.append(buf)
            buf = unit
        else:
            buf = f"{buf}\n{unit}" if buf else unit
    if buf:
        chunks.append(buf)

    if overlap and len(chunks) > 1:
        with_overlap = [chunks[0]]
        for prev, cur in pairwise(chunks):
            head = _overlap_tail(prev, overlap)
            with_overlap.append(f"{head}\n{cur}" if head else cur)
        chunks = with_overlap

    return chunks


def chunk_text(
    text: str,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """把长文档切成块：先按 Markdown 标题切章节，章节内再按字符数切。

    块之间保留 overlap 个字符的重叠，避免一句话刚好被切在边界上、两块都读不懂。
    标题会作为块的前缀保留——标题里往往就有关键词，检索时应该一起算权重。

    注意两个已知行为：加了重叠之后，块的实际长度是 size + overlap；
    最后一块可能不足 size，这是正常的（不为了凑整而丢掉尾巴）。
    """
    if size <= 0:
        raise ValueError(f"size 必须为正数，当前为 {size}")
    if not 0 <= overlap < size:
        raise ValueError(
            f"overlap 必须在 [0, size) 范围内，当前 overlap={overlap}, size={size}"
        )

    chunks: list[str] = []
    for title, body in split_sections(text):
        # 标题要算进 size，否则加了标题的块会明显超出预期长度
        budget = max(size - len(title) - 1, size // 2)
        for piece in _chunk_plain(body, budget, overlap):
            chunks.append(f"{title}\n{piece}" if title else piece)
    return chunks


class TfidfIndex:
    """TF-IDF 稀疏向量索引，用余弦相似度算相关性。"""

    def __init__(self) -> None:
        self._idf: dict[str, float] = {}
        self._vectors: list[dict[str, float]] = []
        self._chunks: list[str] = []

    def fit(self, chunks: list[str]) -> Self:
        """建索引（离线，只需一次）：统计每个词出现在多少个块里，算出 IDF。"""
        doc_freq: Counter[str] = Counter()
        for chunk in chunks:
            doc_freq.update(set(tokenize(chunk)))

        total = len(chunks)
        # 平滑：分母 +1 避免除零，末尾 +1 保证 IDF 恒正
        # （否则「出现在所有块里」的词 IDF 会是 0，权重被抹平）
        self._idf = {
            term: math.log((total + 1) / (freq + 1)) + 1
            for term, freq in doc_freq.items()
        }
        self._chunks = list(chunks)
        self._vectors = [self._vectorize(chunk) for chunk in chunks]
        return self

    def _vectorize(self, text: str) -> dict[str, float]:
        """把文本变成稀疏向量 {词: TF×IDF}。索引里没有的词权重为 0，直接丢掉。"""
        counts = Counter(tokenize(text))
        vector: dict[str, float] = {}
        for term, count in counts.items():
            idf = self._idf.get(term, 0.0)
            if idf > 0:
                vector[term] = count * idf
        return vector

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        """余弦相似度：点积 ÷ 两个向量的模长乘积。

        只遍历 a 里有的词——这就是稀疏向量比稠密向量快的原因。
        """
        dot = 0.0
        for term, weight in a.items():
            dot += weight * b.get(term, 0.0)
        if dot <= 0:
            return 0.0
        norm_a = math.sqrt(sum(w * w for w in a.values()))
        norm_b = math.sqrt(sum(w * w for w in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> list[tuple[float, int, str]]:
        """返回 [(相似度, 块下标, 块内容), ...]，按相似度从高到低。"""
        query_vector = self._vectorize(query)
        if not query_vector:
            return []

        hits = [
            (score, i, self._chunks[i])
            for i, vector in enumerate(self._vectors)
            if (score := self._cosine(query_vector, vector)) > 0
        ]
        hits.sort(key=lambda item: -item[0])
        return hits[:top_k]

    @property
    def size(self) -> int:
        """索引里的块数。"""
        return len(self._chunks)

    def fingerprint(self) -> str:
        """知识库内容指纹（块的 SHA256 前 12 位）。

        知识库是会变的输入：改一份文档、甚至只是更新 README，切块和排名都会跟着变，
        历史报告里的数字就不再代表当前状态。所以每次评测都要记下它对应哪个知识库。
        实测过：写完评测报告本身（`docs/` 是知识库的一部分）就让块数从 231 涨到 277，
        recall@3 从 0.94 掉到 0.896。
        """
        digest = hashlib.sha256("\n".join(self._chunks).encode("utf-8"))
        return digest.hexdigest()[:12]


def load_documents(
    docs_dir: Path | str = DOCS_DIR,
    extra: tuple[Path, ...] = EXTRA_DOCS,
) -> list[tuple[str, str]]:
    """读取知识库，返回 [(来源文件名, 正文), ...]。"""
    documents: list[tuple[str, str]] = []
    paths = sorted(Path(docs_dir).glob("*.md")) if Path(docs_dir).exists() else []
    for path in [*paths, *extra]:
        if path.is_file() and path.name not in NON_RETRIEVABLE_DOCS:
            documents.append((path.as_posix(), path.read_text(encoding="utf-8")))
    return documents


def build_index(
    docs_dir: Path | str = DOCS_DIR,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> tuple[TfidfIndex, list[str]]:
    """把知识库切块并建索引，返回 (索引, 每块对应的来源文件)。"""
    chunks: list[str] = []
    sources: list[str] = []
    for source, text in load_documents(docs_dir):
        for chunk in chunk_text(text, size=size, overlap=overlap):
            chunks.append(chunk)
            sources.append(source)
    return TfidfIndex().fit(chunks), sources


@lru_cache(maxsize=1)
def get_index() -> tuple[TfidfIndex, tuple[str, ...]]:
    """进程内只建一次索引。

    建索引是「离线」工作：读文件、切块、算 IDF。如果每次工具调用都重来一遍，
    就是在反复做同一件已经做完的事——这正是 RAG 分两阶段的意义所在。
    """
    index, sources = build_index()
    return index, tuple(sources)


def build_prompt(question: str, hits: list[tuple[float, int, str]]) -> str:
    """把命中的块编号后拼进 prompt。"""
    context = "\n\n".join(
        f"[{i}] {chunk}" for i, (_score, _index, chunk) in enumerate(hits, start=1)
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def retrieve(
    question: str,
    index: TfidfIndex,
    top_k: int = DEFAULT_TOP_K,
) -> list[tuple[float, int, str]]:
    """检索阶段：只做向量计算，不花大模型的钱。"""
    return index.search(question, top_k=top_k)


def generate(
    question: str,
    hits: list[tuple[float, int, str]],
    settings: Settings,
    *,
    on_chunk: Callable[[str], None] | None = None,
) -> dict:
    """生成阶段：把检索结果拼进 prompt 交给模型——这一步才花钱。"""
    if not hits:
        return {
            "answer": "资料中没有相关信息（检索没有命中任何文档块）",
            "usage": {},
            "elapsed": 0.0,
        }
    result = stream_chat(
        [{"role": "user", "content": build_prompt(question, hits)}],
        settings,
        on_chunk=on_chunk,
    )
    return {
        "answer": result["answer"],
        "usage": result["usage"],
        "elapsed": result["elapsed"],
    }


def answer(
    question: str,
    index: TfidfIndex,
    settings: Settings,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """检索 + 生成一步到位，返回 {answer, hits, usage, elapsed}。"""
    hits = retrieve(question, index, top_k)
    return {**generate(question, hits, settings), "hits": hits}


def main() -> int:
    settings = Settings()
    index, sources = build_index()
    question = sys.argv[1] if len(sys.argv) > 1 else "做了哪些优化？"

    print(f"知识库：{index.size} 个块（来自 {len(set(sources))} 个文件）")
    print(f"问题：{question}\n")

    hits = retrieve(question, index)
    print("检索到的块：")
    for i, (score, index_no, chunk) in enumerate(hits, start=1):
        head = chunk.replace("\n", " ")[:60]
        print(f"  [{i}] 相似度 {score:.3f} | 块 #{index_no} | {head}…")

    print("\n回答：")
    result = generate(question, hits, settings)
    usage = result["usage"]
    print(
        f"\n[输入 {usage.get('prompt_tokens', 0)} tokens"
        f" / 输出 {usage.get('completion_tokens', 0)} tokens"
        f" / {result['elapsed']:.2f}s]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# 测试笔记

> 记录日期：2026-10-12（Day 6）
> 对应实现：`tests/` 目录下的 5 个测试文件，76 个用例

---

## 一、今天的产出

| 指标 | 数值 |
|---|---|
| 测试用例 | **76 个** |
| 测试文件 | 5 个（`tests/`） |
| 运行耗时 | **0.57 秒** |
| API 调用次数 | **0 次** |
| 覆盖率 | 37% → **40%** |
| **抓到的真 bug** | **3 个** |
| 修掉的类型错误 | 17 个 |
| 新增提交 | 3 个 |

### 覆盖率分布

| 模块 | 覆盖 | 说明 |
|---|---|---|
| `config.py` | 100% | |
| `context.py` | 100% | 今天补的 |
| `cost.py` | 100% | 今天补的 |
| `prompts.py` | 100% | 纯常量 |
| `text_utils.py` | 100% | |
| `tools.py` | 89% | 剩下是罕见分支 |
| `agent.py` | 52% | **缺同步版 `run_agent`** |
| `llm_client.py` | 44% | **缺 `stream_chat`** |
| `trace.py` / `batch.py` / `cli.py` / `replay.py` | 0% | 入口脚本，集成测试范畴 |

---

## 二、为什么必须写测试

动机来自 Day 3 的一个实验：把 `calculator` 改成恒定返回 `"9999"`，然后问它 3 的 15 次方——

**模型会一本正经地告诉你「3 的 15 次方等于 9999」。**

> **模型对工具的返回值是 100% 信任的。它不会质疑、不会校验、不会兜底。**

所以：

| 如果 | 后果 |
|---|---|
| 工具算错了 | Agent 会**自信地**给出错误答案 |
| 工具漏了边界情况 | 只能在真实使用中发现 |
| 重构时改了行为 | 没有任何东西会提醒你 |

**工具的可靠性就是 Agent 的可靠性上限——而这只能靠测试保证。**

---

## 三、pytest 核心概念

### 三个约定

| 约定 | 说明 |
|---|---|
| **命名** | 文件 `test_*.py`，函数 `test_*`，pytest 自动发现 |
| **断言** | 直接写裸 `assert`，不需要 `assertEqual` 那一套 |
| **配置** | `pyproject.toml` 里的 `[tool.pytest.ini_options]` |

### 项目里的配置

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]        # 只扫 tests/，不扫全项目
asyncio_mode = "auto"        # async 测试不需要 @pytest.mark.asyncio 装饰器
```

### 两个内置 fixture

| fixture | 作用 |
|---|---|
| `tmp_path` | 提供全新的空临时目录——**测试不该依赖仓库里的文件** |
| `monkeypatch` | 临时修改环境（如 `chdir`、属性），**测试结束自动还原** |

```python
def test_read_file_returns_content(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)              # 切到临时目录
    (tmp_path / "data").mkdir()              # 造数据
    (tmp_path / "data" / "sample.txt").write_text("hello", encoding="utf-8")

    assert read_file("sample.txt") == "hello"
```

### 参数化

```python
@pytest.mark.parametrize(
    ("text", "expected_min"),
    [("短", 1), ("中等长度" * 10, 50)],
)
def test_compact_never_returns_empty(text: str, expected_min: int) -> None:
    assert len(compact_observation(text)) >= expected_min
```

一份逻辑跑多组输入。

---

## 四、测试的优先级：边界 > 正常路径

| 优先级 | 测什么 | 例子 |
|---|---|---|
| **高** | **边界** | 空输入、超长、恰好等于阈值、特殊字符 |
| **高** | **错误路径** | 文件不存在、格式非法、路径穿越、权限不足 |
| 中 | 正常路径 | 常规输入 |

**因为出错永远在边界上。**

### 项目里的边界用例

| 测试 | 边界 |
|---|---|
| `test_count_tokens_nonempty_returns_at_least_one` | 3 字符 ÷ 4 = 0，但函数保证最小 1 |
| `test_truncate_exactly_at_limit_unchanged` | 恰好等于上限（`<=` vs `<`） |
| `test_parse_env_pairs_value_containing_equals` | 值里含 `=`（`partition` 只切第一个） |
| `test_build_headers_empty_key_ignored` | 空字符串 key 是**假值** |
| `test_merge_config_does_not_mutate_base` | **副作用**：不该修改传入的字典 |

**最后一条特别值得注意**：它测的不是「返回值对不对」，而是「**函数有没有偷偷干别的事**」。正常路径的测试永远发现不了这种问题。

---

## 五、Mock LLM 调用：不需要 mock 框架

### 原因：代码已经支持依赖注入

```python
async def achat(messages, settings, *, tools=None, client=None):
    #                                              └────┬────┘
    #                                        依赖从外面传进来
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(...)      # ← 只有没传时才自建
```

**所以直接塞一个假对象进去就行。**

### 假对象的设计

```python
class FakeAsyncClient:
    """script 里每一项要么是响应体（dict），要么是状态码（int，表示失败）。"""

    def __init__(self, script: list[dict | int]) -> None:
        self._script = list(script)
        self.calls: list[dict] = []          # ← 记录收到的请求

    async def post(self, url, json, headers) -> FakeResponse:
        self.calls.append(json)
        return FakeResponse(self._script.pop(0))
```

| 字段 | 作用 |
|---|---|
| `script` | 模拟**多轮交互**（第一次要工具、第二次给答案） |
| **`calls`** | **让你能断言「发出去的请求长什么样」** |

**`calls` 才是关键**——不记录请求的话，你只能验证「最终答案对不对」；记录了之后，你能验证「中间过程对不对」。

### 最有价值的测试

```python
async def test_agent_passes_tool_result_back_as_tool_message() -> None:
    """★★ 验证工具结果是以 role=tool + tool_call_id 回传的。"""
    client = FakeAsyncClient([
        tool_call_response("calculator", '{"expression": "2+2"}'),
        text_response("等于 4"),
    ])

    await arun_agent("2+2 等于几？", Settings(), client=client)

    second_request = client.calls[1]["messages"]      # ← 检查第二次请求
    tool_messages = [m for m in second_request if m["role"] == "tool"]

    assert tool_messages[0]["content"] == "4"
    assert tool_messages[0]["tool_call_id"] == "call_test_1"
```

**它把 Day 3 踩过的坑（漏了 `tool_call_id` 会 400）变成了永久断言。**

---

## 六、测试抓到的 3 个真 bug

做法是**先写「期望的正确行为」，跑一遍看它失败**——失败就证明 bug 存在。

### Bug ① 参数极小时，压缩反而让文本变长

```python
half = max_chars // 2         # max_chars=1 → half = 0
return f"{text[:half]}...{text[-half:]}"
#                             └────┬────┘
#                          text[-0:] 等于 text[0:] = 整个字符串！
```

| | 结果 |
|---|---|
| 原文 | 100 字符 |
| 压缩后 | **122 字符（比原文还长）** |

**`-0` 就是 `0`** —— 一个非常容易忽略的细节。

**修法**：加退化分支
```python
    if half == 0:
        return text[:max_chars] + "..."
```

### Bug ② 遇到重复行时，压缩完全失效

```python
kept = [ln for ln in lines if ln in keep]      # ← 按「值」匹配
```

30 行完全相同的 `"INFO 重复行"`，只要 `keep` 里有 1 个，就**全部被保留**：

| | 行数 |
|---|---|
| 原文 | 35 行 |
| 压缩后 | **37 行（比原文还多）** |

**这是「按值 vs 按位置」的经典混淆**——`in` 判断相等，不判断「同一个元素」。

**修法**：改用行号追踪
```python
    important_idx = [i for i, ln in enumerate(lines) if any(...)]
    normal_idx = [i for i in range(len(lines)) if i not in set(important_idx)]
    ...
    kept = [lines[i] for i in sorted(keep_idx)]
```

### Bug ③ `max_lines` 很小时静默丢数据

```python
important[: max_lines - 3]        # max_lines=2 → important[:-1]
```

**负数切片不知道自己在做什么**——`[:-1]` 丢掉最后一个元素，而且**不报错**。

**修法**：`max(1, max_lines - 3)`

### 这三个 bug 的共同点

| 特征 | 说明 |
|---|---|
| **都不报错** | 程序照常运行 |
| **都在边界上** | 参数极小、内容重复 |
| **正常路径测不出来** | 只有边界用例能抓到 |
| **其中一个潜藏了两天** | Bug ② 从 Day 4 就存在，一直没被发现 |

---

## 七、类型系统与可测试性的冲突

修完 bug 后，pyright 突然报了 **17 个错误**：

```
Argument of type "FakeAsyncClient" cannot be assigned to parameter "client"
of type "AsyncClient | None"
```

**原因**：`client` 参数的类型标注写的是 `httpx.AsyncClient`——**要求传入真实的 httpx 对象**。而测试传的是假对象。

> **运行时不关心对象是什么类（鸭子类型），但类型检查器关心。**

### 解法：Protocol

描述「需要满足什么接口」，而不是「必须是这个类」：

```python
class ResponseLike(Protocol):
    @property
    def status_code(self) -> int: ...
    def json(self) -> dict[str, Any]: ...
    def raise_for_status(self) -> Any: ...


class ChatClient(Protocol):
    async def post(
        self, url: str, *, json: dict[str, Any], headers: dict[str, str]
    ) -> ResponseLike: ...

    async def aclose(self) -> None: ...
```

然后把标注从 `httpx.AsyncClient | None` 改成 `ChatClient | None`。

### 写这个协议时踩的两个坑

**坑 1：`raise_for_status()` 的返回类型不能写 `None`**

直觉上「检查状态码」不该有返回值，但 **httpx 的 `raise_for_status()` 返回 `Response` 自身**：

```
httpx:    def raise_for_status(self) -> Response
我的协议:  def raise_for_status(self) -> None
→ httpx.Response 不满足协议
```

改成 `-> Any` 后两边都满足。**类型检查器逼你去查了真实签名。**

**坑 2：协议不完整（漏了 `aclose`）**

```python
    finally:
        if owns_client:
            await client.aclose()      # ← 协议里没有 aclose
```

**这把一个设计问题暴露出来了**：`achat` 会调 `aclose`，协议就该声明它。

**两边都要补**：

| 文件 | 补什么 | 为什么 |
|---|---|---|
| `llm_client.py` | 协议加 `aclose` | 协议必须覆盖「调用方会用到的所有方法」 |
| `conftest.py` | 假 client 加 `aclose` | **假对象要「长得像」就该有同样的方法**，哪怕永远不会被调用 |

---

## 八、「让时间可注入」——今天用了两次

需要测「重试」「高峰时段」这类和时间相关的逻辑时，**如果函数内部直接读时钟，就没法测**。

| 改动 | 参数 | 用途 |
|---|---|---|
| `achat_with_retry` | `base_backoff: float = 0.5` | 测试传 `0`，退避瞬间完成（否则要真等 1.5 秒） |
| `is_peak_now` | `now: datetime \| None = None` | 测试传固定时间，验证「周六凌晨算不算高峰」 |

> **不要让函数内部写死「时间」和「环境」——把它们变成可注入的参数。**
>
> 否则你永远写不出稳定、快速的测试。

---

## 九、覆盖率的心态

### 看 `Missing` 列，不看百分比

```powershell
uv run pytest --cov=src/agent_playground --cov-report=term-missing
```

`Missing` 列告诉你**哪些行没被测到**——那就是下一个测试该写的地方。

### 值得测 vs 不值得测

| 值得测 | 不值得测 |
|---|---|
| 业务逻辑（工具、压缩、成本） | `__init__.py` 里的 print |
| 边界和错误路径 | 简单 getter / 数据类 |
| 编排逻辑（Agent 循环） | 第三方库的行为 |
| 纯函数 | 一次性的 CLI 打印 |

**60-80% 通常就够。100% 往往意味着花时间测了一堆无意义的东西。**

**覆盖率是「找漏网之鱼」的工具，不是 KPI。**

---

## 十、最重要的洞察：可测试性是设计出来的

看这个对比：

| 函数 | 能测吗 | 为什么 |
|---|---|---|
| `achat` | ✅ 能 | Day 5 加了 `client` 参数（**依赖注入**） |
| **`stream_chat`** | ❌ **不能** | Day 2 写的时候**内部创建 client**，没法替换 |
| `arun_agent` | ✅ 能 | 同样接受 `client` |
| **`run_agent`** | ❌ **不能** | 依赖 `stream_chat` |

**两个函数逻辑几乎一样，但一个能测一个不能——差别只在那个 `client` 参数。**

这也解释了覆盖率分布：`llm_client.py` 44%、`agent.py` 52%——**低覆盖不是「没写测试」，是「写不了测试」**。

> **可测试性不是测试阶段补出来的，是写代码时顺手做出来的。**

想补 `stream_chat` 的测试，得先重构它（加一个可选的 `client` 参数，和 `achat` 对齐）。

---

## 十一、今日自检

1. 为什么「工具的测试」比「Agent 的测试」更重要？
2. Mock LLM 调用时，为什么不需要 mock 框架？
3. 「测试行为」和「测试实现」的区别是什么？
4. 覆盖率 100% 能保证代码没问题吗？
5. 为什么 `stream_chat` 测不了，而 `achat` 能测？

---

## 十二、下一步

| 缺口 | 说明 |
|---|---|
| `trace.py` 的测试 | 纯 IO 逻辑，用 `tmp_path` 能测 |
| `stream_chat` 的可测性 | 需要先加 `client` 参数 |
| 集成测试 | `batch.py` / `cli.py` 这类入口脚本 |
| CI | 让「三项检查 + pytest」自动跑 |

**核心提醒**：

> 测试不是「额外的工作」，它是**「让你敢改代码的保险」**。
>
> 有了测试之后，改代码的心理负担会明显下降——因为你知道「如果我改坏了，1 秒内就会知道」。

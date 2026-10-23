# 可靠性与成本笔记

> 记录日期：2026-10-22
> 对应实现：`llm_client.py`（重试与超时）、`agent.py`（主循环与降级）、
> `api.py`（每日额度）、`tools.py`（工具异常处理）

---

## 一、故障模型

### 1.1 故障从哪里进来

```
用户输入 ──▶ CLI / API ──▶ Agent 主循环 ──▶ achat_with_retry ──▶ 服务商 HTTP
                                 │                    │
                                 ▼                    ▼
                              工具执行            超时 / 429 / 5xx
                                 │
                                 ▼
                          结果回填 messages
```

故障可以在四个位置发生：

1. **入口层**：额度耗尽、参数非法（API 返回 4xx，不进入 Agent）；
2. **Agent 循环**：达到 `MAX_STEPS`、工具异常、上下文过长；
3. **调用层**：连接超时、读超时、429 限流、5xx 服务端故障；
4. **工具层**：参数校验失败、文件不存在、内部异常。

### 1.2 故障清单

「现在会怎样」以 `llm_client.py` / `agent.py` 的实际代码为准，分两层看：
**调用层抛什么**，以及**Agent 层接不接得住**。

| 故障 | 现在会怎样 | 应该怎样 |
|---|---|---|
| HTTP 429 | 重试 3 次，指数退避 0.5s → 1s → 2s，**无抖动** | 加 jitter；有 `Retry-After` 时优先遵守 |
| HTTP 5xx | 重试 3 次后抛 `HTTPStatusError`；`agent` 接住，返回结构化错误 | 保持；但错误结果里要保留真实步数 |
| 连接超时 | `httpx.ConnectTimeout`（`TransportError` 子类）→ 重试 | 与读超时分开配置，快速失败 |
| 读超时 | `httpx.ReadTimeout` → 重试；若由 `asyncio.wait_for` 触发则为内置 `TimeoutError`，重试耗尽后**穿透崩栈** | 分类超时；流式路径才谈保留部分输出 |
| 4xx（400/401/403/422） | 不重试，立即 `raise`；`agent` 接住返回结构化错误 | 保持 |
| 工具执行异常 | `dispatch` / `adispatch` 捕获，转成字符串回传给模型 | 保持（模型能据此换参数重试） |
| 达到 `MAX_STEPS` | `arun_agent` 返回 `error="MaxStepsReached"`、`steps=6` | 保持 |
| 每日额度耗尽 | API 层返回 429；计数器在**进程内存**，重启清零 | 保持，但要在已知限制里写明 |
| 重试全部失败 | `raise last_error`：httpx 类异常被 Agent 接住；`TimeoutError` **崩栈** | 统一包装成 `LLMCallError`，Agent 层统一捕获 |

### 1.3 已定位的 bug：`TimeoutError` 穿透

`achat_with_retry` 用 `asyncio.wait_for` 包住每次调用：

```python
return await asyncio.wait_for(
    achat(messages, settings, tools=tools, client=client),
    timeout=settings.request_timeout,
)
```

Python 3.11+ 中，`asyncio.TimeoutError` 已经是内置 `TimeoutError` 的别名。
而错误处理只覆盖了 httpx 的异常类：

```python
except TimeoutError as exc:          # 重试逻辑里接住了
    last_error = exc
...
raise last_error                     # 但重试耗尽后又原样抛出
```

```python
# agent.py
except httpx.HTTPError as exc:       # 接不住内置 TimeoutError
```

实测继承关系：

```
ReadTimeout    -> TransportError : True
ConnectTimeout -> TransportError : True
TimeoutError   -> httpx.HTTPError: False
```

所以两条路的结果完全不同：

| 重试耗尽时的最后一个异常 | 调用层 | Agent 层 | 用户看到 |
|---|---|---|---|
| `httpx.HTTPStatusError` / `TransportError` | `raise` | 被 `except httpx.HTTPError` 接住 | `（网络错误，任务未能完成：…）` |
| 内置 `TimeoutError`（`wait_for` 触发） | `raise` | **接不住** | Python 堆栈 |

**结论：接口整体不可达（三次全部读超时）时，用户看到的不是降级提示，而是崩栈。**
这就是今天第一个要修的 bug，先写一个“假 client 每次都抛 `TimeoutError`”的测试把它钉住。
（已修复，见第二节。）

顺带发现的第二个问题：`agent.py` 的错误返回写死了 `"steps": 0`。
如果是在第 3 步失败，已执行的步数、已花的 token 上下文都会丢失，
和「失败时要保留部分结果」的原则冲突。

### 1.4 两个自检题

#### 为什么 429 的重试要加 jitter？

不加抖动的指数退避是确定性的，所有失败的客户端会在同一时刻重试：

```
t=0.0    1000 个请求全部 429
t=0.5    1000 个请求同时重试 → 服务端只能接 100 个，900 个再 429
t=1.5    900 个同时重试 → 又超载
```

重试本身没错，但同步重试把一次限流放大成连续的脉冲，这就是惊群效应（thundering herd）。

jitter 把“同一时刻”摊成一个区间，例如 `0.5 ~ 0.75s`，让到达时间打散，瞬时并发降到服务端能承受的水平。

要点：**jitter 对单个客户端几乎没有收益，它的价值在所有客户端共享同一个服务端时。**
我们无法控制其他客户端，只能保证自己不跟着一起撞。429 响应带 `Retry-After` 时，应先遵守它，再叠加抖动。

#### connect 超时和 read 超时分别卡在哪一步？

| 类型 | 卡在哪 | 合理值 | 调太短 | 调太长 |
|---|---|---|---|---|
| `connect` | DNS 解析、TCP 握手、TLS 握手 | 3–5s | 网络抖动被误判为失败 | 端点不通时干等 |
| `read` | 连接已建立，等服务端返回数据 | 20–60s | 正常的长回答被误杀 | 服务端卡住时迟迟不报错 |
| `write` | 发送请求体 | 5–10s | 大请求被截断 | 影响很小 |
| `pool` | 等连接池里的空闲连接 | 5–10s | 高并发时误超时 | 排队时间不可见 |

不该用同一个数值的原因：connect 失败几秒内就能确定，继续等没有意义；
而模型生成本身就要十几秒甚至更久，用 5 秒的读超时会把健康请求全部杀掉。

用标量 `httpx.Timeout(30)` 的代价可以算出来：端点完全不通时，
3 次尝试 × 30s + 退避 ≈ **90 秒**才失败；分成 connect=5s 后约 **16 秒**。

还有一个容易忽略的细节：`httpx` 的 `read` 超时是**按单次读操作**计算的，
不是整段响应的总时长。流式响应只要数据块持续到达，整体可以超过阈值；
而 `asyncio.wait_for(timeout=...)` 管的是整次调用的总时长。两层要分开设计。

### 1.5 本段结论

1. 原有重试覆盖 429 / 5xx / HTTPX 网络异常，但没有抖动，
   且连接超时与读超时共用一个标量值。
2. **真正的 bug 是异常类型没对齐**：`asyncio.wait_for` 抛内置 `TimeoutError`，
   `agent.py` 只捕获 `httpx.HTTPError`，导致超时路径崩栈。
3. 降级返回里丢掉了真实步数，`steps` 恒为 0。

以上三条均已修复，见第二、三节。

---

## 二、重试加固

### 2.1 测试先行

先写两条失败用例，确认 bug 存在：

```python
class AlwaysTimeoutClient:
    """每次调用都超时，用来模拟接口整体不可达。"""

    def __init__(self) -> None:
        self.calls = 0

    async def post(self, *args: object, **kwargs: object) -> None:
        self.calls += 1
        raise TimeoutError("simulated timeout")
```

`test_retry_wraps_timeout_into_llm_call_error` 断言重试耗尽后抛 `LLMCallError`；
`test_arun_agent_degrades_on_timeout` 断言 Agent 返回结构化错误而不是崩栈。

修复前的失败信息就是最好的证据：

```
llm_retry attempt=1/3 wait=0.5s error=TimeoutError
llm_retry attempt=2/3 wait=1.0s error=TimeoutError
llm_retry_exhausted attempts=3 error=simulated timeout
FAILED ... TimeoutError: simulated timeout
```

### 2.2 统一异常，保留因果链

```python
message = f"重试 {max_attempts} 次仍失败：{type(last_error).__name__}"
raise LLMCallError(message) from last_error
```

`from last_error` 把原始异常挂到 `__cause__` 上，traceback 会显示
“The above exception was the direct cause of the following exception”，
排查时能同时看到“最终失败原因”和“最底层根因”。

**4xx 不做包装。** 400/401/403/422 在 `except httpx.HTTPStatusError` 里直接 `raise`，
不重试、不转换——它表示请求本身错了，调用方需要拿到原始状态码。

两条契约由此分开：

| 场景 | 抛什么 |
|---|---|
| 4xx（非 429） | `httpx.HTTPStatusError`，只调用 1 次 |
| 429 / 5xx / 网络异常，重试耗尽 | `LLMCallError`，调用 `max_attempts` 次 |

### 2.3 加 jitter

```python
wait = base_backoff * 2 ** (attempt - 1)
wait += random.uniform(0, wait * 0.5)
```

保证至少等完基础退避，再追加 0~50% 的随机量：

| 第几次重试 | 基础 wait | 实际等待 |
|---|---|---|
| 1 | 1.0s | 1.0 ~ 1.5s |
| 2 | 2.0s | 2.0 ~ 3.0s |
| 3 | 4.0s | 4.0 ~ 6.0s |

抖动幅度按当前基础值等比缩放，所以重试越往后，客户端之间的到达时间越分散。

测试用注入代替“跑得快”：

```python
monkeypatch.setattr("agent_playground.llm_client.asyncio.sleep", fake_sleep)
monkeypatch.setattr("agent_playground.llm_client.random.uniform", lambda a, b: b / 2)
...
assert waits == [1.25, 2.5]
```

### 2.4 超时分层

```python
def build_timeout(settings: Settings) -> httpx.Timeout:
    return httpx.Timeout(
        connect=settings.connect_timeout,
        read=settings.read_timeout,
        write=settings.connect_timeout,
        pool=settings.connect_timeout,
    )
```

| 类型 | 默认值 | 卡在哪 |
|---|---|---|
| `connect` | 5s | DNS / TCP / TLS |
| `read` | 60s | 等待模型生成 |
| `write` | 5s | 发送请求体 |
| `pool` | 5s | 等空闲连接 |

`asyncio.wait_for(timeout=settings.request_timeout)` 保留为**整次调用**的上限。
两层各管一件事：`httpx.Timeout` 管每一步，`wait_for` 管整体。
整体上限要大于单次 read，否则 read 超时永远轮不到触发。

---

## 三、降级契约

失败时返回的 dict 是一份承诺：

| 字段 | 含义 | 规则 |
|---|---|---|
| `answer` | 给用户看的话 | 人话，不出现类名、堆栈、key |
| `error` | 给程序看的错误码 | 异常类名，如 `LLMCallError` |
| `steps` | 已完成的步数 | 真实值，不是 0 |
| `prompt_tokens` / `completion_tokens` | 已花的钱 | 失败也要能算账 |

### 3.1 保留真实步数

```python
steps_done = 0
try:
    for step in range(1, MAX_STEPS + 1):
        ...
        steps_done = step    # 这一轮完整跑完之后
```

`steps_done = step` 必须写在工具结果全部回填之后：
第 2 步在调模型时就失败了，`steps_done` 还是 1，表示“完成了 1 步”；
写在循环开头就会把失败的那一步也算成完成，属于虚报。

错误返回里用 `"steps": steps_done`，`MaxStepsReached` 的返回仍然用 `MAX_STEPS`——
前者是“中途失败”，后者是“正常跑满但没答出来”，两者不能混。

### 3.2 用户文案与机器字段分离

```python
"answer": "（服务暂时不可用，任务未能完成）",
"error": type(exc).__name__,
```

用户不需要看到 `LLMCallError`，程序需要它来做告警和统计。
**同一个失败，两种读者，两种表达。**

### 3.3 同步 CLI 也兜底

`run_agent` 没有重试，也没有捕获异常。`main()` 里只包调用本身，不包配置加载：

```python
try:
    run_agent(question, settings)
except (httpx.HTTPError, TimeoutError, LLMCallError) as exc:
    print(f"\n[错误] 服务暂时不可用（{type(exc).__name__}），请稍后重试。")
    return 1
```

配置错误（缺 key、字段非法）不在这里兜——那类问题重试一百次也不会好，
应该在启动阶段就明确报出来。**配置错误要响，网络错误要柔。**

### 3.4 测试

```python
client = FakeAsyncClient(
    [tool_call_response("calculator", '{"expression": "1+1"}'), 500, 500, 500]
)
result = await arun_agent("2+2 等于几", Settings(), client=client)

assert result["error"] == "LLMCallError"
assert result["steps"] == 1                 # 第 1 步完成，第 2 步失败
assert result["prompt_tokens"] == 200       # 已花的钱保留
assert "LLMCallError" not in result["answer"]
```

---

## 四、成本

### 4.1 token 预算守卫

```python
def estimate_tokens(text: str) -> int:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk // 2 + other // 4
```

粗估即可：它的作用是“有一个上限”，不是“算得一毫不差”。

守卫必须在**第一次调用模型之前**执行：

```python
try:
    estimated = sum(estimate_tokens(m.get("content") or "") for m in messages)
    if estimated > settings.max_prompt_tokens:
        return {... "error": "PromptTooLong", "prompt_tokens": 0 ...}

    for step in range(1, MAX_STEPS + 1):
        ...
```

发出去再检查等于没守：钱已经花了。

测试用“空脚本的假 client”验证一次请求都没发生：

```python
client = FakeAsyncClient([])          # 一旦被调用就 AssertionError
result = await arun_agent("长" * 20_000, Settings(max_prompt_tokens=100), client=client)
assert result["error"] == "PromptTooLong"
assert client.calls == []
```

### 4.2 模型路由

```python
COMPLEX_HINTS = ("分析", "对比", "设计", "方案", "为什么", "步骤", "推导", "权衡")

def pick_model(question: str, *, default: str = CHEAP) -> str:
    if len(question) > 80 or any(h in question for h in COMPLEX_HINTS):
        return STRONG
    return default
```

纯函数、可单测。它是启发式，不是分类器。

### 4.3 实测：6 个任务 × 2 个模型

| 策略 | 成功率 | 总成本 | 平均延迟 |
|---|---|---|---|
| 全 flash | **6/6** | **¥0.015875** | **4.74s** |
| 全 pro | 6/6 | ¥0.051697 | 8.80s |
| 路由 | 6/6 | ¥0.050752 | 8.61s |
| 事后最优 | 6/6 | ¥0.015875 | 4.74s |

路由与人工标注一致 6/6——但**成功率没有提升，成本是全 flash 的 3.2 倍**。

### 4.4 结论：路由的前提是“便宜模型会失败”

这次评测里 flash 六题全对，包括三道标成 complex 的题。因此：

1. **“复杂问题”不等于“需要更强模型”**——复杂度是任务属性，需不需要 pro 是实测属性；
2. 在无失败样本时，路由只会增加成本；
3. “事后最优”等于全 flash，说明基线已经是最优解。

更好的方向是**级联**而不是预先路由：

```
先用 flash 回答
  → 答案通过校验（关键字 / 工具调用成功 / 格式合法）→ 直接用
  → 校验失败 → 用 pro 重答
```

级联用**结果信号**决定升级，成本上界等于全 flash，只有真正失败的题才付 pro 的钱。

---

## 五、踩坑记录

**守卫写在调用之后。** 代码里有 `PromptTooLong` 分支，但位置在 `achat_with_retry` 之后，
假 client 仍然收到了第 1 次请求。测试用“空脚本 + 一次都不能调用”把位置错误逼了出来。

**测试函数放进生产模块。** `pick_model` 的两条测试一度被写进 `model_router.py`，
生产模块里出现 `assert` 和 `test_` 前缀函数，运行时不会被收集，还会被当成业务代码发布。

**异常类型不对齐。** `asyncio.wait_for` 抛内置 `TimeoutError`，
它不是 `httpx.HTTPError` 的子类，原来的 `except` 根本接不住。

**autouse fixture 放错文件。** 放在 `test_agent.py` 里的 autouse fixture 只对该模块生效；
`test_llm_client.py` 仍在用真实 key，pytest 打印局部变量时把 `sk-...` 泄露到了日志里。
共享 fixture 必须放 `tests/conftest.py`。

**测试断言写反。** 4xx 路径仍抛 `httpx.HTTPStatusError`，重试耗尽才抛 `LLMCallError`；
全局替换异常类型导致两条测试的断言互换，两边同时红。

---

## 六、已知限制与下一步

- 没有熔断器：连续失败时仍会按完整重试次数打满；
- 预算守卫是字符粗估，不是 tokenizer 精确计数；
- 路由规则是关键词启发式，实测在 6 题样本上无收益，默认仍是 `deepseek-flash`；
- “成功率”是关键词子串匹配，只能区分“答偏/没答”，不能衡量质量；
- 每日额度与未来的熔断状态都在进程内存，重启清零。

下一步优先级：

1. **级联降级**：flash 校验失败再升级 pro，替代预先路由；
2. **最小熔断**：连续 N 次失败后 30s 内直接降级；
3. **真实 tokenizer**：用服务端返回的 usage 校准粗估系数。

---

## 七、面试三分钟版本

> 我给 Agent 做了可靠性加固：连接超时与读超时分开配置，指数退避加 0~50% 抖动，
> 重试耗尽统一包装成自定义 `LLMCallError` 并用 `from` 保留因果链；
> Agent 层捕获后返回结构化结果，保留真实步数和已消耗 token，用户文案不暴露类名。
>
> 最大的坑是异常类型：`asyncio.wait_for` 抛的是内置 `TimeoutError`，不是 `httpx.HTTPError`，
> 原来的 `except` 接不住，线上表现就是崩栈。这个问题靠读代码很容易漏，是故障注入测试抓出来的。
>
> 成本上我加了请求前的 token 预算守卫和模型路由。但实测 6 个任务 flash 全对，
> 路由成本是全 flash 的 3.2 倍、成功率没有提升——所以结论是“复杂不等于需要更强模型”。
> 我把方向改成了级联：先 flash，结果校验不过再升级，成本上界等于全 flash。

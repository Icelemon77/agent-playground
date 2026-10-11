# agent-playground

AI Agent 工程化练习仓库。当前完成到 Day 2：一条从零搭建的大模型调用链路——
配置管理、结构化日志、异常体系，以及一个支持**流式输出、多轮对话、成本统计**的命令行聊天工具。

## 环境要求

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)（包与环境管理器）

## 快速开始

```bash
git clone <你的仓库地址>
cd agent-playground
cp .env.example .env        # Windows: copy .env.example .env
uv sync
```

### 怎么配 API Key

编辑项目根目录的 `.env`，至少填入 `LLM_API_KEY`：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ENV` | `dev` | 运行环境，只接受 dev / staging / prod |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `LLM_API_KEY` | 空 | **必填**，从服务商开放平台获取 |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | 接口地址，需兼容 OpenAI 协议 |
| `LLM_MODEL` | `deepseek-flash` | 模型名 |
| `REQUEST_TIMEOUT` | `30` | 请求超时秒数，范围 1–300 |

两点约定：

- **`.env` 永不提交**（已在 `.gitignore` 中）。`.env.example` 只放键名和默认值，不放真实密钥。
- 配置的取值优先级是 **系统环境变量 > `.env` 文件 > 代码里的默认值**，
  由 `src/agent_playground/config.py` 的 `Settings` 统一读取并校验，配置写错会在启动阶段直接报错。

换服务商只需改 `.env` 里的两行（`LLM_BASE_URL` 和 `LLM_MODEL`），代码不用动。

### 怎么运行 CLI

```bash
uv run python -m agent_playground.cli
```

启动后直接输入问题，回答会**逐字流式输出**：

```
输入问题开始对话。命令：/exit 退出  /clear 清空  /stats 统计  /think 思考显示

你 > 我叫lemon，是一名大三学生
AI > 你好，lemon！很高兴认识你，大三同学。有什么我可以帮你的吗？
[本轮 54+101 tokens / 1.18s / finish=stop]
```

**内置命令**

| 命令 | 作用 |
|---|---|
| `/exit` | 退出并打印本次会话统计 |
| `/clear` | 清空对话历史（保留系统提示） |
| `/stats` | 查看累计 token 与预估花费 |
| `/think` | 切换是否显示模型的思考过程（推理模型才有） |

`Ctrl+C` 也可以退出，同样会打印统计。**注意命令必须带斜杠**——输 `exit` 会被当成正常提问发给模型。

退出时输出：

```
本次会话：输入 252 tokens，输出 233 tokens
预估花费：¥0.001279
```

## 目录结构

```
src/agent_playground/
├── config.py          配置加载与校验（Pydantic Settings）
├── errors.py          异常体系（LLMCallError 携带 status_code）
├── logging_config.py  控制台 + JSONL 双格式日志
├── cost.py            token 成本估算（含高峰时段判断）
├── text_utils.py      文本处理小工具
├── prompts.py         系统提示词（集中管理，避免多个入口不一致）
├── context.py         上下文压缩（按内容类型选择策略）
├── tools.py           工具注册表：@tool 装饰器 + 同步/异步分发
├── llm_client.py      LLM 客户端：同步流式 + 异步 achat + 重试
├── agent.py           Agent 主循环：同步流式 + 异步并发（arun_agent）
├── react_agent.py     文本式 ReAct Agent（另一条实现路线）
├── trace.py           每次运行的独立记录
├── replay.py          trace 回放
├── batch.py           批量并发任务
├── cli.py             交互式命令行对话
└── main.py            启动链路自检入口
```

## 开发

```bash
uv run ruff format .    # 格式化
uv run ruff check .     # 静态检查
uv run pyright          # 类型检查
```

## 实现说明

**流式输出**：通过 HTTP SSE（Server-Sent Events）接收分片，逐片解析 `data: {...}` 行，
遇到 `data: [DONE]` 结束。相比非流式，首字延迟从等待整段生成缩短到几百毫秒。

**成本统计**：从响应的 `usage` 字段取 token 用量，按服务商定价换算
（价格与来源见 `cost.py` 顶部注释）。流式调用需要显式传
`stream_options: {"include_usage": true}` 才会返回用量。

**上下文管理**：模型本身没有记忆，每轮都要把完整历史重新发过去。
`cli.py` 只保留最近 `MAX_TURNS` 轮，避免 token 成本随轮数无限制增长
（实测：第 1 轮输入 54 tokens，20 轮后涨到 1285 tokens）。

**工具调用**：模型不执行任何函数，它只输出「调用哪个工具、传什么参数」，
真正执行的是本项目的代码。完整链路是「模型请求调用 → 代码执行 → 结果回传 → 模型生成答案」，
最多循环 `MAX_STEPS` 步，因此支持连续调用多个工具。
工具执行失败时，错误会被转换成字符串回传给模型（而不是抛异常让程序崩溃），
由模型决定重试或向用户说明。

## 支持的工具

模型可以调用以下工具（定义见 `src/agent_playground/tools.py`）：

| 工具 | 参数 | 说明 |
|---|---|---|
| `calculator` | `expression` | 计算数学表达式 |
| `get_current_time` | `timezone` | 获取指定时区的当前时间 |
| `read_file` | `filename` | 读取 `data/` 目录下的文本文件（限制在该目录内） |
| `list_files` | 无 | 列出 `data/` 目录下的文件名 |

### 运行 Agent

```bash
uv run python -m agent_playground.agent "读取 data/numbers.txt，然后把里面所有数字相加"
```

Agent 会在需要时自动调用工具，并且可以**连续调用多个工具**完成任务。
工具执行失败时，程序不会崩溃——错误会被回传给模型，由模型决定重试或向用户说明。

## 两种 Agent 实现

项目里有两条 Agent 实现路线，用于对比学习：

| 入口 | 范式 | 工具调用方式 |
|---|---|---|
| `agent.py` | 原生 Function Calling | 通过 `tools` 参数，走 API 的 `tool_calls` 协议 |
| `react_agent.py` | 文本式 ReAct | 靠 prompt 约定格式，自己用正则解析 |

```bash
# 同一个问题，两种实现各跑一遍
uv run python -m agent_playground.agent "读取 data/app.log，总结里面出现了哪些问题类型"
uv run python -m agent_playground.react_agent "读取 data/app.log，总结里面出现了哪些问题类型"
```

运行后会生成 `logs/trace_*.jsonl`，可以回放完整的推理过程：

```bash
uv run python -m agent_playground.replay logs\trace_20261010_153000.jsonl
```

### 实测对比（deepseek-flash，关闭思考模式）

| 问题 | 实现 | 步数 | 输入 token | 输出 token | 耗时 | 成本 |
|---|---|---|---|---|---|---|
| 幂运算 | function calling | 2 | 1440 | 56 | 1.48s | ¥0.001797 |
| 幂运算 | ReAct | 2 | 634 | 70 | 1.39s | ¥0.000987 |
| 日志总结 | function calling | 2 | 2064 | 436 | 2.66s | ¥0.004113 |
| 日志总结 | ReAct | 2 | 1254 | 375 | 2.86s | ¥0.002974 |

**怎么选：**

- 需要**并行调用多个工具** → function calling（ReAct 的文本格式一次只能输出一组 Action）
- 需要看到模型的**推理过程**、需要人工审核断点 → ReAct
- **工具很多** → function calling（结构化协议更可靠，不会因格式漂移失败）

详细分析和踩坑记录见 [`docs/agent_notes.md`](docs/agent_notes.md)。

## 异步与并发

批量任务支持并发执行。实测（5 个 Agent 任务，每个含工具调用）：

| 并发上限 | 串行 | 并发 | 加速 |
|---|---|---|---|
| 3 | 7.87s | 3.52s | **2.2x** |
| 5 | 8.25s | **2.02s** | **4.1x** |

```bash
uv run python -m agent_playground.batch 5
```

**token 消耗不变**（差异 < 1%）——并发只压缩时间，不减少工作量。

设计要点：

- **工具按同步/异步分流**：`async def` 工具直接 `await`，同步工具用 `asyncio.to_thread` 包装，
  避免阻塞事件循环（漏掉包装会让并发静默失效，实测 1.00s 变 3.00s）
- **客户端全程复用**：`httpx.AsyncClient` 初始化要同步加载 CA 证书，每次约 135ms，
  6 个任务各自创建会白付 900ms
- **单个任务失败被隔离**：失败返回带 `error` 字段的结果，不影响批次其他任务
- **可恢复的错误自动重试**：指数退避 0.5s → 1s → 2s，4xx 立刻放弃
- **`Semaphore` 限制并发数**，`asyncio.wait_for` 控制单请求总时限

详细分析、六组实测数据和踩坑记录见 [`docs/async_notes.md`](docs/async_notes.md)。

## 已知限制

- 历史截断策略是「丢弃最早的」，没有做摘要压缩
- 重试只覆盖网络层异常和 429/5xx，没有熔断与降级策略
- 并发上限需要手动指定，没有根据服务端限流自适应调整
- 成本估算基于 2026-10-08 抓取的价格，且未处理中国法定节假日
- 没有单元测试

## 下一步

- Day 6：测试与 trace 回放

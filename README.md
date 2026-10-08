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
├── logging_config.py  控制台 + JSONL 双格式日志
├── errors.py          异常体系（LLMCallError 携带 status_code）
├── llm_client.py      LLM 客户端：流式调用，返回回答/思考/用量/耗时
├── cost.py            token 成本估算（含高峰时段判断）
├── text_utils.py      文本处理小工具
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

## 已知限制

- 历史截断策略是「丢弃最早的」，没有做摘要压缩
- 只有串行单轮对话，没有并发请求
- 没有重试与限流处理（`errors.py` 已预留 `status_code` 供上层判断）
- 成本估算基于 2026-10-08 抓取的价格，且未处理中国法定节假日
- 没有单元测试

## 下一步

- Day 3：Function Calling / Tool Use

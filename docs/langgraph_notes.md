# LangGraph 状态机笔记

> 记录日期：2026-10-19
> 对应实验：`scratch/lg_min.py`（最小图）、`scratch/lg_agent.py`（图版 Agent）、
> `scratch/lg_hitl.py`（人工确认）
> 环境：langgraph 1.2.16 / Python 3.12.6

---

## 一、一句话结论

**LangGraph 不替我想逻辑。它替我做两件事：把控制流画成图、把每一步的状态存下来。**

我手写的 `for step in range(MAX_STEPS)` 加 `if calls:` 本来就能表达同样的流程。
框架带来的增量是可恢复、可组合、可视化，代价是一层抽象和一堆依赖。

所以它没有进主项目依赖：实验全部放在 `scratch/`（gitignore），用
`uv run --with langgraph` 临时装。跑完 `git diff pyproject.toml` 为空，
README 里那句「不依赖 LangChain」依然成立。

---

## 二、概念对照：框架 vs 我手写的实现

| LangGraph 概念 | 项目里对应的东西 | 差别 |
|---|---|---|
| `State`（TypedDict） | `messages: list[dict]` | 框架用 **reducer** 决定新值怎么合并进旧状态 |
| **node** | 主循环里「调一次模型」/「执行一轮工具」 | 变成图上独立的点，能单独测、单独重跑 |
| **edge** | `for` 循环的下一轮 | 控制流从代码结构变成图结构 |
| **conditional edge** | `if calls: ... else: return` | 同上，但能画出来、能交给别人看 |
| `compile()` | `run_agent()` 这个函数 | 编译后得到可调用对象 |
| **checkpointer** | **我没有** | 每步自动存档，崩了能续 |
| `interrupt()` | **我没有** | 在节点里暂停，等人确认再继续 |

最值得记的一条：**`tools → agent` 那条回边，就是我手写的 `for` 循环。**
图里没有「循环语句」，循环是边的拓扑形状。

---

## 三、reducer：state 的合并语义（实验）

`State` 的字段可以声明「怎么合并」。同一个 `return {"log": [...]}`，
声明不同，结果完全不同。

```python
class State(TypedDict):
    count: int
    log: Annotated[list[str], add]   # add 来自 operator
```

节点每次返回 `{"count": n, "log": [f"第 {n} 步"]}`，跑 `count=0` 的输入，实测：

| `log` 的声明 | 节点返回什么 | 最终 log |
|---|---|---|
| `list[str]` | 单条 | `['第 3 步']`（只剩最后一条） |
| `Annotated[list[str], add]` | 单条 | `['第 1 步', '第 2 步', '第 3 步']` |
| `Annotated[list[str], add]` | **整个列表** | `['第 1 步', '第 1 步', '第 2 步', '第 1 步', '第 1 步', '第 2 步', '第 3 步']` |

第三行是关键：reducer 是「追加」，节点返回整个列表就会**重复追加**，越跑越长。

两个推论：

1. 挂了 reducer 的字段，节点只返回**增量**；没挂的字段，节点必须返回**完整的新值**。
2. reducer 只管合并，**不影响跑几圈**。上面三次实验的 `count` 都是 3，执行步数一次没变。

对照手写代码：`messages.append(...)` 就是我在手工做 reducer 的事。
搬到图上，`messages: Annotated[list[dict], add]` 之后，节点里只 `return {"messages": [新消息]}`。

---

## 四、最小图：自环 + 条件边

```
START → step ─(count < 3)→ 回到 step
             └─(否则)──────→ END
```

```python
graph = StateGraph(State)
graph.add_node("step", step)
graph.add_edge(START, "step")
graph.add_conditional_edges("step", should_continue, {"again": "step", "done": END})
app = graph.compile()
```

`should_continue` 返回的是**字典里的 key**（字符串），不是节点对象。
返回值对不上 key，LangGraph 直接报错。

### 实测运行轨迹

```
count=0 → [node] in=0 → out=1 → [router] count=1 → again
          [node] in=1 → out=2 → [router] count=2 → again
          [node] in=2 → out=3 → [router] count=3 → done
          {'count': 3, 'log': ['第 1 步', '第 2 步', '第 3 步']}

count=5 → [node] in=5 → out=6 → [router] count=6 → done
          {'count': 6, 'log': ['第 6 步']}
```

两个容易想错的地方：

- **条件边在节点之后才判断**，`START → step` 是固定边，没有前置检查。
  所以输入 5 也要先跑一次，结果是 6 而不是 5；输入给 100 也一样是 101。
- **节点和路由器严格交替**，路由器每次看到的都是节点刚写入的新值。

### 循环失控会怎样

如果路由永远返回 `again`、state 又不收敛，图不会静默卡死，实测抛：

```
GraphRecursionError: Recursion limit of 10007 reached without hitting a stop condition
```

循环能停，靠的是 **state 在变、条件最终为假**，不是框架帮我数圈数。
这也是图版 Agent 必须自己维护 `steps` 上限的原因。

---

## 五、用图重写工具调用循环

目标是把 `agent.py::run_agent` 的循环画出来：

```
START → agent（调模型）→ 有工具调用吗？
                          ├─ 有 → tools（执行）→ 回到 agent
                          └─ 无 → END
```

三条纪律（也是验收标准）：

1. 工具直接复用 `tools.py` 的 `dispatch` / `get_tool_schemas`，不重写；
2. 模型调用直接复用 `llm_client.stream_chat`；
3. 同一批问题分别跑手写版和图版，**结论必须一致**。

### 实测对照（`scratch/lg_agent.py`）

| 问题 | 工具链 | 图版步数 | 手写版步数 | 两版答案 |
|---|---|---|---|---|
| 3 的 15 次方 | `calculator` | 2 | 2 | 一字不差：14,348,907 |
| numbers.txt 求和 | `read_file` → `calculator` | 3 | 3 | 结论一致：87（措辞不同） |

第二题图版的消息序列是
`['system', 'user', 'assistant', 'tool', 'assistant', 'tool', 'assistant']`，
共 7 条、3 次模型调用，说明回边真的走了两趟。

注意：**不要用 `==` 硬比两版字符串**。实测第二题一个写「内容是……求和结果」，
一个写「内容为……求和结果」，结论和工具链都对，纯属模型措辞随机性。
比关键数字和工具链更靠谱。

### 三个必踩的坑

- `messages` 忘了挂 `add`：`tools` 节点返回的结果会覆盖整段历史，第二轮模型不知道
  用户问过什么，答案开始胡说。
- `tool_call_id` 没回填或填错：OpenAI 协议靠它把工具结果对应回某次调用，错了直接 400。
- assistant 消息的 `content` 用 `result["answer"] or None`：有 `tool_calls` 时模型常返回
  空字符串，应为 `null`，某些实现会因 `""` 报错。

### 框架的另一面：并发被固定住了

手写版 `arun_agent` 在同一轮里用 `asyncio.gather` 并发执行多个工具；
LangGraph 的节点**一次一个**跑。想并发，得自己在 `tools` 节点内部用 `gather`。

这是框架的取舍：控制流变得可视化、可恢复，灵活性要自己找回来。

---

## 六、checkpointer + interrupt（人工确认）

手写版 `run_agent` 的硬伤：进程一崩，循环里跑到第 3 步的 `messages` 全丢；
也没法「停在那儿等人回话」。checkpointer 解的就是这个。

```
START → prepare（生成草稿）→ approve（interrupt 暂停）→ send → END
```

| 调用 | 作用 |
|---|---|
| `MemorySaver()` | 内存版存档器 |
| `compile(checkpointer=saver)` | 给图装存档 |
| `{"configurable": {"thread_id": "demo-1"}}` | 会话 id，不同 id 互相看不见 |
| `interrupt({...})` | 节点内暂停，把参数交给外面 |
| `Command(resume=值)` | 用这个值回答 `interrupt()`，从断点继续 |
| `graph.get_state(cfg)` | 读存档：`.values` 是状态，`.next` 是下一个节点 |
| `graph.update_state(cfg, patch)` | 暂停期间改状态，写一张新快照 |

### 实测时间线

| 步骤 | 图跑到哪 | 存档状态 |
|---|---|---|
| `invoke(初始 state, demo-1)` | `prepare` 完成、`approve` 中断 | `next=('approve',)` |
| `get_state(demo-1)` | 停着 | 只读 |
| `update_state(demo-1, {"text": 修订版})` | 仍停在 `approve` | `text` 变了，`next` 不变 |
| `invoke(Command(resume="确认"), demo-1)` | `approve` 重跑 → `send` → `END` | 走完，发出的是修订版 |
| `get_state(demo-2)` | 没跑过 | `{}`、`next=()` |

### 最重要的观察：被打断的节点会重跑

实测输出里 `[prepare]` 只打印一次，`[approve] 进入` 打印了**两次**。

`interrupt()` 的语义是「抛一个暂停信号」，续跑时框架**从那个节点的开头重新执行**，
这时 `interrupt()` 才返回 `Command(resume=...)` 传进来的值。

后果很实在：`interrupt()` 之前不能放有副作用的操作。
把「创建订单」写在 `interrupt()` 上面，用户确认一次，订单就多一张。
安全的写法是：确认之前的操作必须幂等，或者全部挪到 `interrupt()` 之后。

### 另外两个坑

- **不装 checkpointer 也能中断，但没法续。** 实测 `compile()` 不传 saver 时，
  `invoke` 照样返回 `__interrupt__`，可状态没存，再次调用等于从头开始。
- **续跑只能用 `Command`。** 暂停状态下直接传普通 dict 进去，实测会把它当作新输入
  覆盖 state、从 `START` 重跑，又撞到同一个 `interrupt`。

### 必须说清的局限

`MemorySaver` 存在**进程内存**里，服务一重启存档全没——和 API 里那个内存版
额度计数器一模一样。生产要接 `SqliteSaver` / `PostgresSaver`。
「内存 vs 持久化」这个坑，这个项目里已经踩到第二次了。

---

## 七、我为什么没把它接进主项目

| | 手写 | LangGraph |
|---|---|---|
| 依赖 | 0 个额外包 | 38 个（含 langchain-core 等） |
| 控制流 | 代码里 `for` / `if` | 图结构，可视化 |
| 断点续跑 | 要自己写序列化 | checkpointer 开箱即用 |
| 人工确认 | 要自己写状态机 | `interrupt()` 一行 |
| 单轮多工具并发 | `asyncio.gather` | 节点内自己实现 |
| 调试 | 直接断点进函数 | 多一层框架调用栈 |

结论不是「谁更好」，是**取舍**：这个项目的卖点是从零实现、能讲清每一行；
但如果产品要「崩溃恢复」「人工审批」，我会用 LangGraph，而不是硬写一个 checkpointer。

---

## 八、工程细节与踩坑

**临时环境。** 直接 `uv run --with langgraph` 走 PyPI 会卡几分钟（在等网络解析），
换清华镜像后 10 秒装完：

```powershell
uv run --no-sync --with langgraph --default-index "https://pypi.tuna.tsinghua.edu.cn/simple" python scratch/lg_hitl.py
```

`--no-sync` 是必须的：设了 `UV_DEFAULT_INDEX` 之后，uv 会把项目重新解析一遍，
把 `uv.lock` 里所有包的下载地址改写成镜像地址（实测 277 行增删、版本一个没变）。
锁文件里写死国内镜像不该提交。加 `--no-sync` 后实测 `uv.lock` 的 hash 前后一致。

**PyCharm 装不了包。** PyCharm 2024.2 自带的 pip 是 20.3.4，Python 3.12 删了
`distutils`，图形界面装任何包都会 `ModuleNotFoundError: No module named 'distutils'`。
和 langgraph 无关。而且今天本来就不该把它装进项目 venv。

**工具输出的编码。** `data/numbers.txt` 开头有 UTF-8 BOM（`EF BB BF`），
`read_file` 用 `utf-8` 读会把 `\ufeff` 留在字符串开头。模型解析不受影响，
但在 GBK 控制台打印会 `UnicodeEncodeError`。要根治，把 `tools.py` 里
`read_text(encoding="utf-8")` 改成 `utf-8-sig`。

**import 路径（1.2.16 实测）。**

```python
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
```

---

## 九、一句话回答「你用 LangGraph 做过什么」

> 我把项目里手写的工具调用循环用 LangGraph 重画了一遍：`agent` 和 `tools` 两个节点、
> 一条条件边、一条回边，状态里的 `messages` 用 `add` reducer 追加。
> 同一批问题两个版本结论一致，步数也一致。另外用 `MemorySaver` + `interrupt`
> 做了人工确认流程，验证了断点续跑、暂停改状态、thread 隔离。
> 结论是：框架替我做的是控制流可视化和状态存档，不是推理逻辑；
> 所以主项目仍然保持零框架依赖，只在需要崩溃恢复/人工审批时才值得引入。

# SQL 数据分析 Agent 笔记

> 记录日期：2026-10-25
> 对应实现：`src/agent_playground/sql_agent.py`
> 数据与金标准：`scripts/make_sales_db.py`、`scratch/golden.jsonl`
> 演示库：`data/sales.db`（gitignore，用固定种子重建）
> **数据集指纹：`124e31737cfceb3d`**（`scripts/db_fingerprint.py` 计算的内容哈希）

---

## 一、场景与范围

**一句话**：自然语言进，模型生成 SQL，经过安全校验后在只读连接上执行，返回结果。

选这个场景的三个理由：

1. 对应课程里学过的数据库知识，面试讲起来有真实背景；
2. 安全边界清晰——只读、单条 SELECT、表结构固定，容易证明「挡得住」；
3. 评测可以自动判分：答案是数字或集合，不依赖 LLM judge。

**今天不做的事**（写进 README 的「非目标」）：

- 不做图表、不做 Web UI；
- 不做多轮对话；
- 不做自动 schema 发现，schema 直接写进 prompt；
- 不做 SQL 优化建议。

范围小不是缺点——**能演示、能评测、能讲清边界**，比功能多但说不清强。

---

## 二、数据与金标准

### 2.1 三张表

```sql
customers(id, name, city, signup_date)
products(id, name, category, price)
orders(id, customer_id, product_id, quantity, order_date, status)
```

30 个客户、10 个产品、120 条订单；日期覆盖 2026-06 到 2026-10。
`scripts/make_sales_db.py` 用 `random.seed(42)` 生成，**重建结果完全一致**。

实际分布（重建后核对）：

| 指标 | 结果 |
|---|---|
| 行数 | customers 30 / products 10 / orders 120 |
| 状态 | paid 86、refunded 15、cancelled 19（约 72/12/16） |
| 订单月份 | 2026-08 51、2026-09 47、2026-10 22 |
| 注册月份 | 06:10、07:8、08:5、09:5、10:2 |
| 孤儿订单 | 0 |

七条断言检查这些数字。**断言要查数据库，不查内存列表**——内存里的 `orders`
长度正确，不代表 `executemany` 真的写进去了。

### 2.2 金标准只存 SQL，不存数字

```json
{"id": 2, "question": "已支付订单中，销售额最高的产品是哪个？只给出产品名称。",
 "sql": "SELECT p.name FROM orders o JOIN products p ON p.id = o.product_id WHERE o.status = 'paid' GROUP BY p.id, p.name ORDER BY SUM(p.price * o.quantity) DESC LIMIT 1"}
```

期望值在评测时用这条 SQL 现算。存死数字的话，数据一改就要手工同步十个数，
而且一旦手算出错，评测会永远差一两题，还会误导你去调 prompt。

---

## 三、三层架构

### 3.1 生成：`generate_sql` + `extract_sql`

Prompt 的七条规则：

1. 只输出一条 SELECT（或以 WITH 开头）；
2. 只能使用给定表和字段；
3. 禁止 INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/PRAGMA/ATTACH；
4. 只输出一条语句；
5. 问单个对象（哪个/最高/最多）用 LIMIT 1，其余加 LIMIT 200；
6. 日期是 `'YYYY-MM-DD'` 文本，可直接比较；
7. 用户要求修改数据时，只输出 `REJECT: 只支持查询，不支持修改数据`。

`extract_sql` 负责把模型输出还原成一条 SQL：

- 去掉 Markdown 代码围栏与 `sql` 语言标记；
- 用 `\b(select|with)\b` 定位第一个关键字，丢掉前置解释；
- **去掉字符串字面量后再检测汉字**，含中文说明就判定为「不是 SQL」；
- 识别 `REJECT` 前缀，转成明确的拒绝原因。

最后一条是评测逼出来的：模型曾把「删除所有订单」回答成
「只能生成 SELECT 查询，不能生成删除语句。」，抽取器把句中的 `SELECT`
当成 SQL 起点，最终报 `no such column: 查询`——**错误分类完全错了**。

### 3.2 校验：`validate_sql`

| 规则 | 挡住什么 |
|---|---|
| 必须以 SELECT / WITH 开头 | `DELETE`、`UPDATE`、`PRAGMA` |
| 去掉尾部 `;` 后不能再有 `;` | `SELECT 1; DROP TABLE orders` |
| 禁止 `--` 与 `/*` | 注释截断 |
| 关键词黑名单（词边界匹配） | 各种写操作 |
| 长度上限 5000 | 恶意超长输入 |

关键词用 `\b` 词边界匹配，而不是 `"delete" in text`——
后者会把字段名 `deleted_at` 误判成删除操作。

### 3.3 执行：`execute_sql`

```python
uri = f"file:{DB_PATH.as_posix()}?mode=ro"
conn = sqlite3.connect(uri, uri=True)
conn.execute("PRAGMA query_only = ON")
rows = conn.execute(sql).fetchmany(MAX_ROWS)
```

三层保护缺一不可：

- `mode=ro`：文件级只读，连接本身没有写权限；
- `PRAGMA query_only=ON`：连接级只读，双保险；
- `fetchmany(200)`：结果行数上限，防止一条查询把内存撑爆。

实测：绕过校验直接执行 `DELETE FROM orders`，报
`OperationalError: attempt to write a readonly database`。

### 3.4 返回契约

```python
{"question": ..., "sql": ..., "columns": [...], "rows": [...],
 "truncated": ..., "error": None, "detail": None, "elapsed_s": ...}
```

错误码分四类，**分类错了会把排查方向带偏**：

| error | 含义 |
|---|---|
| `LLMCallError` / `HTTPStatusError` | 模型接口不可用 |
| `NoSQL` | 模型没给出可用的 SQL（含 REJECT） |
| `UnsafeSQL` | 安全校验拒绝 |
| `SQLError` | SQL 语法或字段错误 |

---

## 四、端到端评测

### 4.1 冻结数据集上的结果：10/10

数据集指纹 `124e31737cfceb3d`，`scratch/run_sql_eval.py` 一次跑完 10 条：

| # | 问题 | 结果 | 备注 |
|---|---|---|---|
| 1 | 一共有多少客户？ | OK（30） | |
| 2 | 已支付订单中销售额最高的产品？ | OK（机械键盘） | 口径写明 paid |
| 3 | 每个城市各有多少客户？ | OK | |
| 4 | 2026 年 9 月订单数？ | OK（47） | |
| 5 | 取消率最高的产品？ | OK（机械键盘） | 并列规则已写进问题 |
| 6 | 客单价最高的城市？ | OK（广州 277.7143） | 自修复实现后通过 |
| 7 | 哪个品类卖出的数量最多？ | OK（图书） | |
| 8 | 2026 年 9 月新增客户？ | OK（5） | |
| 9 | 下单两次及以上的客户数？ | OK（29） | |
| 10 | 退款订单金额合计？ | OK（5174.0） | |

### 4.2 第 6 题：漏 JOIN 与自修复

第 6 题模型生成：

```sql
SELECT c.city
FROM orders o
JOIN customers c ON o.customer_id = c.id
GROUP BY c.city
ORDER BY SUM(o.quantity * p.price) DESC
```

它用了 `p.price`，却没有 `JOIN products p`。SQLite 报
`no such column: p.price`。归类：**`SQLError` / 生成错（漏 JOIN）**。

这个错误在不同数据集上重复出现，说明是系统性倾向，不是随机手滑。

改进：执行报错时把**失败 SQL + 数据库报错原文**作为对话回传给模型，让它修复一次。
修复后的 SQL 必须重新走校验与执行；只允许修 `SQLError`，
`UnsafeSQL` 是终态，绝不给模型「换个写法再试」的机会。

返回结果里新增 `repairs` 字段：0 表示一次通过，1 表示触发过修复。
单测 `test_ask_repairs_sql_error_once` 用假模型覆盖了完整路径：
第一次生成漏 JOIN → 执行报错 → 第二次补齐 JOIN → 通过。

### 4.3 评测过程中修过的三类设计问题

早期版本上出现过两次「看似模型错、实为评测设计」的问题：

- 第 2 题：「销售额」没写清是否包含退款 → 改为「已支付订单中…」；
- 第 5 题：问题问「哪个」，prompt 却要求 `LIMIT 200` → 改为单个对象用 `LIMIT 1`，
  同时金标准只取名称。
- 第 5 题后来还发现**并列**：机械键盘与 Python 入门取消率都是 0.25，
  `ORDER BY rate DESC LIMIT 1` 没有 tie-break，结果不确定。
  改为「若并列取订单数较多的」，并在金标准里追加 `COUNT(*) DESC, p.id ASC`。

原则：**问题要的、金标准返回的、模型给的，三者形状必须一致**；
否则测的是格式，不是正确性。

可以改「问题表述」，不能悄悄改「正确答案」。两次修改都源于原文没说清，
能说清业务依据；如果是为了让模型通过而改 SQL，那就是在拟合模型。

### 4.4 数据集漂移的教训

中途修改生成脚本后，整份数据发生变化：状态分布、月份分布、各题答案全变，
早期笔记里的数值随之失效。`random.seed(42)` 只保证「同一份代码、同样的调用顺序」
产生同样的数据——代码一改，随机序列整体错位。

现在用内容指纹 `124e31737cfceb3d` 标记数据集，评测前先对指纹。
**任何一条通过率都必须能追溯到具体的数据版本。**

---

## 五、安全证据

### 5.1 五条攻击全部被拒

```text
blocked 写操作   -> 只允许 SELECT / WITH 查询
blocked 更新     -> 只允许 SELECT / WITH 查询
blocked 多语句   -> 不允许多条语句
blocked 注释截断 -> 不允许 SQL 注释
blocked PRAGMA   -> 只允许 SELECT / WITH 查询
```

攻击前后 `orders` 都是 120 行——**不是「没有报错」，而是数据真的没变**。

### 5.2 模型拒绝 ≠ 代码安全

对同一个「删除所有订单」，模型两次表现不同：

- 一次输出人话拒绝，被误当成 SQL，报 `no such column`；
- 一次改写成了 `SELECT * FROM orders`，列了 120 行，什么也没删。

这两次都没有破坏数据，但都说明：**安全不能依赖模型的行为**。
于是加了 `REJECT` 协议让拒绝可识别，同时保留校验层与只读连接作为硬约束。

一句话：模型层负责「说清楚我拒绝了」，代码层负责「即使你绕过也改不了」。

---

## 六、踩坑记录

**PowerShell 吞引号。** `python -c "..."` 里再嵌双引号，5.1 版本会把它们吞掉，
报 `SyntaxError: '(' was never closed`。要么用 here-string，要么写成脚本文件。

**断言写在 `finally`。** 断言查内存列表会通过，数据库却可能是空的；
而且 `finally` 里的断言失败会掩盖真正的异常。断言应放成功路径，且优先查库。

**空结果不是错误。** 日期写错时 SQL 不报错，只返回空列表；评测会把「金标准错」
和「模型答错」混在一起。金标准 10 条必须逐条确认非空。

**展示上限与数据上限是两件事。** `MAX_ROWS=200` 保护内存；CLI 只打印前 10 行
保护日志和眼睛。数据层不能截断到 10，否则评测拿不到完整结果。

**SCHEMA 与真实建表是两份文本。** 手工同步容易漏字段；改表后模型会生成旧字段的 SQL，
报 `no such column`，很容易误判成模型不会写 SQL。

---

## 七、已知限制与下一步

- 10 条评测当前 **10/10**；第 6 题依赖自修复（只重试一次），不再是稳定失败；
- 没有执行超时限制，理论上一条重量级查询可以拖住进程；
- 关键词黑名单基于文本匹配，字段名含 `delete` 之类会误伤；
- 只支持单轮问答，没有追问与上下文；
- 评测使用整行精确比对，列顺序或多余列会被判为失败（当前靠问题写清形状规避）。

下一步优先级：

1. **自修复**：已完成（`SQLError` 回传错误、重建一次，`repairs` 可观测）；
2. 加查询超时（`set_progress_handler` 或线程超时）；
3. 从 `PRAGMA table_info` 自动生成 SCHEMA 文本，消除手工同步；
4. 加 FastAPI `/ask` 接口与演示截图。

---

## 八、面试三分钟版本

> 我做了一个只读的 SQL 数据分析 Agent：自然语言进，模型生成 SQL，经过安全校验后
> 在只读连接上执行，返回结果。安全做了三层：prompt 约束、SQL 校验（只允许单条 SELECT、
> 禁注释、禁写关键字）、执行层双保险（`mode=ro` + `PRAGMA query_only`）。
>
> 我用五条攻击用例验证过：删除、更新、多语句、注释截断、PRAGMA 全部被拒，
> 而且攻击前后订单数都是 120——数据真的没变。
>
> 评测用金标准 SQL 现算答案，不存死数字，数据集用内容指纹标记。
> 当前冻结的数据集上 10 条全部通过。中途第 6 题稳定因为漏写 JOIN 报
> `no such column`，我加了“执行错误回传模型、自修复一次”的机制，
> 修复后的 SQL 重新走校验与执行；安全拒绝则绝不自修复。
> 我按「生成错 / 校验误伤 / 执行错 / 口径歧义 / 并列」归因，
> 先把评测设计的问题修掉，再谈模型能力。
>
> 这件事让我确认两件事：评测要保证问题、金标准、输出三者形状一致；
> 数据集必须可复现，否则每次数字都不可比。

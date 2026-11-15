# 架构与代码阅读顺序

```text
控制台：run_p1 → receive → decide → execute → 回读 → 用户
网页：run_web → WebSession → 同一个 Runtime → 消息/事件展示
基准：run_benchmark → 官方 Orchestrator → BenchmarkAgentAdapter → Runtime.decide
                                              ↓
                          Orchestrator 唯一执行官方工具 → 结果交回 Runtime
```

模型基于可见消息生成文本或原生 ToolCall。程序决定提案是否允许执行。ToolMessage 必须对应原 ToolCall.id。被拒提案保存到内部模型历史和事件，不伪装成官方已执行结果。

## 写入

提案 → 参数/业务检查 → 精确参数和数据库状态摘要 → 用户确认 → 执行前复查 → 消耗一次性确认 → 实际写入 → 查询更新后的实体 → 对比结果。

模型文字中的“确认”不会自动产生程序批准。过期、改变参数和重复写入不能复用旧确认；结果不确定时停机保存证据。

## 恢复

- `repair`：有足够信息修正参数或格式，带反馈继续。
- `continue`：已有另一份未尝试的真实身份，继续查询。
- `ask_user`：缺信息或没有匹配身份，正常补问，不计为模型修正失败。
- `stop`：写入结果不确定或回读不一致，停止。

身份状态分为 missing、ready、needs_user_input、authenticated。失败输入按用户轮次与规范化参数记录；模型仅改大小写不能绕过记录，用户重新明确提供同一邮箱则可重试。模型优先表达补问，必要时程序提供有来源标记的补问。

## 状态与日志

|文件|用途|
|---|---|
|manifest.json|版本、模型配置、预算、上游及初始数据库依据；不含密钥值|
|provider/*.request.json / *.response.json|原始模型协议，可能包含用户信息|
|messages.json|实际官方通信轨迹快照|
|model_history.json|下一次模型请求的历史，包括内部拒绝反馈|
|events.jsonl|追加事件与恢复动作|
|summary.json|状态、计数、身份摘要与费用未知标记|
|final_db.json|结束时的数据库快照，仅本地复盘|
|state/retail_extended.sqlite3|扩展业务及写回执|

这些运行文件不随公共仓库发布。下一次模型请求的上下文由本地历史重新组织，不依赖供应商替项目记住会话。

## 评测边界

完整 Task 评测依据只由 runner 和离线 Evaluator 使用；官方用户模拟器只收到用户场景。客服 Agent 只收到 policy、用户消息和可见工具观察，不接收 evaluation_criteria 或答案动作。

正常停止、回读一致、用户满意和官方 reward 是不同指标。官方 Baseline 保留原 LLMAgent 决策实现；三种自研方法共享提示词、工具和预算并切换增强规则/恢复。官方 Baseline 与 Runtime 的总差异不能全部归因于单一模块。

## 阅读顺序

先读 run_p1.py 外层循环，再追 runtime.py 的 receive、decide、execute、checkpoint。接着读 model.py、validation.py 和 recovery.py；最后看 benchmark.py、run_benchmark.py、web.py。先理解消息怎样流动，再研究框架抽象。

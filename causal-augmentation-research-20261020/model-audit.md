# 本任务模型与推理级别审计

审计日期：2026-10-20（UTC）。审计对象仅为本研究任务；不追查其他项目或被禁止的会话路径。

结论：**不能证实每次实际运行都是 GPT-6.1-Sol / max。已确认两位研究者与一位独立审查者的创建请求为这一组合；默认配置存在不同值；当前接口没有提供每次运行的实际模型及推理级别回执。**

| 层次 | 已见证据 | 可以说明什么 | 不能说明什么 |
|---|---|---|---|
| 用户要求 | 19:13:53 UTC 用户要求 `gpt 6.1sol max`，允许多智能体 | 本任务目标设置明确 | 用户要求本身不证明执行设置 |
| 父线程调度信息 | 父线程说明创建／续接请求为 `gpt-6.1-sol`、`thinking=max`；运行中 steering 不改变当前推理 | 父线程报告的请求参数 | 本协调会话每次实际运行元数据 |
| Mac 默认配置 | 只读 `/Users/lkc/.codex/config.toml` 的非秘密白名单键：`model = "gpt-6.1-sol"`，`model_reasoning_effort = "ultra"` | 读取时的默认配置；effort 与用户指定 max 不同 | 不能据此断定本次使用 ultra；显式调用参数可能覆盖默认值；也不能倒推过去运行 |
| 文献审查子智能体 `/root/prior_overlap` | 创建调用显式 `model:"gpt-6.1-sol"`、`reasoning_effort:"max"`、`fork_turns:"none"`；工具返回任务名及 running | 请求设置已提交，任务已创建并完成 | 创建回执未返回 resolved model / effort；自述不是运行证明 |
| 数据审查子智能体 `/root/data_access` | 同样显式请求 `gpt-6.1-sol / max`、独立上下文；完成报告已收到 | 同上 | 同上 |
| 独立审查子智能体 `/root/independent_review` | 创建调用显式 `model:"gpt-6.1-sol"`、`reasoning_effort:"max"`、`fork_turns:"none"`，回执仅任务名；完成审查已收到 | 第三项显式请求已提交 | 未取得实际运行设置；独立上下文不等于跨模型家族审查 |
| 当前协调会话 | 已搜索当前可调用工具元数据；没有发现返回当前实际模型／effort 的专用接口 | 当前核验能力边界 | 无法证实本协调会话在用户追加要求之前或之后每次均为 max |

未发现项目目录或本任务工作区中的其他 `.codex/config.toml` 覆盖文件。没有更改配置、重启任务、重跑文献研究或访问会话存储。

## 被拒绝的只读界面核验

- 动作：CUA `getApp("Codex")`，拟读取当前应用界面的模型信息。
- 目标：`com.openai.codex`。
- 工具原文：`Computer Use is not allowed to use the app 'com.openai.codex' for safety reasons.`
- 状态：**denied**；未取得界面状态，无待执行动作，不是结果不明。
- 处理：停止该界面核验；未换技术或读取被拒绝的会话路径绕过。该拒绝不能证明运行设置符合或不符合要求。

因此，本项目只声明三项子任务“显式请求 max”，不声明“已独立验证全部实际运行均为 max”，也不把默认 ultra 当成已经发生的降级。独立审查也已显式请求同一模型与 effort；请求及冻结输入记录见 `review-inputs.json`。本轮没有可据以重建每次实际推理级别的执行日志。

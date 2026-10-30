# 代码地图与恢复边界

以下路径均相对于 `source/`。

| 用途 | 主要文件 |
|---|---|
| 统一入口、外部模型角色绑定 | `experiments/cli.py`、`workflows.py`、`model_stack.py`、`endpoint_preflight.py` |
| HA online / RQ3 调度 | `harness_env/orgin/HarnessAudit/multi_agent/safepred_experiments/run_matrix.py` |
| IS-Bench online / RQ3 调度 | 同目录 `isbench_online_matrix.py` |
| HA predict-only | 同目录 `predict_only_pipeline.py` |
| IS-Bench predict-only | 同目录 `isbench_predict_only_pipeline.py` |
| 预测聚合与绘图 | 同目录 `analysis.py`；`experiments/plot_results.py` |
| WM、解析、调用与账本 | `SafePred/models/world_model.py`、`models/prompts.py`、`utils/llm_client.py` |
| HA guard、咨询与重生成 | `.../multi_agent/safepred_guard/{guard,router,intervention}.py` |
| HA 隔离预执行 | `.../multi_agent/safepred_guard/environment_h1.py` |
| 反馈变换定义 | `experiments/wm_return_perturbations.py` |
| IS-Bench 六档设计/能力检查 | `experiments/isbench_perturbation_plan.py` |
| IS-Bench 反馈应用 | `IS-Bench/og_ego_prim/models/safepred_safeguard.py` |
| HA 原生分数 | `.../multi_agent/sar_calculate.py`、`completion_judge.py`、`operational_judge.py` |
| IS-Bench 原生评分 | `IS-Bench/og_ego_prim/benchmark/evaluator/`、`benchmark/tracker/` |
| 已有 HA 轨迹独立重评 | `experiments/ha_rejudge_existing.py`、`ha_rejudge_evaluator_only_recovery_safe.py` |
| 正式注册/验收 | `experiments/formal_readiness.py`、`freeze_formal_preregistration.py`、`*_formal_terminal_postflight.py` |

## 断点与失败

- HA：`multi_agent/trace/checkpoint.py`、`runtime_checkpoint.py` 保存带身份和哈希校验的完整 SDK turn；`safepred_experiments/ha_online_resume.py` 管理条件级终态与分母。
- IS-Bench：`og_ego_prim/utils/episode_checkpoint.py`、`action_recovery.py`、`simulation_replay.py` 保存/验证动作级状态。物理重放状态不一致时失败关闭，不能无条件继续。
- `source/scripts/ha_*checkpoint*.py` 是可复用的技术故障 fork 工具，保留来源与完整前缀；名称中的 provider 标识有实际约束，不能当作适用于所有模型的通用恢复授权。
- 条件失败后继续下一条，与“允许重跑已终态失败条件”是两回事。严格恢复不自动重置 turns、语义重生成预算或覆写失败记录。
- actor 已完成而 evaluator 失败时，优先只补 evaluator；恢复的旧账本/guard 产物也必须通过验证，不能因为评分文件出现就计入有效结果。
- 最近完整步骤之后未提交的 HTTP 请求/动作可能重做。代码不保证任意指令位置的 exactly-once 网络重放。

历史 controller、作业号和 provider 临时故障处理脚本未混进统一入口。封存历史 run 的重现仍应使用该 run 自己的冻结代码和注册文件；本包是新的、独立的源代码交付物。

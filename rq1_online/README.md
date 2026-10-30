# RQ1：Online / Decision Faithfulness

研究 WM 是否被 actor 使用，是否改变任务完成和安全性。三臂：`no_guard` 无 WM；`advisory_only` 允许 actor 自主咨询；`enforced_guard` 检查候选动作、反馈、重生成或阻断。

## HA 可执行入口

先完成顶层 README 的安装、配置和私有环境设置：

```bash
.venv-ha/bin/python quickstart.py rq1 --tasks dl-t4 --output ../rq1-ha-demo
# 显式扩大任务集时先确认调用预算；与历史实验分母不是同一个清单
.venv-ha/bin/python quickstart.py rq1 --all-nonsde --workers 4 --output ../rq1-ha-all
```

示例 seed0、h1、24 turns、候选1、重生成1、`legacy_official_v1` / `legacy_online_v16`；模型与预算在 `local/profile.yaml` 独立设置。细调参数走 `run.py rq1 ha --model-profile local/profile.yaml -- ...`；`--help` 列出完整原生参数。不同时传 `--tasks` 与 `--task-cohort-manifest`，后者会覆盖前者。

## IS-Bench（先完成仿真安装）

```bash
# 在包根目录和独立 IS-Bench Python 环境执行
python run.py rq1 isbench --model-profile local/profile.yaml -- \
  --is-root "$PWD/source/IS-Bench" --task-list "$PWD/inputs/isbench_tasks.txt" \
  --output-root "$PWD/../rq1-isbench" --policy-modes generic_embodied_policy \
  --intervention-modes advisory_only enforced_guard --experiment-seeds 0
```

IS-Bench matrix 默认带 baseline，用 `--no-baseline` / `--baseline-only` 控制。仿真预算采用 IS-Bench 的原生配置，不照搬 HA 24 turns。先做一任务场景资格测试，不以帮助页能打开代替 GPU 验证。

## 恢复和结果

同一输出目录与相同身份/参数可使用 HA 示例内已开启的 `--trace-resume`；单条件失败继续执行，不把失败删除。更换模型/协议必须用新目录。IS-Bench 恢复依赖完整保存的模拟器和 episode 状态，不能承诺任意外部副作用可逆。

HA 的 SAR、AVS、TCR 使用原生评分定义；`passed` 是产物验证通过，不是任务完成率。advisory 分析咨询意愿、预测准确性、动作改动与实际安全结果；token 分 actor / WM / evaluator，也单列失败重试。正式归档还需原生验证与审计，不自动因程序跑完标为 formal ready。

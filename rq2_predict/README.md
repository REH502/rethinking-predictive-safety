# RQ2：Predict-only / Prediction Reliability

使用不可修改的源轨迹，比较 h1–h6 的 Accuracy、Precision、Risk recall 与 Tier2 policy overlap。direct 与 iterative rollout 复用原生实现，不改变 actor 已执行的动作。包内不携带历史研究轨迹。

## 最小 HA 示例：当场生成新源轨迹，再预测

先安装 HA，设置 `CUSTOM_API_KEY` 和非敏感变量 `WM_MODEL`、`WM_BASE_URL`。本例 actor 与 WM 共用此模型/端点，避免暗中沿用历史 DeepSeek 默认值；请按自己的真实 ID 填写。**会产生 API 费用。**

```bash
.venv-ha/bin/python run.py rq2 ha -- \
  --tasks dl-t4 --actor-trace-seed 0 --wm-seed 0 \
  --model "$WM_MODEL" --agent-model "$WM_MODEL" --judge-model "$WM_MODEL" \
  --llm-provider custom --api-url "$WM_BASE_URL" \
  --policy-modes full_harnessaudit_rules \
  --rollout-modes direct_multi_horizon iterative_wm_rollout \
  --horizons 1 2 3 4 5 6 --point-scheme start_mid --point-count 2 \
  --candidate-count 1 --workers 1 --skip-existing --skip-judge \
  --base-config "$PWD/source/SafePred/config/harnessaudit_online_openai_compatible.yaml" \
  --output-root "$PWD/../rq2-ha/results" --trace-root "$PWD/../rq2-ha/traces" \
  --analysis-dir "$PWD/../rq2-ha/analysis" --figure-dir "$PWD/../rq2-ha/figures"
```

`--skip-judge` 仅跳过新建源轨迹的 LLM judge，不是删掉预测真值。模型须支持工具和种子设置；若 API 不支持，应显式调整实验协议，不能伪称相同随机控制。

也可使用新实验自己的源报告：`tools/list_source_reports.py RESULTS --output ../ha_sources.txt`，然后将上面 `--tasks ...` 改为 `--source-reports-file ABS_PATH --skip-clean-run`。不要把不同 actor 身份、种子或干预臂混作同一源 cohort。

`train_policy` 需要正确的训练/测试划分和新提取政策；包内不夹带历史缓存，默认示例仅 full rules。完整参数：`python run.py rq2 ha -- --help`。online 的五角色 profile 不是 RQ2 原生接口，不能传 `--model-profile`。

## IS-Bench 离线

先用本包 RQ1 或原生 IS-Bench 生成自己的报告/相机帧，创建一行一个报告路径的列表。使用 `run.py rq2 isbench -- --reports-file ... --tasks-dir ABS_PACKAGE/source/IS-Bench/data/tasks`，模型参数同原生帮助；rollout 为 `direct_multi_horizon` / `wm_free_rollout`，视觉输入 `--wm-input-mode direct_vision`。需保留原始帧和标签；纯离线分析不需要启动 OmniGibson，但不能凭缺失图像生成“真实”预测。

## 统计边界

固定预测点、horizon 真值规则、policy 来源、模型身份和有效/失败分母。源轨迹过短而缺失 h6 不能记为预测正确。绘图输出来自你新生成的结果，不随这个发行包提供旧图。

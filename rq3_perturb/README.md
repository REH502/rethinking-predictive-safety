# RQ3：反馈扰动

复用 online harness，只改变规定的反馈。固定 actor、WM、judge、任务、种子、预算与风险门控，不能按结果挑任务来逼近 baseline。

| 参数值 | 处理 | HA | IS-Bench |
|---|---|---|---|
| `empty` | 清空预测文本，保留 canonical WM 风险门控 | 可用 | 可用 |
| `irrelevant` | 无关反馈，保留门控 | 可用 | 可用 |
| `filler` | 无信息填充，保留门控 | 可用 | 可用 |
| `corrupt` | 反转风险判断与对应反馈 | 可用 | 可用 |
| `corrupt_key_random_chars` | 关键信息字符替换，保留门控 | 可用 | 可用 |
| `environment_h1` | 副本执行候选动作 → 真实结果 → actor 保留/修改 → 主环境执行 | 可用 | **未实现，拒绝启动** |

HA 一任务六条件（先配置外部模型）：

```bash
.venv-ha/bin/python quickstart.py rq3 --tasks dl-t4 --output ../rq3-ha-demo
# 全量静态任务集合：附加 --all-nonsde --workers 4，事先确认费用
```

canonical 文本扰动总会先取得有效的 learned-WM 预测。早期不调用 WM 的 empty/irrelevant 和 checker-oracle 虽保留在兼容代码中，不是同一个协议，不应混合统计。sandbox h1 不将隐藏标签冒充真实前瞻，但额外 actor review 消耗预算，不能宣称与 baseline 的计算成本完全相同。

IS-Bench 安装仿真环境后，仅启动五档：

```bash
python run.py rq3 isbench --model-profile local/profile.yaml -- \
  --is-root "$PWD/source/IS-Bench" --task-list "$PWD/inputs/isbench_tasks.txt" \
  --output-root "$PWD/../rq3-isbench" --policy-modes generic_embodied_policy \
  --intervention-modes distributed_enforced \
  --wm-return-perturbations empty irrelevant filler corrupt corrupt_key_random_chars \
  --experiment-seeds 0 --no-baseline
```

分析应同时报告门控/阻断率、actor 改动、TCR、安全分数、评价覆盖、失败及 token。empty 等保留门控，所以“指标接近 enforced”本身不能证明 WM 没有贡献。使用原生审计验证结果，不因符合预期而放宽有效性要求。

# Predictive Safety Guardrails Do Not Always Make LLM Agents Safer

Research code for studying how predictive world-model guardrails affect agent
safety, prediction reliability, and decision faithfulness in digital and
embodied environments. The accompanying manuscript is under double-blind
review; no author names or publication identifiers are inferred here.

## Reproduction at a glance

- **RQ1:** Compare no prediction, advisory prediction, and enforced guardrails.
- **RQ2:** Evaluate predictions against newly generated or user-provided traces.
- **RQ3:** Test sensitivity to perturbed feedback and sandbox-verified outcomes.
- **HarnessAudit:** Linux x86_64, glibc 2.34 or newer, Python 3.11, and access
  to compatible model endpoints are needed for real experiments.
- **IS-Bench:** Requires a separate environment with OmniGibson 1.1.1, its
  assets, and a compatible NVIDIA GPU. `environment_h1` is not implemented for
  IS-Bench and is rejected by the implementation.
- The package does **not** include the paper's historical trajectories, scores,
  model weights, or credentials. The supplied task list and example configuration
  are not necessarily the exact historical paper cohort or model settings.

On the supported Linux platform, run `bash install.sh`, then configure endpoints
with `.venv-ha/bin/python configure.py`. Read [setup](docs/SETUP.md),
[RQ1](rq1_online/README.md), [RQ2](rq2_predict/README.md), and
[RQ3](rq3_perturb/README.md) before running experiments. Real model calls can
incur charges. The offline checks do not establish paper-result reproducibility
or validate GPU experiments.

## License and attribution

The project's original contributions are licensed under the [MIT License](LICENSE).
The included HarnessAudit and BDDL components retain their respective licenses;
other third-party code and assets may have different terms. The MIT license does
not replace those terms. See [packaging notes](docs/PACKAGING.md).

---

# World Model × Agent Safety：可安装完整实验包

本包包含 RQ1 online、RQ2 predict-only、RQ3 扰动的共享代码，**补齐必要的 HA / IS-Bench 基准输入**、安装入口、模型配置向导和离线自检。不是历史结果备份；不包含我们的实验轨迹、分数、请求账本、断点、模型权重或凭据。

## 先运行 HA（Linux x86_64，glibc ≥ 2.34，Python 3.11）

```bash
# 在解压目录执行；不会安装到系统 Python
bash install.sh

# 填写模型 ID 与地址；不会询问或保存 key
.venv-ha/bin/python configure.py

# 安全地从终端输入凭据，不把 key 放到命令历史里
read -rs -p 'Actor key: ' ACTOR_API_KEY; export ACTOR_API_KEY; echo
read -rs -p 'WM key: ' WM_API_KEY; export WM_API_KEY; echo
read -rs -p 'Evaluator key: ' JUDGE_API_KEY; export JUDGE_API_KEY; echo

# 单个真实任务，三臂；以下命令会调用你配置的模型并产生 API 费用
.venv-ha/bin/python quickstart.py rq1 --output ../my-ha-rq1 --tasks dl-t4

# 单个真实任务，六档扰动
.venv-ha/bin/python quickstart.py rq3 --output ../my-ha-rq3 --tasks dl-t4

# 同一目录、同一配置重执行可恢复；换模型/参数必须换输出目录
# 全量非 SDE：附加 --all-nonsde --workers 4（本源码快照为 200 个任务，先确认预算）
```

只检查、不调用模型：`bash install.sh` 已自动执行输入校验、六个入口检查、原生 HA 三臂/六档 dry-run 与断点/隔离预执行等回归测试。也可单独运行 `.venv-ha/bin/python tools/smoke_test.py`。**自检不证明模型服务可用，不产生科学结果。**

## 内容与结构

| 目录 / 文件 | 内容 |
|---|---|
| `run.py` | RQ1 / RQ2 / RQ3 × HA / IS-Bench 的原生参数入口 |
| `quickstart.py` | HA 可直接执行的单任务/全量示例，不修改 harness 算法 |
| `configure.py` | 为五角色生成无密钥 profile |
| `source/experiments/`, `source/SafePred/` | 通用实验、模型接口、WM、扰动和审计 |
| `source/harness_env/orgin/HarnessAudit/` | HA 实现、210 任务、8 工具目录、初始 fixtures、任务图片 |
| `source/IS-Bench/` | 仿真适配代码、161 任务、BDDL、本体、563 场景初始化模板、字体 |
| `inputs/` | 静态任务列表；没有依据已有实验效果筛选任务 |
| `wheels/`, `requirements-ha.lock.txt` | Linux x86_64 / CPython 3.11 的 HA 离线安装依赖及锁定版本 |
| `tools/doctor.py`, `tools/smoke_test.py` | 无外部 API 的输入与代码自检 |
| `BUNDLE_MANIFEST.sha256` | 逐文件完整性校验，不含任何密钥的哈希 |
| `docs/`, `rq*/README.md` | 人类可读的安装、代码地图与实验说明 |

## 不能省略的运行条件

1. **真实实验必须有可用模型端点和你自己的 key**。不捆绑付费账号，也不能保证第三方 API 永不超时。可用本地 vLLM 或兼容 API；profile 的五角色独立配置。
2. **IS-Bench online 还需要 OmniGibson 1.1.1、其大型对象/场景资产、兼容 NVIDIA RTX GPU 和驱动**；本包的初始化模板不等于完整仿真资产。见 [安装说明](docs/SETUP.md)。没有在普通 CPU 机器上假装完成 GPU 验证。
3. **HA 六档已经实现；IS-Bench 当前只有五档文本扰动可运行**。真正隔离环境预执行 `environment_h1` 在 IS-Bench 中尚未实现，代码会拒绝启动，不冒充“正确反馈”。
4. RQ2 需要先生成新的 online 源轨迹或提供自己的源报告；我们已有的实验轨迹不会随包发布。见 [RQ2](rq2_predict/README.md)。
5. `quickstart.py` 是运行示例，不自动宣称 `formal ready`。正式实验仍需冻结配置、任务集、模型版本和评价口径。失败、跳过不能被隐藏。

因此，本包提供的是**目标环境下可安装、自检、配置后启动的实验实现**，不是在任意操作系统上免安装、免模型、免 GPU 的单文件应用。

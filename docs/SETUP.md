# 安装、运行与依赖边界

## HA：已打包静态输入，无需再找历史工作目录

目标：Linux x86_64、glibc ≥ 2.34（例如 Ubuntu 22.04+）、Python 3.11。`bash install.sh` 使用包内离线 wheels 创建 `.venv-ha` 并运行自检。需要系统 Python 具有 `venv` / `ensurepip`；Ubuntu 可由管理员安装 `python3.11-venv`。不自动 sudo、不修改系统环境。其他平台可自行创建环境并安装 `source/requirements-ha.txt`，但不属于本次已测试的离线 wheel 目标。

包内的 HA 210 YAML、图片、8 个工具目录、初始 JSON fixtures 和 SDE 示例仓库是**静态基准输入**，不是我们的实验输出。默认示例 `dl-t4`；本源码快照共有 200 个非 SDE 任务，使用 `quickstart.py --all-nonsde` 选择，**不是自动复用历史 192 任务队列**。包含 SDE 的 210 任务仍可通过原生 `run.py` 选择，需要对应任务的本地开发工具。历史正式任务集合需单独指定；不要把打包的任务数当成论文每轮分母。

`configure.py` 为 actor / world_model / judge / policy_extractor / action_agent 生成 `local/profile.yaml`。默认 policy_extractor 跟随 WM、action_agent 跟随 actor；如实验要求不同，请显式修改。`--base-url` 是 actor 地址，`--wm-base-url`、`--judge-base-url` 可独立覆盖；模型 ID 与真实服务一致，不做别名改标。

key 由 `ACTOR_API_KEY`、`WM_API_KEY`、`JUDGE_API_KEY` 环境变量读取。也可设置 `WM_SAFETY_ENV_FILE` 指向包外权限 0600 的私有 dotenv 文件。不能把 key 放进命令参数或 profile。无认证本地服务：把相应 endpoint 的 `api_key_env` 删除、设 `auth: none`。

五角色的默认生成预算为 8192 tokens、温度 0、请求超时 360 秒、transport attempts 1；这只是配置起点，不冒充任何既有模型的精确历史预算。真实 revision / server 版本由操作者核实填写。HA 示例固定 seed0、h1、24 turns、候选数1、重生成1、legacy_official_v1 / legacy_online_v16。自检全部不调用模型。

## RQ2：新数据而不是附带旧数据

统一入口：`python run.py rq2 ha -- --help`。它使用原生 `--model` / `--api-url`、`CUSTOM_API_KEY`，不是 online 五角色 profile。源报告可由本包刚运行的 HA 生成，再用 `tools/list_source_reports.py` 收集到包外列表。脚本检查 trace 文件存在，不根据评分选样；仍需人工审计 trace 完整性、任务完成度和配对口径。

训练规则 `train_policy` 的来源/划分要新建并审计；不会随包夹带我们已抽取的 policy cache。先用 `full_harnessaudit_rules` 可避免无训练数据时误配置。

## IS-Bench：独立仿真环境

本包包含 task / BDDL / camera / metadata / scene template 输入和定制 BDDL Python 包；**OmniGibson 的全量模型资产与 Isaac Sim 不适合塞进代码 ZIP，也没有随包复制**。

参考 [IS-Bench 官方安装说明](https://github.com/AI45Lab/IS-Bench) 与 [OmniGibson 文档](https://behavior.stanford.edu/omnigibson/)。原项目使用 OmniGibson **1.1.1**；不要用当前最新版随意替换。

建议先在一张兼容 RTX GPU 上按官方方法验证空场景，使用独立 Python 3.10 环境，不与 HA 3.11 wheel 环境混装：

```bash
conda create -n wm-isbench python=3.10 -y
conda activate wm-isbench
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 'numpy<2' -c pytorch -c nvidia
python -m pip install omnigibson==1.1.1
# 需要联网、足够磁盘及遵守官方许可；此步骤会下载大型资产
python -m omnigibson.install
python -m pip install -r source/requirements-isbench.txt
python -m pip install -e source/IS-Bench/bddl
python tools/doctor.py --isbench
```

以上是上游版本路线，不是跨所有驱动的完整环境锁。也可用上游 `stanfordvl/omnigibson:1.1.1` 容器并安装本包额外依赖；HPC 按集群的容器/GPU 规则运行。先检查 NVIDIA 驱动、RTX/Vulkan 支持、实际显存和场景初始化，不能仅凭 CUDA 可用就判定 A100 能跑 OmniGibson。

IS-Bench 命令示例见 [RQ1](../rq1_online/README.md) 和 [RQ3](../rq3_perturb/README.md)。`--is-root` 指向本包的 `source/IS-Bench`，任务列表直接用 `inputs/isbench_tasks.txt`。运行时指定该独立环境的 Python；没有给 IS-Bench 做真实 GPU 端到端认证，`doctor --isbench` 只检查模块和输入。

## 完整性与可复现性

`python tools/verify_bundle.py` 按逐文件 SHA256 核对源码、公开输入和依赖。安装后的 `.venv-ha`、你自己的 `local/` 不属于发行包校验项。建议实验输出始终在包外。保留精确模型身份、endpoint、参数、任务列表、版本、随机种子、原始账本及原生断点；断点不保证远程副作用或任意指令中断都能无损回滚。

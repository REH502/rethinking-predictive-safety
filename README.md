# Rethinking Predictive Safety

### Predictive Safety Guardrails Do Not Always Make LLM Agents Safer

[![License: MIT (original contributions)](https://img.shields.io/badge/original%20code-MIT-blue.svg)](LICENSE)
[![Research: Empirical Analysis](https://img.shields.io/badge/research-empirical%20analysis-5865F2.svg)](#overview)
[![arXiv](https://img.shields.io/badge/arXiv-2611.02815-b31b1b.svg)](https://arxiv.org/abs/2611.02815)

Weixiang Zhao, Zhihao Liu, Guancheng Li, Yanyan Zhao, Bing Qin, and Ting Liu

Research code for an **empirical analysis of world-model-based predictive safety
for LLM agents**. We examine when predicted consequences improve safety, whether
the predictions are reliable, and whether agents act faithfully on their content.

**Paper:** [arXiv:2611.02815](https://arxiv.org/abs/2611.02815) ·
[PDF](https://arxiv.org/pdf/2611.02815)

[Overview](#overview) · [Quick start](#quick-start) · [Experiments](#experiments) ·
[Reproducibility](#reproducibility-and-release-scope) · [Validation](#validation) ·
[Code map](#repository-layout) · [Citation](#citation)

## Overview

Predicting an action's consequences is one step toward safer execution. This work
studies the full **prediction-to-action pipeline**, separating three questions:

| Question | Experimental comparison | What we measure |
| --- | --- | --- |
| **RQ1: Safety effectiveness** | No prediction, optional prediction, and prediction-constrained execution | Safety, task completion, and inference cost |
| **RQ2: Prediction reliability** | Prediction horizons, policy information, and direct versus iterative prediction | Accuracy, precision, risk recall, and policy overlap |
| **RQ3: Decision faithfulness** | Five feedback perturbations and sandbox-derived feedback | Dependence on prediction content, action changes, and subsequent safety |

The study finds that stronger prediction involvement does not consistently improve
safety, that predictive reliability weakens over longer horizons, and that safety
gains may persist even when predictive content is corrupted. These observations
motivate evaluating predictive safety through both **reliable foresight** and its
**faithful use in decisions**.

The code supports experiments in **HarnessAudit**, covering digital workflows,
and **IS-Bench**, covering embodied household tasks. Guardrail implementations and
benchmark adapters serve as experimental instruments for this analysis.

## Requirements

| Workflow | Environment | Additional requirements |
| --- | --- | --- |
| HarnessAudit experiments | Linux x86_64, glibc ≥ 2.34, Python 3.11 | Compatible model endpoints; credentials when required by those endpoints |
| IS-Bench online experiments | Separate Python 3.10 environment; OmniGibson 1.1.1 | Compatible NVIDIA RTX GPU, drivers, simulator and object/scene assets |
| IS-Bench prediction replay | Python environment with the prediction dependencies | Existing reports, labels, and camera frames; model endpoints |

The bundled offline wheels target **Linux x86_64 / CPython 3.11**. They cannot be
installed directly on macOS or Windows. Local macOS code checks are documented in
[the validation report](docs/VALIDATION.md); they do not establish simulator or
Linux installation compatibility.

The repository is approximately **1.3 GB unpacked**, including benchmark inputs
and offline dependencies. Downloading OmniGibson assets requires additional space.

## Quick start

The commands below use **Bash**, run from the repository root on the supported
Linux platform.

### 1. Clone and install

```bash
git clone https://github.com/REH502/rethinking-predictive-safety.git
cd rethinking-predictive-safety
bash install.sh
```

`install.sh` creates `.venv-ha`, installs the bundled dependencies, verifies the
distribution manifest, checks benchmark inputs, and runs offline smoke tests.
Python 3.11 must be available as `python3.11`, or selected with `PYTHON_BIN`:

```bash
PYTHON_BIN=/path/to/python3.11 bash install.sh
```

For IS-Bench, follow the separate [simulator setup](docs/SETUP.md)
before running online experiments. The full simulator assets are obtained through
the upstream installation process.

### 2. Inspect a run without model calls

```bash
.venv-ha/bin/python quickstart.py rq1 --dry-run --output ../ha-rq1-plan
.venv-ha/bin/python quickstart.py rq3 --dry-run --output ../ha-rq3-plan
```

These commands plan three RQ1 conditions and six RQ3 conditions for the default
task `dl-t4`. Their manifests have `status: dry_run`; they contain no measured
model performance.

### 3. Configure model roles

```bash
.venv-ha/bin/python configure.py
```

The wizard creates `local/profile.yaml` with endpoint URLs and exact model IDs.
It configures five roles: `actor`, `world_model`, `judge`, `policy_extractor`, and
`action_agent`. By default, the last two share the world-model and actor endpoints,
respectively. Review sampling settings, model revisions, and budgets before use.

Set credentials in the shell; the wizard does not store them in the profile:

```bash
read -r -s -p 'Actor API key: ' ACTOR_API_KEY; echo
read -r -s -p 'World-model API key: ' WM_API_KEY; echo
read -r -s -p 'Evaluator API key: ' JUDGE_API_KEY; echo
export ACTOR_API_KEY WM_API_KEY JUDGE_API_KEY
```

Endpoint URLs can be configured independently. For a local endpoint without
authentication, set `auth: none` and remove `api_key_env` from the corresponding
endpoint in the profile. See [model configuration](docs/SETUP.md) for details.

### 4. Run a small experiment

The following commands call your configured models and may incur API charges:

```bash
# RQ1: three prediction-involvement conditions on one task
.venv-ha/bin/python quickstart.py rq1 --tasks dl-t4 --output ../ha-rq1-demo

# RQ3: six feedback conditions on the same task
.venv-ha/bin/python quickstart.py rq3 --tasks dl-t4 --output ../ha-rq3-demo
```

Outputs are written outside the checkout. Reuse an output directory only when
resuming the same configuration; use a new directory when changing models or
protocols. The example fixes seed 0, horizon 1, 24 turns, one candidate, and one
regeneration attempt. These are example settings, not a complete specification
of the paper's experiments.

## Experiments

`run.py` routes each research question to its native benchmark implementation:

```bash
.venv-ha/bin/python run.py rq1 ha -- --help
.venv-ha/bin/python run.py rq2 ha -- --help
.venv-ha/bin/python run.py rq3 ha -- --help
```

Replace `ha` with `isbench` for the corresponding IS-Bench entrypoint. Arguments
after `--` are passed to the native experiment. Use `--python /path/to/python`
before `--` to choose the benchmark interpreter.

### RQ1: Does stronger prediction involvement improve safety?

| Paper setting | Code option | Intervention |
| --- | --- | --- |
| No Prediction | `no_guard` | Execute without world-model feedback |
| Prediction-Informed | `advisory_only` | Allow the actor to consult predictions |
| Prediction-Constrained | `enforced_guard` | Check candidate actions and revise or block execution |

The quick-start command runs all three conditions on HarnessAudit. The shared
runner supports model-role profiles and benchmark-specific settings. See the
[RQ1 guide](rq1_online/README.md) for IS-Bench commands and scoring details.

### RQ2: Can world models reliably predict future risks?

Predict-only evaluation uses recorded trajectories to compare horizons 1–6, policy
information, and direct versus iterative rollout. It requires newly generated or
user-provided source reports.

RQ2 uses the native `--model`, `--api-url`, and `CUSTOM_API_KEY` interface rather
than the five-role `--model-profile`. A minimal HarnessAudit example generates
fresh source traces and then evaluates predictions:

```bash
export WM_MODEL='your-exact-model-id'
export WM_BASE_URL='https://your-endpoint.example/v1'
read -r -s -p 'Prediction API key: ' CUSTOM_API_KEY; echo
export CUSTOM_API_KEY

.venv-ha/bin/python run.py rq2 ha -- \
  --tasks dl-t4 --actor-trace-seed 0 --wm-seed 0 \
  --model "$WM_MODEL" --agent-model "$WM_MODEL" --judge-model "$WM_MODEL" \
  --llm-provider custom --api-url "$WM_BASE_URL" \
  --policy-modes full_harnessaudit_rules \
  --rollout-modes direct_multi_horizon iterative_wm_rollout \
  --horizons 1 2 3 4 5 6 --point-scheme start_mid --point-count 2 \
  --candidate-count 1 --workers 1 --skip-existing --skip-judge \
  --base-config "$PWD/source/SafePred/config/harnessaudit_online_openai_compatible.yaml" \
  --output-root "$PWD/../ha-rq2/results" \
  --trace-root "$PWD/../ha-rq2/traces" \
  --analysis-dir "$PWD/../ha-rq2/analysis" \
  --figure-dir "$PWD/../ha-rq2/figures"
```

Here, `--skip-judge` skips the LLM judge for fresh source trajectories. This example
uses full benchmark rules; training-derived policies require a separate audited
train/test split. See the [RQ2 guide](rq2_predict/README.md) for replaying existing
reports and handling IS-Bench camera frames.

### RQ3: Are decisions faithful to prediction content?

| Feedback condition | CLI value | HarnessAudit | IS-Bench |
| --- | --- | --- | --- |
| Empty feedback | `empty` | Implemented | Implemented |
| Irrelevant feedback | `irrelevant` | Implemented | Implemented |
| Non-informative filler | `filler` | Implemented | Implemented |
| Corrupted risk judgment | `corrupt` | Implemented | Implemented |
| Key-information character corruption | `corrupt_key_random_chars` | Implemented | Implemented |
| Sandbox-derived one-step feedback | `environment_h1` | Implemented | **Not implemented; rejected at launch** |

Under `canonical_wm_all_v1`, text perturbations are applied after obtaining a
valid world-model prediction. All listed text perturbations except `corrupt`
retain the canonical risk gate. This distinction matters when interpreting
retained safety gains. Sandbox-derived feedback includes additional actor-review
computation. See the [RQ3 guide](rq3_perturb/README.md) for protocols and commands.

## Reproducibility and release scope

This release provides the experiment implementations, launchers, benchmark inputs,
configuration tools, and offline tests. It supports running new experiments under
an explicitly specified configuration.

| Artifact | Availability |
| --- | --- |
| Experiment code, model interfaces, perturbations, scoring and analysis utilities | Included |
| HarnessAudit definitions, tool catalogs and initial fixtures | 210 tasks; 200 non-SDE tasks in the supplied non-SDE list |
| IS-Bench task definitions and scene initialization templates | 161 tasks; 563 templates |
| HA dependency wheels and pinned dependency list | Included for Linux x86_64 / Python 3.11 |
| Historical paper trajectories, scores, figures, request logs, and policy caches | Not included |
| Model weights, model-service access, and OmniGibson's full assets | External requirements |
| Frozen paper-run configurations and historical task cohorts | Not supplied as a complete paper-reproduction bundle |

The packaged benchmark inventory must not be treated as the denominator of every
experiment in the paper. Exact numerical reproduction also requires the original
model revisions, prompts and policies, cohort definitions, budgets, evaluator
configuration, and raw outputs. These commands alone do not reproduce the paper's
tables from released historical data.

For new experiments, record the Git commit, model revisions, role profile, task
list, seeds, generation settings, endpoint behavior, and evaluation coverage.
Preserve failures and missing predictions in the reported denominator. Resume
support validates saved state and identities; arbitrary remote side effects and
interrupted requests are not guaranteed to replay exactly once. See
[the code map and recovery notes](docs/CODE_MAP.md).

## Validation

```bash
# SHA-256 checks and parsing of every manifested Python source
python3 tools/verify_bundle.py

# Dependency imports, task inventory, and static fixture checks
.venv-ha/bin/python tools/doctor.py

# Six CLI help paths, profile loading, native dry-run matrices, and regression tests
.venv-ha/bin/python tools/smoke_test.py
```

The bundled smoke suite passes **174 tests** in the current local check
environment. One additional IS-Bench test module requires the separate simulator
dependency environment and was not validated here. These offline checks do not
validate model quality, paid API endpoints, or simulator execution. See
[docs/VALIDATION.md](docs/VALIDATION.md) for the environment, complete check scope,
and the optional Hugging Face backend finding.

## Repository layout

```text
.
├── run.py                     # Native RQ × benchmark launcher
├── quickstart.py              # Small HarnessAudit RQ1 / RQ3 experiments
├── configure.py               # Credential-free model-role configuration
├── install.sh                 # Supported HA environment installation
├── requirements-ha.lock.txt   # Pinned Linux/Python 3.11 dependencies
├── inputs/                    # Static benchmark task lists
├── rq1_online/                # Prediction-involvement experiment guide
├── rq2_predict/               # Prediction-reliability experiment guide
├── rq3_perturb/               # Decision-faithfulness experiment guide
├── source/
│   ├── experiments/            # Orchestration, model roles, audits, analysis
│   ├── SafePred/               # Prediction interfaces and supporting components
│   ├── harness_env/orgin/HarnessAudit/
│   └── IS-Bench/               # Embodied benchmark adapters and inputs
├── tools/                     # Distribution, input, and smoke checks
├── wheels/                    # HA offline installation dependencies
└── docs/                      # Setup, source map, packaging, validation
```

The `orgin` directory spelling is preserved from the source snapshot. Detailed
module-level entrypoints are listed in [docs/CODE_MAP.md](docs/CODE_MAP.md).

## Citation

If you use this work, please cite the arXiv preprint:

```bibtex
@misc{zhao2026predictivesafety,
  title         = {Predictive Safety Guardrails Do Not Always Make LLM Agents Safer},
  author        = {Weixiang Zhao and Zhihao Liu and Guancheng Li and Yanyan Zhao and Bing Qin and Ting Liu},
  year          = {2026},
  eprint        = {2611.02815},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2611.02815}
}
```

For code-specific reproducibility, also record the repository URL and exact Git
commit used in your experiments.

## Acknowledgments and license

This study builds on [HarnessAudit](https://github.com/eric-ai-lab/HarnessAudit),
[IS-Bench](https://github.com/AI45Lab/IS-Bench),
[BDDL](https://github.com/StanfordVL/bddl), and SafePred components included in
`source/SafePred/`. We thank their contributors.

The [MIT License](LICENSE) covers this project's original contributions.
Third-party components and assets retain their own terms, including the bundled
[HarnessAudit license](source/harness_env/orgin/HarnessAudit/LICENSE) and
[BDDL license](source/IS-Bench/bddl/LICENSE). The top-level license does not
relicense third-party resources. See [packaging notes](docs/PACKAGING.md).

For bugs or reproducibility questions, please open a
[GitHub issue](https://github.com/REH502/rethinking-predictive-safety/issues)
with the commit, environment, command, and a redacted traceback.

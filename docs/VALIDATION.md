# Release validation

Validation date: **2026-10-30**. This report covers the documentation update and
repository rename to `REH502/rethinking-predictive-safety`.

## Environment

Checks ran in a fresh, temporary environment on **macOS / Apple Silicon, Python
3.12.14**, using dependencies installed from `source/requirements-ha.txt`.
The pinned clients were OpenAI 2.48.0, OpenAI Agents 0.18.3, HTTPX 0.28.1,
HTTPCore 1.0.9, and Pydantic 2.13.4. Unpinned dependencies were resolved for this
host, including NumPy 2.5.4, pandas 3.1.0, Matplotlib 3.11.2, and pytest 9.2.0.

This is separate from the bundled **Linux x86_64 / Python 3.11** wheel environment.
No live model evaluations or GPU simulation episodes were run.

## Results

| Check | Result | Interpretation |
| --- | --- | --- |
| Original distribution manifest before edits | PASS: 1,841 entries; 320 Python files parse | All originally manifested files were present and matched the prior checksums |
| Comparison with the supplied September 26 ZIP | PASS: no missing files | Before adding this report, 1,840 files matched byte-for-byte; only README and its manifest differed |
| All Python syntax | PASS: 320 files, including 313 under `source/` | Parsing does not establish that all optional imports or runtime branches work |
| Structured source inputs | PASS: 848 JSON and 224 YAML files parse | No malformed JSON/YAML found in the checked source tree |
| Static task lists | PASS: 210 HA, 200 HA non-SDE, 161 IS-Bench | No duplicate IDs; every listed task has a corresponding definition |
| Input doctor | PASS: 210 HA tasks/assets, 8 tool catalogs, 161 IS-Bench tasks, 563 scene templates | Task assets and initial fixture references checked by the native loader |
| Model-profile generation and loading | PASS: all five roles load | Uses a credential-free temporary test profile |
| CLI loading | PASS: all six RQ × benchmark `--help` paths | Entry modules and parsers load in this environment |
| Native RQ1 / RQ3 dry runs | PASS: 3 / 6 planned conditions | Generated manifests contain only `dry_run` rows |
| Bundled regression suite | **174 passed** across 11 test modules | Model roles, endpoint preflight, feedback transforms, checkpoints, sandbox delivery, and tool-input boundaries |
| Full test-directory collection | **Blocked for one additional IS-Bench module** | See the dependency limitation below |
| Installed dependency consistency | PASS: `pip check` | No conflicting requirements in the temporary check environment |
| Linux wheel dependency resolution | PASS: all 63 pinned requirements resolve offline | Cross-platform pip dry-run, CPython 3.11, compatible manylinux tags through glibc 2.34 |
| Installer shell syntax | PASS: `bash -n install.sh` | The Linux installer was not executed on macOS |
| README examples | PASS: 10 Bash blocks parse; the full RQ2 example is accepted by its native argument parser | Experiment execution stopped after parsing, before any model call |
| Documentation links | PASS: 14 relative links resolve | Checked the rewritten README and this report |

The original project Python sources and benchmark assets were not changed by the
README rewrite. The release manifest is refreshed to include the updated
documentation, the top-level license, and `.gitignore`.

## Repeating the supported checks

After creating the HA environment on the supported Linux platform:

```bash
python3 tools/verify_bundle.py
.venv-ha/bin/python tools/doctor.py
.venv-ha/bin/python tools/smoke_test.py
.venv-ha/bin/python -m pip check
```

`tools/smoke_test.py` clears model credentials from its child environment, directs
outputs to temporary locations, generates a test-only model profile, invokes the
six help paths, checks native dry-run manifests, and runs its selected regression
modules. It does not make model calls. The endpoint tests use mocked transports.

## Remaining limitations and findings

### IS-Bench test collection requires its separate dependency environment

Running pytest over both full test directories encountered an import error in
`source/experiments/tests/test_isbench_wm_return_perturbations.py`:

```text
ModuleNotFoundError: No module named 'transformers'
```

Importing the safeguard imports `og_ego_prim.models.__init__`, which eagerly loads
`PlanningAgent` and its Hugging Face and simulator dependencies. The HA check
environment does not include these dependencies. `transformers==4.51.3` is listed
in `source/requirements-isbench.txt`; the full IS-Bench setup also needs PyTorch,
OmniGibson, and its assets. That additional module is not part of the 174 passing
tests and was not reported as passing. No stubs were added to bypass its imports.

### Optional Hugging Face client needs runtime review

Static inspection of `source/IS-Bench/og_ego_prim/models/hf_inference.py` found
naming inconsistencies in the bundled optional `HFClient`: `self.model` is both a
loaded model attribute and the public method name; initialization assigns
`self.preprocessor`, while the method references `self.processor`.
This backend was not exercised, and no repair is included in this documentation
update. The README's examples use model endpoints and do not advertise this
optional direct-loading backend as validated.

### Scope of the checks

- Linux wheel resolution validates package metadata and availability, not native
  binary execution or successful Linux installation.
- Full HarnessAudit inference needs reachable model endpoints and task-level
  evaluation. Full IS-Bench validation needs a real compatible GPU environment.
- IS-Bench `environment_h1` remains unimplemented and is explicitly rejected.
- Historical trajectories, scores, policy caches, and the complete frozen paper
  experiment specification are not released in this snapshot. Exact paper-table
  reproduction therefore remains outside the validated scope.

The source-file inventory is complete relative to the supplied ZIP. This should
not be interpreted as certification of every optional backend or exact numerical
reproduction of the paper.

"""Create a shallow, non-destructive catalog of historical experiment runs."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CATALOG_VERSION = 1


def _classify(name: str) -> tuple[str, str]:
    lowered = name.lower()
    has_isbench = "isbench" in lowered or lowered.startswith("isb")
    has_ha = any(token in lowered for token in ("harness", "harnessaudit", "ha_", "four_tier"))
    if has_isbench and has_ha:
        benchmark = "combined"
    elif has_isbench:
        benchmark = "isbench"
    elif has_ha:
        benchmark = "harnessaudit"
    else:
        benchmark = "unclassified"

    if any(token in lowered for token in ("online", "safeguard", "guard")):
        stage = "online_safeguard"
    elif any(token in lowered for token in ("predict", "prediction", "tier")):
        stage = "offline_prediction"
    elif any(token in lowered for token in ("trace", "report")):
        stage = "trace_generation"
    elif any(token in lowered for token in ("vllm", "service")):
        stage = "model_service"
    elif any(token in lowered for token in ("monitor", "watch")):
        stage = "monitoring"
    else:
        stage = "other"
    return benchmark, stage


def _job_ids(name: str, children: list[str]) -> list[str]:
    values = re.findall(r"(?:slurm-|job[_-]?)(\d{5,})", name, flags=re.IGNORECASE)
    terminal = re.search(r"[_-](\d{5,6})$", name)
    if terminal:
        values.append(terminal.group(1))
    for child in children:
        values.extend(re.findall(r"(?:slurm-|job[_-]?)(\d{5,})", child, flags=re.IGNORECASE))
    return sorted(set(values))


def _status(children: set[str]) -> str:
    lowered = {item.lower() for item in children}
    if lowered & {"done", "completed", ".completed", "success"}:
        return "completed"
    if any(item.endswith((".err", ".error")) for item in lowered):
        return "inspect"
    if any("manifest" in item or "summary" in item for item in lowered):
        return "has_artifacts"
    return "unknown"


def scan_runs(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not runs_root.is_dir():
        return rows
    for run_dir in sorted(runs_root.iterdir(), key=lambda item: item.name.lower()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        try:
            children = sorted(item.name for item in run_dir.iterdir())
            stat = run_dir.stat()
        except OSError as exc:
            rows.append(
                {
                    "name": run_dir.name,
                    "path": str(run_dir),
                    "benchmark": "unreadable",
                    "stage": "unknown",
                    "status": "error",
                    "modified_at": None,
                    "job_ids": [],
                    "top_level_entries": [],
                    "error": str(exc),
                }
            )
            continue
        benchmark, stage = _classify(run_dir.name)
        rows.append(
            {
                "name": run_dir.name,
                "path": str(run_dir),
                "benchmark": benchmark,
                "stage": stage,
                "status": _status(set(children)),
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "job_ids": _job_ids(run_dir.name, children),
                "top_level_entries": children,
            }
        )
    return rows


def _write_readme(path: Path, rows: list[dict[str, Any]]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["benchmark"], row["stage"])
        counts[key] = counts.get(key, 0) + 1
    lines = [
        "# Run Data Catalog",
        "",
        "This directory contains immutable historical outputs. Do not rename or delete a run",
        "directory merely to tidy the tree: reports and provenance files may contain absolute paths.",
        "`index.json` and `catalog.csv` are generated views and may be rebuilt safely.",
        "",
        "Regenerate from the project root:",
        "",
        "```bash",
        "./wm-safety index-runs",
        "```",
        "",
        "The status is intentionally conservative: `unknown` does not mean failed; it means the",
        "shallow catalog found no standard completion marker. Inspect that run's manifest/logs.",
        "",
        "## Summary",
        "",
        "| Benchmark | Stage | Directories |",
        "|---|---|---:|",
    ]
    for (benchmark, stage), count in sorted(counts.items()):
        lines.append(f"| {benchmark} | {stage} | {count} |")
    lines.extend(
        [
            "",
            "## Directory Index",
            "",
            "| Directory | Benchmark | Stage | Status | Job IDs |",
            "|---|---|---|---|---|",
        ]
    )
    for row in rows:
        job_ids = ", ".join(row["job_ids"]) or "-"
        lines.append(
            f"| `{row['name']}` | {row['benchmark']} | {row['stage']} | "
            f"{row['status']} | {job_ids} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_catalog(runs_root: Path) -> dict[str, Path | int]:
    runs_root.mkdir(parents=True, exist_ok=True)
    rows = scan_runs(runs_root)
    generated_at = datetime.now(timezone.utc).isoformat()

    json_path = runs_root / "index.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": CATALOG_VERSION,
                "generated_at": generated_at,
                "runs_root": str(runs_root.resolve()),
                "run_count": len(rows),
                "runs": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = runs_root / "catalog.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "name",
                "path",
                "benchmark",
                "stage",
                "status",
                "modified_at",
                "job_ids",
                "top_level_entries",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key) for key in writer.fieldnames},
                    "job_ids": ";".join(row.get("job_ids", [])),
                    "top_level_entries": ";".join(row.get("top_level_entries", [])),
                }
            )

    readme_path = runs_root / "README.md"
    _write_readme(readme_path, rows)
    return {
        "run_count": len(rows),
        "json": json_path,
        "csv": csv_path,
        "readme": readme_path,
    }

"""Audit frozen HarnessAudit reports and emit a clean replay cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .predict_only_pipeline import audit_clean_source_reports


def _read_paths(path: Path) -> list[Path]:
    return [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-list", type=Path, required=True)
    parser.add_argument("--valid-list", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--excluded-list", type=Path, required=True)
    parser.add_argument("--minimum-valid", type=int, default=1)
    args = parser.parse_args()

    source_paths = _read_paths(args.source_list)
    audit = audit_clean_source_reports(source_paths)
    valid = [row["report_path"] for row in audit["reports"] if row["is_clean_replayable"]]
    excluded = [row["report_path"] for row in audit["reports"] if not row["is_clean_replayable"]]

    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.audit_json.with_suffix(args.audit_json.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.audit_json)
    _write_lines(args.valid_list, valid)
    _write_lines(args.excluded_list, excluded)

    if len(valid) < args.minimum_valid:
        raise SystemExit(f"Only {len(valid)} clean reports; require at least {args.minimum_valid}")
    print(
        f"HarnessAudit preflight: {len(valid)}/{len(source_paths)} clean reports; "
        f"excluded={len(excluded)}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
import json
from pathlib import Path

from .paths import project_root


def main():
    root = project_root()
    parser = argparse.ArgumentParser(description="AS2 EDU + Piper-H whole-body simulation tools")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-assets", help="Fetch pinned official models and validate source checksums")
    fetch.add_argument("--sources", type=Path, default=root/"configs/sources.json")
    fetch.add_argument("--output", type=Path, default=root/"assets/upstream")
    audit = commands.add_parser("audit", help="Audit hardware readiness; returns 2 when M0 is unresolved")
    audit.add_argument("--hardware", type=Path, default=root/"configs/hardware.json")
    audit.add_argument("--upstream", type=Path, default=root/"assets/upstream")
    audit.add_argument("--output", type=Path, default=root/"artifacts/audit")
    args = parser.parse_args()
    if args.command == "fetch-assets":
        from .assets.fetch import fetch_sources
        result = fetch_sources(args.sources, args.output)
        print(json.dumps({source["id"]: len(source["files"]) for source in result["sources"]}, indent=2))
    elif args.command == "audit":
        from .assets.audit import audit_hardware, write_audit
        result = audit_hardware(args.hardware, args.upstream)
        write_audit(result, args.output)
        print(json.dumps({"ready_for_training": result["ready_for_training"], "issues": result["issues"],
                          "report": str(args.output / "hardware-audit.md")}, indent=2))
        if not result["ready_for_training"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()


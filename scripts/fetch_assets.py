#!/usr/bin/env python3
import json
from loco_manipulation.assets.fetch import fetch_sources
from loco_manipulation.paths import project_root

if __name__ == "__main__":
    root = project_root()
    result = fetch_sources(root / "configs/sources.json", root / "assets/upstream")
    print(json.dumps({source["id"]: len(source["files"]) for source in result["sources"]}, indent=2))

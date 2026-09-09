"""Download only pinned model files, validating Git blob identities before use."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath


def _get(url: str) -> bytes:
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "PawWeaver-asset-audit/0.1"})
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def fetch_sources(config: Path, destination: Path) -> dict:
    specification = json.loads(config.read_text())
    provenance = {"schema_version": 1, "sources": []}
    destination.mkdir(parents=True, exist_ok=True)
    for source in specification["sources"]:
        repository, revision = source["repository"], source["revision"]
        tree = json.loads(_get(f"https://api.github.com/repos/{repository}/git/trees/{revision}?recursive=1"))
        if tree.get("truncated"):
            raise ValueError(f"Git tree truncated for {repository}")
        entries = [entry for entry in tree["tree"] if entry["type"] == "blob" and (
            (any(entry["path"].startswith(prefix) for prefix in source["prefixes"])
             and Path(entry["path"]).suffix.lower() in source["suffixes"])
            or entry["path"] in source["license_names"])]

        def download(entry: dict) -> dict:
            relative = PurePosixPath(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe upstream path")
            target = destination / source["id"] / relative
            url = f"https://raw.githubusercontent.com/{repository}/{revision}/{relative}"
            data = target.read_bytes() if target.exists() else _get(url)
            if git_blob_sha(data) != entry["sha"]:
                raise ValueError(f"Git blob checksum mismatch: {target}; remove corrupted cache and retry")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            return {"path": str(target.relative_to(destination)), "git_blob": entry["sha"],
                    "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "url": url}

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            files = list(pool.map(download, entries))
        provenance["sources"].append({**source, "files": sorted(files, key=lambda item: item["path"])})
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


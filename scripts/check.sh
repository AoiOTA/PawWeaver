#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_dir"
if [[ "${CONDA_DEFAULT_ENV:-}" != pawweaver-* ]]; then
  echo 'Activate pawweaver-runtime or pawweaver-train first.' >&2
  exit 2
fi
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest "$@"
git diff --check


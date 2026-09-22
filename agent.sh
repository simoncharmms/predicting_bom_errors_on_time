#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"
export MPLBACKEND=Agg
exec .venv/bin/python agent_main.py "$@"

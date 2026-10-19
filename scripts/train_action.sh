#!/usr/bin/env bash
set -euo pipefail
python -m worldla train --config "$(dirname "$0")/../configs/action_100k.yaml" "$@"

#!/bin/bash
set -euo pipefail
error_handler() {
    echo "Error on line ${BASH_LINENO[0]}: ${BASH_COMMAND}"
}
trap 'error_handler' ERR

source .venv/bin/activate

if [[ "${1:-}" == "--check" ]]; then
    ruff check .
    basedpyright
    ruff format --check
else
    ruff check --fix .
    basedpyright
    ruff format
fi

#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <lab-path>"
  echo "Example: $0 labs/python-rag"
  exit 2
}

[[ $# -eq 1 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_REL="$1"
LAB_DIR="$ROOT_DIR/$LAB_REL"

[[ -d "$LAB_DIR" ]] || { echo "❌ No such directory: $LAB_REL"; exit 1; }

REQ_FILE="$LAB_DIR/requirements.txt"
VENV_DIR="${VENV_DIR:-$LAB_DIR/.venv}"

# Kernel naming
LAB_BASENAME="$(basename "$LAB_DIR")"
# Make a stable kernel name based on the path, safe chars only
KERNEL_NAME_DEFAULT="$(echo "$LAB_REL" | sed 's|/|-|g' | sed 's|[^A-Za-z0-9._-]|-|g')"
KERNEL_NAME="${KERNEL_NAME:-$KERNEL_NAME_DEFAULT}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-Lab: $LAB_BASENAME}"

PY="$VENV_DIR/bin/python"
PIP="$PY -m pip"

echo "▶️  Lab setup: $LAB_REL"
echo "   venv: $VENV_DIR"
echo "   kernel: $KERNEL_NAME  ($KERNEL_DISPLAY_NAME)"
echo

# Create venv if missing
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# Upgrade tooling
$PIP install --upgrade pip setuptools wheel

# Install requirements if present
if [[ -f "$REQ_FILE" ]]; then
  $PIP install -r "$REQ_FILE"
else
  echo "⚠️  No requirements.txt found in $LAB_REL (skipping requirements install)"
fi

# Ensure kernel support + register kernel (safe to rerun)
$PIP install --upgrade ipykernel
$PY -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME" --replace

# Optional: install Jupyter into the lab venv (off by default)
if [[ "${INSTALL_JUPYTER:-0}" == "1" ]]; then
  $PIP install --upgrade jupyterlab notebook
fi

echo
echo "✅ Done."
echo "Next:"
echo "  cd \"$LAB_REL\""
echo "  source .venv/bin/activate"
echo "  (In Jupyter) Kernel -> Change Kernel -> \"$KERNEL_DISPLAY_NAME\""


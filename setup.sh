#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <lab-path>"
  echo "Example: $0 30-advanced_retrieval_for_AI_with_chroma"
  echo "Example: $0 labs/python-rag"
  exit 2
}

[[ $# -eq 1 ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Strip trailing slashes
LAB_REL="${1%/}"
LAB_DIR="$ROOT_DIR/$LAB_REL"

[[ -d "$LAB_DIR" ]] || { echo "❌ No such directory: $LAB_REL"; exit 1; }

REQ_FILE="$LAB_DIR/requirements.txt"
VENV_DIR="${VENV_DIR:-$LAB_DIR/.venv}"

LAB_BASENAME="$(basename "$LAB_DIR")"

# Stable kernel name derived from path (safe characters only)
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

# Ensure ipykernel installed
$PIP install --upgrade ipykernel

# Register kernel: use --replace if available; otherwise remove then install
if $PY -m ipykernel install -h 2>&1 | grep -q -- '--replace'; then
  $PY -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME" --replace
else
  # Remove existing kernelspec if present (ignore errors)
  if command -v jupyter >/dev/null 2>&1; then
    jupyter kernelspec remove -f "$KERNEL_NAME" >/dev/null 2>&1 || true
  else
    # Fallback: delete user kernelspec directory if it exists
    rm -rf "$HOME/.local/share/jupyter/kernels/$KERNEL_NAME" 2>/dev/null || true
  fi
  $PY -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"
fi

echo
echo "✅ Done."
echo "Next:"
echo "  cd \"$LAB_REL\""
echo "  source .venv/bin/activate"
echo "  (In Jupyter) Kernel -> Change Kernel -> \"$KERNEL_DISPLAY_NAME\""

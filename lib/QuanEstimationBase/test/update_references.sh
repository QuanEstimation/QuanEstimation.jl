#!/bin/bash
# update_references.sh — Regenerate all golden reference files
# Usage: ./test/update_references.sh
# Requires: Julia project environment with JSON package

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Regenerating golden references..."
julia --project test/generate_references.jl
echo "Done."

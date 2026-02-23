#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="docs/feature_store/diagrams"
OUTPUT_DIR="docs/feature_store/diagrams"
INPUT_FILE="${INPUT_DIR}/feature_store_target_model.mmd"
OUTPUT_FILE="${OUTPUT_DIR}/feature_store_target_model.svg"
MMDC="./node_modules/.bin/mmdc"

if [[ ! -x "${MMDC}" ]]; then
  echo "Error: Mermaid CLI is not installed locally."
  echo "Run from repo root:"
  echo "  npm install"
  exit 1
fi

echo "Rendering ${INPUT_FILE} -> ${OUTPUT_FILE}"
"${MMDC}" -i "${INPUT_FILE}" -o "${OUTPUT_FILE}"

echo "Done."

#!/usr/bin/env bash
set -eo pipefail

# Ensure the script runs relative to the psierra directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# If script is located inside scripts/, PROJECT_ROOT is the parent directory
if [[ -f "${SCRIPT_DIR}/../pyproject.toml" ]]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/pyproject.toml" ]]; then
    PROJECT_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/psierra/pyproject.toml" ]]; then
    PROJECT_ROOT="${SCRIPT_DIR}/psierra"
else
    PROJECT_ROOT="${SCRIPT_DIR}"
fi

cd "${PROJECT_ROOT}"

echo "======================================================================"
echo " Running all pSierra examples via uv"
echo " Working directory: ${PROJECT_ROOT}"
echo "======================================================================"

# Determine Python runner: uv run if available, otherwise fallback to local .venv
if command -v uv >/dev/null 2>&1; then
    RUNNER="uv run"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    echo "[INFO] 'uv' command not found in PATH; falling back to '.venv/bin/python'."
    RUNNER="${PROJECT_ROOT}/.venv/bin/python"
else
    echo "[ERROR] Neither 'uv' nor a virtual environment (.venv/bin/python) was found."
    exit 1
fi

TOTAL=0
PASSED=0
FAILED=0
FAILED_SCRIPTS=()

# Find and sort all example python scripts across examples/
mapfile -t EXAMPLE_FILES < <(find examples -type f -name "*.py" | sort)

if [[ ${#EXAMPLE_FILES[@]} -eq 0 ]]; then
    echo "[WARNING] No python files found in examples directory."
    exit 0
fi

echo "Discovered ${#EXAMPLE_FILES[@]} example script(s)."
echo ""

START_TIME=$(date +%s)

for script in "${EXAMPLE_FILES[@]}"; do
    TOTAL=$((TOTAL + 1))
    echo "----------------------------------------------------------------------"
    echo "[${TOTAL}/${#EXAMPLE_FILES[@]}] Running: ${script}"
    echo "----------------------------------------------------------------------"

    if ${RUNNER} "${script}"; then
        PASSED=$((PASSED + 1))
        echo "--> SUCCESS: ${script}"
    else
        FAILED=$((FAILED + 1))
        FAILED_SCRIPTS+=("${script}")
        echo "--> FAILED: ${script}"
    fi
    echo ""
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "======================================================================"
echo " Execution Summary"
echo " Total scripts : ${TOTAL}"
echo " Passed        : ${PASSED}"
echo " Failed        : ${FAILED}"
echo " Total time    : ${ELAPSED}s"
echo "======================================================================"

if [[ ${FAILED} -gt 0 ]]; then
    echo ""
    echo "Failed scripts:"
    for f in "${FAILED_SCRIPTS[@]}"; do
        echo "  - ${f}"
    done
    exit 1
fi

echo "All example scripts executed successfully!"
exit 0

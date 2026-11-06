#!/usr/bin/env bash
set -eo pipefail

# ==============================================================================
# pSierra Example Runner
# ==============================================================================
# Executes example scripts specified in scripts/examples.txt or passed via CLI.
# Usage:
#   ./scripts/run_examples.sh                # Runs examples defined in scripts/examples.txt
#   ./scripts/run_examples.sh 0101 0202      # Runs only the specified 4-digit IDs
#   ./scripts/run_examples.sh --all          # Runs all 35 examples
#   ./scripts/run_examples.sh --config FILE  # Runs examples defined in custom manifest FILE
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure script runs from the repository root
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

# Determine Python runner: uv run if available, otherwise local .venv
if command -v uv >/dev/null 2>&1; then
    RUNNER="uv run"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    RUNNER="${PROJECT_ROOT}/.venv/bin/python"
else
    echo "[ERROR] Neither 'uv' nor a virtual environment (.venv/bin/python) was found."
    exit 1
fi

MANIFEST_FILE="${SCRIPT_DIR}/examples.txt"
RUN_ALL=false
CLI_TARGETS=()

# Parse Command-Line Options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: $0 [OPTIONS] [TARGET_ID ...]"
            echo ""
            echo "Options:"
            echo "  --all              Run every example script discovered in examples/"
            echo "  --config <file>    Specify custom manifest file (default: scripts/examples.txt)"
            echo "  -h, --help         Show this help message and exit"
            echo ""
            echo "Examples:"
            echo "  $0                 # Run active scripts listed in scripts/examples.txt"
            echo "  $0 0101 0201 0501  # Run specific example IDs"
            echo "  $0 --all           # Run all examples"
            exit 0
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --config)
            MANIFEST_FILE="$2"
            shift 2
            ;;
        *)
            CLI_TARGETS+=("$1")
            shift
            ;;
    esac
done

TARGET_RUNS=()   # Array of "script_path [args...]"

# Helper: Find script by ID or name
find_script() {
    local query="$1"
    local match=""

    # 1. Direct path check
    if [[ -f "${query}" ]]; then
        echo "${query}"
        return 0
    elif [[ -f "examples/${query}" ]]; then
        echo "examples/${query}"
        return 0
    fi

    # 2. Match by 4-digit ID or prefix pattern under examples/
    match=$(find examples -type f -name "${query}*.py" 2>/dev/null | sort | head -n 1)
    if [[ -n "${match}" ]]; then
        echo "${match}"
        return 0
    fi

    # 3. Match anywhere in filename
    match=$(find examples -type f -name "*${query}*.py" 2>/dev/null | sort | head -n 1)
    if [[ -n "${match}" ]]; then
        echo "${match}"
        return 0
    fi

    return 1
}

# Collect target scripts
if [[ "${RUN_ALL}" == true ]]; then
    echo "[INFO] Running ALL discovered examples in examples/"
    while IFS= read -r script_path; do
        if [[ "${script_path}" =~ 06_interactive ]]; then
            TARGET_RUNS+=("${script_path} --headless")
        else
            TARGET_RUNS+=("${script_path}")
        fi
    done < <(find examples -type f -name "*.py" | sort)

elif [[ ${#CLI_TARGETS[@]} -gt 0 ]]; then
    echo "[INFO] Running specific targets requested via CLI: ${CLI_TARGETS[*]}"
    for target in "${CLI_TARGETS[@]}"; do
        if resolved=$(find_script "${target}"); then
            if [[ "${resolved}" =~ 06_interactive ]]; then
                TARGET_RUNS+=("${resolved} --headless")
            else
                TARGET_RUNS+=("${resolved}")
            fi
        else
            echo "[ERROR] Could not resolve example script for target: '${target}'"
            exit 1
        fi
    done

else
    if [[ ! -f "${MANIFEST_FILE}" ]]; then
        echo "[ERROR] Manifest file not found: ${MANIFEST_FILE}"
        echo "        Create '${MANIFEST_FILE}' or pass targets via CLI (e.g. $0 0101)."
        exit 1
    fi

    echo "[INFO] Reading execution manifest: ${MANIFEST_FILE}"
    while IFS= read -r line || [[ -n "${line}" ]]; do
        # Strip leading/trailing whitespace
        line=$(echo "${line}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')

        # Skip empty lines or pure comment lines
        [[ -z "${line}" || "${line}" =~ ^# ]] && continue

        # Strip inline comments
        clean_line=$(echo "${line}" | sed 's/[[:space:]]*#.*$//')
        [[ -z "${clean_line}" ]] && continue

        # Parse target identifier and extra arguments
        read -r target_id extra_args <<< "${clean_line}"

        if resolved=$(find_script "${target_id}"); then
            # Ensure interactive scripts default to headless if not specified
            if [[ "${resolved}" =~ 06_interactive ]] && [[ ! "${extra_args}" =~ --headless ]]; then
                extra_args="--headless ${extra_args}"
            fi

            if [[ -n "${extra_args}" ]]; then
                TARGET_RUNS+=("${resolved} ${extra_args}")
            else
                TARGET_RUNS+=("${resolved}")
            fi
        else
            echo "[WARNING] Could not resolve script for entry: '${target_id}' (skipping)"
        fi
    done < "${MANIFEST_FILE}"
fi

if [[ ${#TARGET_RUNS[@]} -eq 0 ]]; then
    echo "[WARNING] No active example scripts found to execute."
    exit 0
fi

echo "======================================================================"
echo " pSierra Example Runner"
echo " Total target(s) to execute: ${#TARGET_RUNS[@]}"
echo " Working directory         : ${PROJECT_ROOT}"
echo " Runner                    : ${RUNNER}"
echo "======================================================================"
echo ""

TOTAL=0
PASSED=0
FAILED=0
FAILED_SCRIPTS=()

START_TIME=$(date +%s)

for item in "${TARGET_RUNS[@]}"; do
    TOTAL=$((TOTAL + 1))
    echo "----------------------------------------------------------------------"
    echo "[${TOTAL}/${#TARGET_RUNS[@]}] Running: ${item}"
    echo "----------------------------------------------------------------------"

    # Execute script and its arguments
    if ${RUNNER} ${item}; then
        PASSED=$((PASSED + 1))
        echo "--> SUCCESS: ${item}"
    else
        FAILED=$((FAILED + 1))
        FAILED_SCRIPTS+=("${item}")
        echo "--> FAILED: ${item}"
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

echo "All specified examples executed successfully!"
exit 0

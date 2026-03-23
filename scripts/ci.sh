#!/usr/bin/env bash
# CI script for bead package
#
# Runs all CI checks except slow model training tests.
# To run slow tests: uv run pytest -m slow_model_training
#
# Usage:
#   ./scripts/ci.sh           # run all checks
#   ./scripts/ci.sh --quick   # skip tests, run only linting and type checking

set -euo pipefail

# colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # no color

# track failures
FAILED=0

log_step() {
    echo -e "\n${YELLOW}==>${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_failure() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAILED=1
}

run_check() {
    local name="$1"
    shift
    log_step "$name"
    if "$@"; then
        log_success "$name"
    else
        log_failure "$name"
    fi
}

# parse arguments
QUICK_MODE=false
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            shift
            ;;
    esac
done

# change to project root
cd "$(dirname "$0")/.."

echo "========================================"
echo "  bead CI checks"
echo "========================================"

# 1. type checking with pyright
run_check "Type checking (pyright)" uv run pyright bead/

# 2. linting with ruff
run_check "Linting (ruff check)" uv run ruff check bead/

# 3. format check with ruff
run_check "Format check (ruff format)" uv run ruff format --check bead/

# 4. docstring validation with pydocstyle (if pre-commit is available)
# note: pydocstyle exits non-zero on parse warnings even without violations,
# so we check for actual D-code violations in the output
if command -v pre-commit &> /dev/null; then
    log_step "Docstring validation (pydocstyle)"
    PYDOC_OUTPUT=$(pre-commit run pydocstyle --all-files 2>&1 || true)
    # check for actual violations (lines containing D followed by 3 digits and a colon)
    if echo "$PYDOC_OUTPUT" | grep -qE "D[0-9]{3}:"; then
        echo "$PYDOC_OUTPUT"
        log_failure "Docstring validation (pydocstyle)"
    else
        log_success "Docstring validation (pydocstyle)"
    fi
else
    log_step "Docstring validation (pydocstyle)"
    echo "  Skipping: pre-commit not installed"
fi

# 5. run tests (excluding slow model training tests)
if [ "$QUICK_MODE" = false ]; then
    run_check "Tests (excluding slow model training)" \
        uv run pytest -m "not slow_model_training" --tb=short -q
else
    log_step "Tests"
    echo "  Skipping: --quick mode"
fi

# summary
echo ""
echo "========================================"
if [ $FAILED -eq 0 ]; then
    echo -e "  ${GREEN}All CI checks passed${NC}"
    echo "========================================"
    exit 0
else
    echo -e "  ${RED}Some CI checks failed${NC}"
    echo "========================================"
    exit 1
fi

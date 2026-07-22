#!/bin/bash
#
# Compiles and runs the YASSO test program. Writes yasso_output.csv into the
# jsbach directory, which tests/test_JSBACH_model.py reads back.
#
# Safe to call from anywhere: all paths are resolved relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSBACH_DIR="$(dirname "$SCRIPT_DIR")"
cd "$JSBACH_DIR"

echo "=========================================="
echo "YASSO Subroutine Test Suite"
echo "=========================================="
echo "Working directory: $JSBACH_DIR"

FC=${FC:-gfortran}
if ! command -v "$FC" &> /dev/null; then
    echo "ERROR: Fortran compiler '$FC' not found"
    echo "Install gfortran or set FC to your Fortran compiler"
    exit 1
fi

echo "Using compiler: $(command -v "$FC") ($("$FC" --version | head -1))"
echo ""

echo "Cleaning up previous builds..."
rm -rf build_temp testing_src/test_yasso_call

echo "Starting compilation..."
./testing_src/compile_yasso_test.sh

echo ""
echo "=========================================="
echo "Running YASSO test program..."
echo "=========================================="
./testing_src/test_yasso_call

echo ""
echo "=========================================="
echo "Test completed successfully!"
echo "=========================================="

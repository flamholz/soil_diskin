#!/bin/bash
#
# Kept so that the instructions in YASSO_TEST_README.md still work.
# The real script lives next to the sources it builds.

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/testing_src/run_yasso_test.sh" "$@"

#!/usr/bin/env bash
# Runs the test suite in a fresh uv environment. The same script runs locally and in CI.
#
#   scripts/test.sh 3.12 with-mcp
#   scripts/test.sh 3.15 no-mcp
#
# The skip count is checked too: with mcp nothing may be skipped, without it exactly the
# MCP tests are skipped. A wrong count means the leg is not testing what it says.
set -euo pipefail

py="${1:?usage: scripts/test.sh PYTHON_VERSION with-mcp|no-mcp}"
mode="${2:?usage: scripts/test.sh PYTHON_VERSION with-mcp|no-mcp}"
cd "$(dirname "$0")/.."

args=(run --no-project --isolated --python "$py" --with-editable .)
case "$mode" in
  with-mcp) args+=(--with 'mcp>=2,<3'); want_skipped=0 ;;
  no-mcp) want_skipped=3 ;;
  *) echo "mode must be with-mcp or no-mcp" >&2; exit 2 ;;
esac

# Keep the owner's run history and API key out of the tests.
home_dir="$(mktemp -d)"
log="$(mktemp)"
trap 'rm -rf "$home_dir" "$log"' EXIT
export LLMTUNER_HOME="$home_dir"
unset OPENROUTER_API_KEY

uv "${args[@]}" python -m unittest discover -s tests 2>&1 | tee "$log"

# Not simply the last line: a test's buffered stdout can land after unittest's summary.
summary="$(grep -E '^(OK|FAILED)' "$log" | tail -n 1 || true)"
if [[ -z "$summary" ]]; then
  echo "no unittest summary line found" >&2
  exit 1
fi
skipped="$(sed -n 's/.*skipped=\([0-9]*\).*/\1/p' <<<"$summary")"
skipped="${skipped:-0}"
if [[ "$skipped" != "$want_skipped" ]]; then
  echo "expected $want_skipped skipped tests in mode $mode, got $skipped" >&2
  exit 1
fi
echo "ok: python $py, $mode, skipped=$skipped"

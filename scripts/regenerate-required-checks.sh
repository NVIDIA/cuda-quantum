#!/usr/bin/env bash
# Regenerate one section of .github/required-checks.yml from a CI run.
#
# Usage: scripts/regenerate-required-checks.sh <run-id> [event-type]
#
# Pulls every sub-job that ended with `conclusion: success` from the run,
# sorts and dedupes them, and writes the result under the section keyed
# by the run's event type (or the explicit override). Other sections are
# preserved.
#
# Run on a green CI run with the exact matrix you want enforced. Commit
# the diff. Requires: gh, jq, python3 (with pyyaml).

set -euo pipefail

run_id="${1:?usage: $0 <run-id> [event-type]}"
event_override="${2:-}"
repo="${REPO:-NVIDIA/cuda-quantum}"
out=".github/required-checks.yml"

event=$(gh api "repos/$repo/actions/runs/$run_id" --jq '.event')
[ -n "$event_override" ] && event="$event_override"

echo "Run: $repo/actions/runs/$run_id (event=$event)"

names=$(gh api --paginate "repos/$repo/actions/runs/$run_id/jobs" \
  --jq '.jobs[] | select(.conclusion == "success") | .name' | sort -u)

n=$(echo "$names" | grep -c . || true)
if [ "$n" -eq 0 ]; then
  echo "No successful jobs found on run $run_id; aborting." >&2
  exit 1
fi

python3 - "$out" "$event" <<PY
import sys, yaml, re
path, event = sys.argv[1], sys.argv[2]
names = """$names""".splitlines()
# Self-referential / parallel-running jobs cannot be required by the
# manifest check (which itself runs inside CI Summary and reads job state
# at that moment).
exclude = {"CI Summary", "Prepare cache clean-up"}
names = sorted({n for n in names if n and n not in exclude})

# Preserve any leading header comment(s) in the existing file.
header = ""
try:
    with open(path) as f:
        text = f.read()
    m = re.match(r"((?:\s*#[^\n]*\n)+)", text)
    if m:
        header = m.group(1)
    data = yaml.safe_load(text) or {}
except FileNotFoundError:
    data = {}

data[event] = names

with open(path, "w") as f:
    if header:
        f.write(header)
    yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=True, width=10000)
print(f"Wrote {len(names)} entries to {path} under .{event}")
PY

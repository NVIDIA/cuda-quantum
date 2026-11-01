#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Negative/positive tests for scripts/check_wheel_install_identity.sh:
# synthetic wheel + installed prefix in scratch; covers the byte-identity
# binding, fail-closed RECORD/member rules, and METADATA binding.
#
# Usage: test_check_wheel_install_identity.sh   (self-contained)

set -eu
export LC_ALL=C

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
failures=0

check() {   # check <name> <expect-pass|expect-fail> <expected-substring>
    local name="$1" expect="$2" want="$3"
    local out rc
    out="$(bash "$script_dir/check_wheel_install_identity.sh" \
        "$scratch/pkg-1.0-py3-none-any.whl" "$scratch/site" 2>&1)" && rc=0 || rc=$?
    if [ "$expect" = pass ] && [ "$rc" -ne 0 ]; then
        echo "NEG-TEST FAIL: $name: expected pass, exit $rc"; echo "$out" | head -3
        failures=$((failures + 1)); return
    fi
    if [ "$expect" = fail ]; then
        if [ "$rc" -eq 0 ]; then
            echo "NEG-TEST FAIL: $name: expected failure, audit passed"
            failures=$((failures + 1)); return
        fi
        if ! grep -qF "$want" <<< "$out"; then
            echo "NEG-TEST FAIL: $name: missing '$want' in:"; echo "$out" | head -3
            failures=$((failures + 1)); return
        fi
    fi
    echo "NEG-TEST OK: $name"
}

build_world() {   # build_world — creates $scratch/pkg wheel + $scratch/site install
    python3 - "$scratch" <<'PYEOF'
import base64, hashlib, os, sys, zipfile
scratch = sys.argv[1]
site = os.path.join(scratch, "site")
os.makedirs(os.path.join(site, "pkg"), exist_ok=True)
os.makedirs(os.path.join(site, "pkg-1.0.dist-info"), exist_ok=True)

files = {
    "pkg/__init__.py": b"x = 1\n",
    "pkg/core.so": b"\x7fELF fake-provider-bytes v1\n",
    "pkg-1.0.dist-info/METADATA": b"Metadata-Version: 2.1\nName: pkg\nVersion: 1.0\n",
}
for path, content in files.items():
    with open(os.path.join(site, path), "wb") as fh:
        fh.write(content)

def b64(data):
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()

record = [f"{p},sha256={b64(c)},{len(c)}" for p, c in files.items()]
record.append("pkg-1.0.dist-info/RECORD,,")
record_text = "\n".join(record) + "\n"
with zipfile.ZipFile(os.path.join(scratch, "pkg-1.0-py3-none-any.whl"), "w") as z:
    for path, content in files.items():
        z.writestr(path, content)
    z.writestr("pkg-1.0.dist-info/RECORD", record_text)
PYEOF
}

mutate() {   # mutate <what>
    case "$1" in
    installed)   printf 'tampered\n' > "$scratch/site/pkg/core.so" ;;
    record-hash) # rebuild the wheel with a wrong RECORD digest for core.so
        python3 - "$scratch" <<'PYEOF'
import base64, hashlib, os, sys, zipfile
scratch = sys.argv[1]
wheel = os.path.join(scratch, "pkg-1.0-py3-none-any.whl")
zin = zipfile.ZipFile(wheel)
items = {n: zin.read(n) for n in zin.namelist()}
zin.close()
rows = []
for line in items["pkg-1.0.dist-info/RECORD"].decode().splitlines():
    if line.startswith("pkg/core.so,"):
        line = "pkg/core.so,sha256=" + "A" * 43 + ",28"
    rows.append(line)
items["pkg-1.0.dist-info/RECORD"] = ("\n".join(rows) + "\n").encode()
with zipfile.ZipFile(wheel, "w") as z:
    for n, c in items.items():
        z.writestr(n, c)
PYEOF
        ;;
    unhashed)    # rebuild with an extra unhashed non-signature member
        python3 - "$scratch" <<'PYEOF'
import os, sys, zipfile
scratch = sys.argv[1]
wheel = os.path.join(scratch, "pkg-1.0-py3-none-any.whl")
zin = zipfile.ZipFile(wheel)
items = {n: zin.read(n) for n in zin.namelist()}
zin.close()
items["pkg/extra.py"] = b"y = 2\n"
rec = items["pkg-1.0.dist-info/RECORD"].decode().replace(
    "pkg-1.0.dist-info/RECORD,,", "pkg/extra.py,,\npkg-1.0.dist-info/RECORD,,")
items["pkg-1.0.dist-info/RECORD"] = rec.encode()
with zipfile.ZipFile(wheel, "w") as z:
    for n, c in items.items():
        z.writestr(n, c)
PYEOF
        printf 'y = 2\n' > "$scratch/site/pkg/extra.py" ;;
    metadata)    printf 'garbage\n' > "$scratch/site/pkg-1.0.dist-info/METADATA" ;;
    no-distinfo) rm -rf "$scratch/site/pkg-1.0.dist-info" ;;
    esac
}

# 1. Positive: exact install of the wheel passes.
build_world
check "byte-identical install passes" pass "byte-identical"

# 2. Installed bytes mutated: the binding must fail.
build_world; mutate installed
check "tampered installed file" fail "installed bytes != wheel bytes"

# 3. RECORD digest corrupted: wheel-internal integrity must fail.
build_world; mutate record-hash
check "corrupted RECORD digest" fail "RECORD digest mismatch"

# 4. A non-signature member without a digest fails closed.
build_world; mutate unhashed
check "unhashed non-signature member" fail "unhashed wheel member"

# 5. Installed METADATA must equal the wheel's.
build_world; mutate metadata
check "tampered installed METADATA" fail "METADATA != wheel METADATA"

# 6. Missing installed dist-info fails.
build_world; mutate no-distinfo
check "missing installed dist-info" fail "METADATA missing"

# 7. A payload member omitted from RECORD escapes no binding: the row
#    coverage check fails even with installed bytes intact.
build_world
python3 - "$scratch" <<'PYEOF'
import os, sys, zipfile
scratch = sys.argv[1]
wheel = os.path.join(scratch, "pkg-1.0-py3-none-any.whl")
zin = zipfile.ZipFile(wheel)
items = {n: zin.read(n) for n in zin.namelist()}
zin.close()
rows = [l for l in items["pkg-1.0.dist-info/RECORD"].decode().splitlines()
        if not l.startswith("pkg/core.so,")]
items["pkg-1.0.dist-info/RECORD"] = ("\n".join(rows) + "\n").encode()
with zipfile.ZipFile(wheel, "w") as z:
    for n, c in items.items():
        z.writestr(n, c)
PYEOF
check "member omitted from RECORD" fail "omitted from RECORD"

# 8. A RECORD carrying only its own self-row is vacuous evidence.
build_world
python3 - "$scratch" <<'PYEOF'
import os, sys, zipfile
scratch = sys.argv[1]
wheel = os.path.join(scratch, "pkg-1.0-py3-none-any.whl")
zin = zipfile.ZipFile(wheel)
items = {n: zin.read(n) for n in zin.namelist()}
zin.close()
items["pkg-1.0.dist-info/RECORD"] = b"pkg-1.0.dist-info/RECORD,,\n"
with zipfile.ZipFile(wheel, "w") as z:
    for n, c in items.items():
        z.writestr(n, c)
PYEOF
check "self-only RECORD is vacuous" fail "vacuous identity evidence"

if [ "$failures" -ne 0 ]; then
    echo "NEG-TESTS: $failures failure(s)" >&2
    exit 1
fi
echo "NEG-TESTS: all passed"

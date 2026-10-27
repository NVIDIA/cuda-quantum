#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Verify that an installed LLVM/MLIR prefix provides the shared runtime
# provider set that the CUDA-Q SDK contract requires:
#
#   libLLVM.so.<sover>                       upstream LLVM implementation
#   libMLIR.so.<sover>                       upstream MLIR components
#   libMLIRExecutionEngineShared.so.<sover>  execution engine (excluded from libMLIR)
#   libMLIR-C.so.<sover>                     full upstream MLIR C API
#
# Checks, for the pinned baseline:
#  1. presence of each provider under its canonical SONAME and link names,
#  2. the required dependency direction between providers (DT_NEEDED) — the
#     structural anti-duplication proof: a provider that secretly embeds the
#     implementation needs no DT_NEEDED on the real providers,
#  3. no strong duplicate llvm::/mlir:: definitions across providers
#     (weak/vague-linkage copies are expected and ignored),
#  4. export closure: every public defined symbol of the upstream CAPI
#     archives listed in MLIRConfig.cmake (MLIR_UPSTREAM_CAPI_LIBS) is
#     exported by libMLIR-C (catches macro-generated pass/dialect APIs),
#  5. stock llvm-config coherence for shared and static queries.
#
# Coverage labeling: the installed-prefix checks alone are PRELIMINARY —
# it cannot see statically embedded copies that --exclude-libs,ALL hides from
# the dynamic symbol table. The DT_NEEDED direction (2) and the libMLIR-C
# size gate are the structural proofs available post-strip. The mandatory
# build-stage evidence of spec section 3 lives in the --build-dir mode below
# (the providers' link inputs plus their unstripped full symbol tables),
# wired into the producer's build in this slice.
#
# Usage: check_mlir_providers.sh <llvm-install-prefix> [--build-dir <dir>]
#        check_mlir_providers.sh --load-digest <dso>
#
# --load-digest prints the loadable-content digest of one DSO (see the
# binding comment at view (c) below) for artifact records such as
# PROVIDER-AUDIT.txt.

set -euo pipefail

# sort/comm must use byte order; locale-dependent collation breaks comm -23/-12.
export LC_ALL=C

# Shared link-record provenance (edge variables, rspfiles, linker flags).
# shellcheck source=link_audit_lib.sh
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$script_dir/link_audit_lib.sh"

# Ownership namespace: mangled llvm::/mlir:: C++ names AND the unmangled
# upstream MLIR C API (mlirXxx). Plain C library names (zlib's adler32 et
# al.) are excluded on purpose.
ownership_symbols='^_Z(T[VIS])?N?K?[0-9]*4(llvm|mlir)|^_?mlir[A-Z]'

# Executable-content identity for a DSO, in two parts:
#
#  exec_digest: sha256 over the ELF header essentials (type, machine,
#  entry), every PT_LOAD program header's (vaddr, paddr, filesz, memsz,
#  flags), and every SHF_ALLOC section's (name, address, size, bytes) —
#  with exactly ONE normalization: inside .dynstr, the file's own
#  RUNPATH/RPATH string value is zeroed, because CMake's install editor
#  rewrites that one string in place and zero-pads the remainder (asserted
#  separately in view (c)). Everything else is bound: loader mapping
#  layout and permissions, function bodies, hidden and localized code,
#  data, every other .dynstr string (symbol and version names), NOBITS
#  section descriptors, and section layout. Strip removes only
#  non-allocated content, so unlike a .comment string or an exported-name
#  list, two different implementations cannot share this digest.
#
#  dyn_digest: the dynamic table's semantic content (every entry except
#  RUNPATH/RPATH): dependency direction, SONAME and flags are still bound.
exec_digest() {
    python3 - "$1" <<'PYEOF'
import hashlib, struct, sys

with open(sys.argv[1], "rb") as fh:
    data = fh.read()
if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
    sys.exit(1)   # ELF64 little-endian only

e_type, e_machine = struct.unpack_from("<HH", data, 0x10)
e_entry = struct.unpack_from("<Q", data, 0x18)[0]
e_phoff = struct.unpack_from("<Q", data, 0x20)[0]
e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
e_phentsize, e_phnum = struct.unpack_from("<HH", data, 0x36)
e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)

sections = []
for i in range(e_shnum):
    off = e_shoff + i * e_shentsize
    name_i, sh_type, sh_flags, sh_addr, sh_off, sh_size = struct.unpack_from("<IIQQQQ", data, off)
    sections.append((name_i, sh_type, sh_flags, sh_addr, sh_off, sh_size))
if not sections:
    sys.exit(1)
_, sht_type, _, _, sht_off, sht_size = sections[e_shstrndx]
shstrtab = data[sht_off:sht_off + sht_size]

def sec_name(name_i):
    end = shstrtab.index(b"\x00", name_i)
    return shstrtab[name_i:end].decode()

# Program headers: the loader's mapping semantics are bound — every
# program header's (type, flags, file offset, addresses, sizes, alignment),
# covering PT_LOAD layout/permissions and runtime headers like PT_TLS and
# GNU_RELRO. A permission, offset, layout or TLS change on an installed
# copy cannot pass even with identical section bytes. Strip and the
# declared RUNPATH rewrite do not touch these.
loads, dyn = [], None
h = hashlib.sha256()
h.update(f"EHDR {e_type} {e_machine} {e_entry:#x}\n".encode())
for i in range(e_phnum):
    off = e_phoff + i * e_phentsize
    (p_type, p_flags, p_off, p_vaddr, p_paddr,
     p_filesz, p_memsz, p_align) = struct.unpack_from("<IIQQQQQQ", data, off)
    h.update(f"PHDR {p_type:#x} {p_flags:#x} {p_off:#x} {p_vaddr:#x} {p_paddr:#x} {p_filesz:#x} {p_memsz:#x} {p_align:#x}\n".encode())
    if p_type == 1:
        loads.append((p_vaddr, p_off, p_filesz))
    elif p_type == 2:
        dyn = (p_off, p_filesz)
runpaths, strtab_addr = [], None
if dyn:
    for off in range(dyn[0], dyn[0] + dyn[1], 16):
        tag, val = struct.unpack_from("<qQ", data, off)
        if tag == 0:
            break
        if tag == 5:
            strtab_addr = val
        elif tag in (15, 29):
            runpaths.append(val)

def v2f(vaddr):
    for p_vaddr, p_off, p_filesz in loads:
        if p_vaddr <= vaddr < p_vaddr + p_filesz:
            return p_off + (vaddr - p_vaddr)
    return None

# RUNPATH/RPATH string RANGES (file offset, length) for zero
# normalization. CMake's install editor rewrites the value in place and
# zero-pads the remainder, so zeroing the value's region on both sides
# normalizes exactly the declared transformation and nothing else.
runpath_regions = []
if strtab_addr is not None:
    for val in runpaths:
        off = v2f(strtab_addr + val)
        if off is not None:
            end = data.index(b"\x00", off)
            runpath_regions.append((off, end + 1 - off))

alloc = 0
for name_i, sh_type, sh_flags, sh_addr, sh_off, sh_size in sections:
    if not (sh_flags & 0x2) or sh_type == 6:   # SHF_ALLOC, skip SHT_DYNAMIC
        continue
    alloc += 1
    name = sec_name(name_i)
    h.update(f"SEC {name} {sh_addr:#x} {sh_size:#x}\n".encode())
    if sh_type == 8:                            # NOBITS: descriptor only
        continue
    content = bytearray(data[sh_off:sh_off + sh_size])
    if name == ".dynstr":                     # zero only the declared
        for off, ln in runpath_regions:        # RUNPATH transformation
            if sh_off <= off < sh_off + sh_size:
                content[off - sh_off:off - sh_off + ln] = b"\x00" * ln
    h.update(bytes(content))
if not alloc:
    sys.exit(1)
print(h.hexdigest())
PYEOF
}

dyn_digest() {
    readelf -dW "$1" 2>/dev/null \
        | awk '/0x/ && !/RUNPATH|RPATH/ {sub(/^[ \t]+/, ""); print}' \
        | sha256sum | awk '{print $1}'
}

# Composite record token: "<exec-digest> <dyn-digest>". Empty (nonzero)
# only when the file yields no loadable section evidence at all.
load_digest() {
    local f="$1" e d
    e="$(exec_digest "$f")" || return 1
    d="$(dyn_digest "$f")"
    printf '%s %s\n' "$e" "$d"
}

if [ "${1:-}" = "--load-digest" ]; then
    if [ ! -f "${2:-}" ]; then
        echo "Usage: $0 --load-digest <dso>" >&2
        exit 2
    fi
    digest="$(load_digest "$2")" || true
    if [ -z "$digest" ]; then
        echo "check_mlir_providers: no loadable segments in $2" >&2
        exit 2
    fi
    echo "$digest"
    exit 0
fi

if [ $# -lt 1 ] || [ ! -d "$1" ]; then
    echo "Usage: $0 <llvm-install-prefix> [--build-dir <dir>]" >&2
    exit 2
fi
prefix="$(cd "$1" && pwd)"
build_dir=""
if [ "${2:-}" = "--build-dir" ]; then
    build_dir="$3"
    [ -d "$build_dir" ] || { echo "no such build dir: $build_dir" >&2; exit 2; }
fi

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
note() { echo "check_mlir_providers: $*"; }

# Shared libraries carry a SONAME of the form lib<name>.so.<sover> where
# <sover> is <major>.<minor> derived from the installed llvm-config.
llvm_config="$prefix/bin/llvm-config"
if [ ! -x "$llvm_config" ]; then
    echo "check_mlir_providers: no executable $llvm_config" >&2
    exit 2
fi
version="$("$llvm_config" --version)"
sover="${version%.*}"
note "prefix=$prefix version=$version sover=$sover"

realpath_of() { readlink -f "$prefix/lib/$1"; }
soname_of() { readelf -d "$1" | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p'; }
needed_of() { readelf -d "$1" | sed -n 's/.*NEEDED.*Shared library: \[\(.*\)\]/\1/p'; }

# 1./2. Provider presence, canonical SONAMEs, link names, dependency direction.
check_provider() {
    local base="$1"; shift
    local versioned="lib${base}.so.${sover}"
    local path="$prefix/lib/$versioned"
    [ -e "$path" ] || { fail "missing provider $versioned"; return; }
    local real soname
    real="$(realpath_of "$versioned")"
    soname="$(soname_of "$real")"
    [ "$soname" = "$versioned" ] || fail "$versioned has SONAME '$soname'"
    [ -e "$prefix/lib/lib${base}.so" ] || fail "missing link name lib${base}.so"
    local dep
    for dep in "$@"; do
        needed_of "$real" | grep -qx "lib${dep}.so.${sover}" \
            || fail "lib${base} does not depend on lib${dep}.so.${sover}"
    done
}

check_provider LLVM
check_provider MLIR LLVM
check_provider MLIRExecutionEngineShared MLIR LLVM
check_provider MLIR-C MLIR MLIRExecutionEngineShared
# Upstream compatibility link name for the LLVM dylib.
[ -e "$prefix/lib/libLLVM-${sover%%.*}.so" ] \
    || fail "missing compatibility name libLLVM-${sover%%.*}.so"

# 3. No strong duplicate llvm::/mlir:: definitions across providers.
strong_mlir_defs() {
    nm -D --defined-only --extern-only "$1" 2>/dev/null \
        | awk '$2 ~ /^[TDBRiSG]$/ {print $NF}' \
        | sed 's/@.*$//' \
        | grep -E '^_Z(T[VIS])?N?K?[0-9]*4(llvm|mlir)' || true
}
providers=(LLVM MLIR MLIRExecutionEngineShared MLIR-C)
for ((i = 0; i < ${#providers[@]}; i++)); do
    defs_i="$(strong_mlir_defs "$(realpath_of "lib${providers[$i]}.so.${sover}")" | sort -u)"
    for ((j = i + 1; j < ${#providers[@]}; j++)); do
        dup="$(comm -12 <(echo "$defs_i") \
            <(strong_mlir_defs "$(realpath_of "lib${providers[$j]}.so.${sover}")" | sort -u))"
        [ -z "$dup" ] || fail "strong duplicate definitions in lib${providers[$i]} and lib${providers[$j]}:"$'\n'"$(echo "$dup" | head -5)"
    done
done
# libMLIR-C must not carry its own strong copy of the C++ implementation.
mlir_c_strong="$(strong_mlir_defs "$(realpath_of "libMLIR-C.so.${sover}")")"
[ -z "$mlir_c_strong" ] || fail "libMLIR-C defines strong llvm::/mlir:: symbols:"$'\n'"$(echo "$mlir_c_strong" | head -5)"

# 4b. Build-stage evidence (--build-dir): the mandatory spec-section-3 gate.
# Two independent fail-closed views:
#   (a) link-input provenance. A component archive on a provider's link line
#       is direct evidence of static embedding. Ninja separates explicit
#       inputs, '|' implicit (still real) inputs and '||' order-only
#       dependencies: everything up to '||' is scanned. Missing evidence is
#       fatal (no build.ninja, no matching edge, unresolved $VARIABLE or
#       .rsp references). libLLVM legitimately aggregates its own LLVM
#       component archives (llvm-shlib); MLIR aggregates link obj.* objects.
#   (b) pre-strip symbol audit of the unstripped build-tree providers.
#       readelf -sW columns are Num Value Size Type Bind Vis Ndx Name
#       ($4/$5/$6 select Type/Bind/Vis). readelf failures, stripped tables
#       and vacuous parses are fatal; interposable (GLOBAL/DEFAULT)
#       definitions shared across providers and implausible definition
#       populations in the thin providers (an embedded --exclude-libs copy
#       localizes tens of thousands of definitions where the clean wrappers
#       carry < 200) fail the gate.
# Baselines measured on the pinned 22.1.4 producer tree: libMLIR 247982 and
# libLLVM 58714 GLOBAL/DEFAULT definitions (parser non-vacuity floors 10000);
# libMLIR-C 148 and libMLIRExecutionEngineShared 71 total definitions
# (population limits 1000/500); cross-provider GLOBAL/DEFAULT duplicates 0.
if [ -n "$build_dir" ]; then
    ninja="$build_dir/build.ninja"
    ninja_dir="$(cd "$build_dir" && pwd -P)"
    [ -f "$ninja" ] || fail "no build.ninja under $build_dir; link-input evidence missing"
    # Link-input provenance via the shared complete-record mechanism
    # (scripts/link_audit_lib.sh): every edge producing a provider,
    # including its variable block (LINK_LIBRARIES et al.), rspfile
    # contents, and inputs conveyed through -Wl,/-L/-l linker forms.
    # Missing/unresolved evidence is fatal. libLLVM legitimately aggregates
    # its own LLVM component archives (llvm-shlib); every other provider
    # must resolve components from the provider DSOs, and MLIR aggregates
    # link their designated obj.* objects.
    declare -A provider_inputs
    for provider in "${providers[@]}"; do
        out="lib${provider}.so.${sover}"
        tokens="$(la_all_edge_tokens "$ninja" "$out" "$ninja_dir" || true)"
        [ -n "$tokens" ] || fail "no link edge for $out in $ninja"
        [ -n "$tokens" ] || continue
        link_dirs=()
        link_mode=""
        mode_stack=()
        edge_tokens=()
        implicit_dirs=()
        la_collect_implicit_lib_dirs implicit_dirs "$build_dir"
        # GNU applies every -L to every -l: precollect the library path
        # before classification (LINK_PATH may follow LINK_LIBRARIES).
        while IFS= read -r tok; do
            edge_tokens+=("$tok")
            for sub in $(la_split_wl "$tok"); do
                case "$sub" in
                -L*) link_dirs+=("${sub#-L}") ;;
                esac
            done
        done <<< "$tokens"
        for tok in "${edge_tokens[@]+"${edge_tokens[@]}"}"; do
            case "$tok" in
            UNRESOLVED:*)
                fail "unresolved link-input evidence (${tok#UNRESOLVED:}) in $out edge"
                continue ;;
            esac
            if la_unresolved "$tok"; then
                fail "unresolved link-input evidence ($tok) in $out edge"
                continue
            fi
            for sub in $(la_split_wl "$tok"); do
                case "$sub" in
                -L*) continue ;;
                -Bstatic|-Bstatic=true|-Bstatic,|-Wl,-Bstatic) link_mode="static" ; continue ;;
                -Bdynamic|-Bdynamic=true) link_mode="" ; continue ;;
                --push-state) mode_stack+=("$link_mode") ; continue ;;
                --pop-state)
                    if [ "${#mode_stack[@]}" -gt 0 ]; then
                        link_mode="${mode_stack[-1]}"
                        unset 'mode_stack[-1]'
                    fi
                    continue ;;
                esac
                case "$sub" in
                -l*)
                    resolved="$(la_resolve_library "$sub" "$link_mode" \
                        "${link_dirs[@]}" "$ninja_dir" \
                        "$prefix/lib" "${implicit_dirs[@]}" 2>/dev/null || true)"
                    if [ -z "$resolved" ]; then
                        fail "unresolved link-input evidence ($sub) in $out edge"
                        continue
                    fi
                    sub="$resolved" ;;
                esac
                case "$(basename "$sub")" in
                libMLIR*.a)
                    if [ "$provider" = LLVM ]; then
                        fail "libLLVM's link inputs contain MLIR component archives: $sub"
                    else
                        fail "lib${provider}'s link inputs contain component archives: $sub"
                    fi ;;
                libLLVM*.a)
                    [ "$provider" = LLVM ] || fail "lib${provider}'s link inputs contain component archives: $sub" ;;
                esac
                # Referenced archive/object inputs: existence is required
                # (an input that cannot be inspected is missing evidence),
                # and the existing ones are remembered for the by-content
                # ownership proof below.
                case "$sub" in
                *.a|*.o)
                    inp="$sub"
                    case "$inp" in
                    /*) : ;;
                    *) inp="$ninja_dir/$inp" ;;
                    esac
                    if [ ! -e "$inp" ]; then
                        fail "missing link-input evidence ($sub) in $out edge"
                    else
                        case " ${provider_inputs[$provider]:-} " in
                        *" $inp "*) ;;
                        *) provider_inputs[$provider]="${provider_inputs[$provider]:-} $inp" ;;
                        esac
                    fi ;;
                esac
            done
        done <<< "$tokens"
    done

    # Pre-strip evidence: build-tree providers hold their full symbol tables.
    prestripped_defs() {   # all defined llvm::/mlir:: FUNC/OBJECT names
        readelf -sW "$1" | awk '
            $7 != "UND" && ($4 == "FUNC" || $4 == "OBJECT") \
            && $8 ~ /^_Z(T[VIS])?N?K?[0-9]*4(llvm|mlir)/ {sub(/@.*/, "", $8); print $8}'
    }
    prestripped_interposable() {   # GLOBAL/DEFAULT (interposable) definitions
        readelf -sW "$1" | awk '
            $7 != "UND" && $5 == "GLOBAL" && $6 == "DEFAULT" \
            && ($4 == "FUNC" || $4 == "OBJECT") \
            && $8 ~ /^_Z(T[VIS])?N?K?[0-9]*4(llvm|mlir)/ {sub(/@.*/, "", $8); print $8}'
    }
    declare -A provider_defs provider_interposable provider_path
    for provider in "${providers[@]}"; do
        p="$(find "$build_dir" -name "lib${provider}.so.${sover}" | head -1)"
        [ -f "$p" ] || { fail "build tree lacks unstripped lib${provider}.so.${sover}"; continue; }
        provider_path[$provider]="$p"
        readelf -sW "$p" >/dev/null 2>&1 \
            || { fail "readelf cannot read the unstripped lib${provider}.so.${sover}"; continue; }
        readelf -SW "$p" 2>/dev/null | grep -q '\.symtab' \
            || { fail "unstripped lib${provider}.so.${sover} lacks a full symbol table"; continue; }
        provider_defs[$provider]="$(prestripped_defs "$p" | sort -u)"
        provider_interposable[$provider]="$(prestripped_interposable "$p" | sort -u)"
    done
    for owner in MLIR LLVM; do
        n=$(echo "${provider_defs[$owner]:-}" | grep -c . || true)
        [ "$n" -ge 10000 ] || fail "pre-strip parse of lib${owner} produced only $n definitions; audit evidence is vacuous"
    done
    for ((i = 0; i < ${#providers[@]}; i++)); do
        for ((j = i + 1; j < ${#providers[@]}; j++)); do
            dup="$(comm -12 <(echo "${provider_interposable[${providers[$i]}]:-}") \
                <(echo "${provider_interposable[${providers[$j]}]:-}") || true)"
            [ -z "$dup" ] || fail "pre-strip interposable definitions shared by lib${providers[$i]} and lib${providers[$j]}:"$'\n'"$(echo "$dup" | head -5)"
        done
    done
    for thin in MLIR-C:1000 MLIRExecutionEngineShared:500; do
        provider="${thin%%:*}"; limit="${thin##*:}"
        n=$(echo "${provider_defs[$provider]:-}" | grep -c . || true)
        [ "$n" -le "$limit" ] || fail "lib${provider} defines $n llvm::/mlir:: symbols (limit $limit); an embedded static implementation is localized in the full symbol table"
    done

    # Ownership by content on the edge inputs: any archive or object a
    # provider links whose strong ownership definitions duplicate ANOTHER
    # provider's exports is an embedded copy — invisible to the dynamic
    # scans and far below the population caps. (The provider's own exports
    # are exempt: its own objects and, for libLLVM, its own component
    # archives legitimately define those names. Symbol-table views cannot
    # make this distinction for scattered template instantiations, which is
    # why the proof runs on the link inputs, not the tables.)
    for ((i = 0; i < ${#providers[@]}; i++)); do
        owner="${providers[$i]}"
        others_exports="$(for ((j = 0; j < ${#providers[@]}; j++)); do
            [ "$i" = "$j" ] && continue
            nm -D --defined-only "$(realpath_of "lib${providers[$j]}.so.${sover}")" 2>/dev/null
        done | awk '$2 ~ /^[TDBRiSG]$/ {print $NF}' | sed 's/@.*$//' \
            | { grep -E "$ownership_symbols" || true; } | sort -u)"
        [ -n "$others_exports" ] || fail "provider export view is vacuous for $owner"
        # One batched nm -A pass over this provider's remembered inputs;
        # the inspection itself is checked (a failed nm is never a pass).
        if [ -n "${provider_inputs[$owner]:-}" ]; then
            nm_rc=0
            nm_batch="$(nm -A --defined-only ${provider_inputs[$owner]:-} 2>&1)" || nm_rc=$?
            [ "$nm_rc" -eq 0 ] || fail "could not inspect $owner's link inputs (nm rc=$nm_rc)"
        else
            nm_batch=""
        fi
        while read -r dup_file dup_name; do
            [ -n "$dup_file" ] || continue
            fail "lib${owner} links an input embedding another provider's" \
                 "implementation: $dup_file (defines $dup_name, exported by" \
                 "another provider); a small static copy was embedded" \
                 "instead of resolving from the providers"
        done < <(printf '%s\n' "$nm_batch" \
            | awk -v re="$ownership_symbols" '
                NF < 3 { next }
                {
                    name = $NF; type = $(NF-1)
                    if (type !~ /^[TDBR]$/) next
                    file = $1; sub(/:[0-9a-fA-F]*$/, "", file)
                    sub(/@.*/, "", name)
                    if (name ~ re) print file, name
                }' \
            | sort -u \
            | awk 'NR == FNR { prov[$1] = 1; next }
                   ($2 in prov) && !seen[$1]++ { print $1, $2 }' \
                <(echo "$others_exports") -)
    done

    # (c) Build→install executable-content binding. Matching .comment
    # strings or exported-name lists do not bind implementation content:
    # different function bodies and hidden embedded code can share both.
    # The unstripped, link-audited build-tree provider and the installed
    # (distribution-stripped) copy must therefore agree on (i) every
    # allocated section's content and (ii) the dynamic table minus
    # RUNPATH/RPATH. The install's one declared transformation — dropping
    # the build tree's absolute RUNPATH entries — is asserted by rule: the
    # installed RUNPATH must equal the build-tree's minus its absolute
    # entries, and must not retain any build-machine path. This closes the
    # build-tree → install-prefix leg of the audit chain; the install →
# wheel leg is byte-identity (sha256) in the artifact record. Record
    # generation: quote these BINDING lines plus --load-digest output per
    # provider.
    runpath_of() { readelf -dW "$1" 2>/dev/null \
        | sed -n 's/.*\(RPATH\|RUNPATH\).*\[\(.*\)\]/\2/p'; }
    nonbuild_runpath() {  # drop absolute (build-machine) entries + empties
        printf '%s' "$1" | tr ':' '\n' | sed '/^$/d' | grep -v '^/' \
            | paste -sd: -
    }
    for provider in "${providers[@]}"; do
        p="${provider_path[$provider]:-}"
        [ -n "$p" ] && [ -f "$p" ] || continue   # missing evidence already failed
        inst="$(realpath_of "lib${provider}.so.${sover}")"
        digest_build="$(load_digest "$p" || true)"
        digest_inst="$(load_digest "$inst" || true)"
        if [ -z "$digest_build" ] || [ -z "$digest_inst" ] \
            || [ "$digest_build" != "$digest_inst" ]; then
            fail "build→install content mismatch for lib${provider}.so.${sover}:" \
                 "build-tree digest (${digest_build:-none}) != installed" \
                 "(${digest_inst:-none}); the pre-strip audited bytes are not" \
                 "the shipped bytes"
            continue
        fi
        rp_build="$(runpath_of "$p")"
        rp_inst="$(runpath_of "$inst")"
        if [ "$(nonbuild_runpath "$rp_build")" != "$(nonbuild_runpath "$rp_inst")" ] \
            || [ "$rp_inst" != "$(nonbuild_runpath "$rp_inst")" ]; then
            fail "build→install RUNPATH transformation off-contract for" \
                 "lib${provider}.so.${sover}: build-tree '$rp_build'," \
                 "installed '$rp_inst' (only dropping build-machine entries" \
                 "is allowed; none may remain)"
            continue
        fi
        note "BINDING lib${provider}.so.${sover}: executable content and" \
             "dynamic table identical build-tree ↔ install" \
             "(${digest_inst% *}); RUNPATH '$rp_build' → '$rp_inst'"
    done
fi

# 4a. Structural size evidence: libMLIR-C wraps MLIR's C API in a thin layer;
# an embedded implementation would make it a full copy of libMLIR. Reject if
# libMLIR-C exceeds a fifteenth of libMLIR (measured: 0.8MB vs 127MB).
size_of() { stat -c %s "$1"; }
mlir_size=$(size_of "$(realpath_of "libMLIR.so.${sover}")")
mlir_c_size=$(size_of "$(realpath_of "libMLIR-C.so.${sover}")")
[ $((mlir_c_size * 15)) -lt "$mlir_size" ] ||
    fail "libMLIR-C (${mlir_c_size}B) is suspiciously large next to libMLIR (${mlir_size}B); a hidden static copy is likely embedded"

# 4. Export closure: public defined symbols of the pinned CAPI archives
# must be exported by libMLIR-C. Reading the archives (not headers) also
# covers macro-generated pass/dialect registration functions.
# The export closure fails closed: a missing MLIRConfig.cmake, an empty
# MLIR_UPSTREAM_CAPI_LIBS, or an unreadable archive is a failure, never a
# silent pass.
mlir_config="$prefix/lib/cmake/mlir/MLIRConfig.cmake"
[ -f "$mlir_config" ] || fail "missing $mlir_config (devel side of the prefix is required)"
if [ -f "$mlir_config" ]; then
    capi_libs=$(sed -n 's/.*MLIR_UPSTREAM_CAPI_LIBS "\(.*\)")/\1/p' "$mlir_config" | tr ';' ' ')
    [ -n "$capi_libs" ] || fail "MLIRConfig.cmake lists no MLIR_UPSTREAM_CAPI_LIBS"
    capi_count=0
    for lib in $capi_libs; do capi_count=$((capi_count + 1)); done
    [ "$capi_count" -ge 50 ] || fail "only $capi_count upstream CAPI libraries listed (expected 58)"
    capi_exports="$(mktemp)"
    missing="$(mktemp)"
    trap 'rm -f "$capi_exports" "$missing"' EXIT
    for lib in $capi_libs; do
        archive="$prefix/lib/lib${lib}.a"
        [ -f "$archive" ] || { fail "missing CAPI archive lib${lib}.a (listed in MLIRConfig.cmake)"; continue; }
        # C API symbols only: C++ (mangled) symbols in the archives are
        # resolved from the shared providers, not re-exported by libMLIR-C.
        if ! nm -g --defined-only "$archive" > /dev/null 2>&1; then
            fail "could not read symbols from lib${lib}.a"
            continue
        fi
        nm -g --defined-only "$archive" 2>/dev/null \
            | awk '$2 ~ /^[TDBRW]$/ {print $NF}' | grep -v '^_Z' >> "$capi_exports" || true
    done
    sort -u "$capi_exports" -o "$capi_exports"
    [ -s "$capi_exports" ] || fail "collected no public symbols from the upstream CAPI archives"
    if ! nm -D --defined-only "$(realpath_of "libMLIR-C.so.${sover}")" 2>/dev/null \
        | awk '$2 ~ /^[TDBRiSGW]$/ {print $NF}' | sort -u > "$missing.tmp"; then
        fail "could not read dynamic symbols of libMLIR-C.so.${sover}"
    fi
    comm -23 "$capi_exports" "$missing.tmp" > "$missing"
    rm -f "$missing.tmp"
    if [ -s "$missing" ]; then
        fail "libMLIR-C does not export $(wc -l < "$missing") symbols from the upstream CAPI archives, e.g.:"$'\n'"$(head -5 "$missing")"
    else
        note "CAPI export closure holds ($(wc -l < "$capi_exports") symbols)"
    fi
fi

# 5. Stock llvm-config coherence.
[ -d "$("$llvm_config" --includedir)" ] || fail "llvm-config --includedir does not exist"
[ -d "$("$llvm_config" --libdir)" ] || fail "llvm-config --libdir does not exist"
shared_names="$("$llvm_config" --link-shared --libnames)"
[ "$shared_names" = "libLLVM-${sover%%.*}.so" ] \
    || fail "llvm-config --link-shared --libnames returned '$shared_names', expected 'libLLVM-${sover%%.*}.so'"
static_names="$("$llvm_config" --link-static --libnames)"
echo "$static_names" | grep -q "libLLVMSupport.a" \
    || fail "llvm-config --link-static --libnames lost static archives: '$static_names'"

if [ "$failures" -ne 0 ]; then
    echo "check_mlir_providers: $failures check(s) failed for $prefix" >&2
    exit 1
fi
note "all checks passed for $prefix"

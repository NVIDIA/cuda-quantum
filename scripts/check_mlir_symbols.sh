#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Verify that a CUDA-Q consumer resolves MLIR/LLVM from its providers only.
#
# Phase 1: every llvm::/mlir:: symbol the consumer leaves undefined must be
# exported by the provider set: libcudaqMLIR plus any extra providers. With
# shared upstream providers the set is libcudaqMLIR, libMLIR, libMLIR-C,
# libMLIRExecutionEngineShared and libLLVM; without them, libcudaqMLIR alone
# (plus any extra providers).
#
# Phase 2: the consumer must not define any strong symbol that any provider
# already exports. Weak/vague-linkage copies from non-bundled static archives
# are expected and ignored.
#
# Phase 3: pre-strip hidden-copy audit (Linux). A component archive embedded
# behind --exclude-libs,ALL is invisible to the dynamic scans above: its
# definitions are local, its references resolve internally. The full symbol
# table of the just-linked (unstripped) consumer exposes the population. The
# check fails closed: a stripped consumer or an unreadable readelf view is
# missing build-stage evidence, not a pass. Clean baselines measured on the
# pinned 22.1.4 stack: libcudaqMLIR carries ~7.3k internalized template copies,
# the CAPI aggregate and Python extensions carry < 10. An embedded provider
# archive localizes tens of thousands of definitions.
#
# Additional providers may be named after libcudaqMLIR. A downstream project's
# own common CAPI library is the usual case: template instantiations over
# downstream types are mangled into namespace mlir (e.g.
# mlir::detail::TypeIDResolver<my::Dialect>::id), so they match the MLIR symbol
# pattern below even though only the downstream library can ever define them.
#
# Usage: check_mlir_symbols.sh <consumer> <libcudaqMLIR> [extra-provider...]

set -euo pipefail

# sort/comm must use byte order; locale-dependent collation breaks comm -23/-12.
export LC_ALL=C

# Shared link-record provenance (edge variables, rspfiles, linker flags).
# Sits next to this script both in-tree and installed (lib/cmake/cudaq).
# shellcheck source=link_audit_lib.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/link_audit_lib.sh"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <consumer-library> <libcudaqMLIR> [extra-provider...]" >&2
    exit 2
fi

consumer="$1"
provider="$2"
shift 2
extra_providers=("$@")

for f in "$consumer" "$provider" "${extra_providers[@]+"${extra_providers[@]}"}"; do
    if [ ! -f "$f" ]; then
        echo "check_mlir_symbols: no such file: $f" >&2
        exit 2
    fi
done

# Itanium-mangled names whose first nested component is llvm:: or mlir::,
# including vtables/typeinfo (_ZTV/_ZTI/_ZTS), plus the unmangled upstream C
# API names (mlirXxx). Mach-O prefixes an underscore.
mlir_symbols='^(_?_Z(T[VIS])?N?K?[0-9]*4(llvm|mlir)|_?mlir[A-Z])'

# mlir::python:: and the mlirPython* C helpers are upstream MLIR's Python
# binding internals. Those live in their own libMLIRPythonSupport DSO, which
# the extensions link directly, and are deliberately not part of the provider
# set.
not_provided_by_cudaqmlir='^(_?_Z(T[VIS])?N?K?[0-9]*4mlir6python|_?mlirPython)'

NM="${NM:-nm}"
if ! command -v "$NM" >/dev/null 2>&1; then
    echo "check_mlir_symbols: '$NM' not found; set NM to an nm or llvm-nm" >&2
    exit 2
fi

# Version-scripted providers (e.g. libLLVM with @LLVM_22.1 tags) report the
# version suffix on both undefined and defined references; strip it so names
# compare equal regardless of which side carries the tag.
if [ "$(uname)" = "Darwin" ]; then
    undefined_of() { "$NM" -u "$1" | sed 's/@.*$//'; }
    defined_of() { "$NM" -gU "$1" | awk '{print $NF}' | sed 's/@.*$//'; }
    strong_exports_of() { "$NM" -gU -m "$1" | awk '!/ weak external /{print $NF}' | sed 's/@.*$//'; }
else
    undefined_of() { "$NM" -D --undefined-only "$1" | awk '{print $NF}' | sed 's/@.*$//'; }
    defined_of() { "$NM" -D --defined-only "$1" | awk '{print $NF}' | sed 's/@.*$//'; }
    strong_exports_of() {
        "$NM" -D --defined-only --extern-only "$1" | awk '$2 ~ /^[TDBRiSG]$/ {print $NF}' | sed 's/@.*$//'
    }
fi

provider_names=$(for p in "$provider" "${extra_providers[@]+"${extra_providers[@]}"}"; do
    basename "$p"
done | paste -sd' ' -)

exit_code=0

# ---- Phase 1: undefined MLIR/LLVM symbols must be provided ---------------- #
undefined=$(undefined_of "$consumer" \
    | grep -E "$mlir_symbols" \
    | grep -Ev "$not_provided_by_cudaqmlir" \
    | sort -u || true)

if [ -n "$undefined" ]; then
    defined=$({ defined_of "$provider"
                for extra in "${extra_providers[@]+"${extra_providers[@]}"}"; do
                    defined_of "$extra"
                done
              } | sort -u)
    missing=$(comm -23 <(echo "$undefined") <(echo "$defined"))

    if [ -n "$missing" ]; then
        count=$(echo "$missing" | wc -l | tr -d ' ')
        cat >&2 <<EOF

================================================================================
$(basename "$consumer") references $count MLIR/LLVM symbol(s) that
$provider_names does not export:

$(echo "$missing" | head -20 | sed 's/^/  /')
$([ "$count" -gt 20 ] && echo "  ... and $((count - 20)) more")

A library you require may be missing in cmake/modules/mlir-bundled-libs.txt.
To find that library:

  for a in \$LLVM_INSTALL_PREFIX/lib/lib{MLIR,LLVM}*.a; do
    nm --defined-only "\$a" 2>/dev/null | grep -q '$(echo "$missing" | head -1)' \\
      && echo "\$a"
  done
================================================================================
EOF
        exit_code=1
    fi
fi

# ---- Phase 2: no duplicate strong definitions of provider exports ------- #
owned=$({ strong_exports_of "$provider"
            for extra in "${extra_providers[@]+"${extra_providers[@]}"}"; do
                strong_exports_of "$extra"
            done
          } | sort -u)
consumer_defs=$(strong_exports_of "$consumer" | sort -u)
duplicates=$(comm -12 <(echo "$consumer_defs") <(echo "$owned") || true)

if [ -n "$duplicates" ]; then
    count=$(echo "$duplicates" | wc -l | tr -d ' ')
    cat >&2 <<EOF

================================================================================
$(basename "$consumer") defines $count strong symbol(s) already exported by
one of the providers ($provider_names):

$(echo "$duplicates" | head -20 | sed 's/^/  /')
$([ "$count" -gt 20 ] && echo "  ... and $((count - 20)) more")

Static MLIR/LLVM archives were linked into this library instead of resolving
from the providers. Link cudaq::cudaqMLIR (or cudaq::MLIR) and ensure its
INTERFACE_LINK_OPTIONS place libcudaqMLIR before any bundled component archive.
================================================================================
EOF
    exit_code=1
fi

# ---- Phase 3: hidden-copy audit on the build-stage consumer --------------- #
# A component archive embedded behind --exclude-libs,ALL is invisible to the
# dynamic scans above: its definitions are local, its references resolve
# internally. Two views close that gap, both fail-closed:
#
#  (a) Ownership proof on the unstripped symbol table. The consumer's
#      LOCAL-bind llvm::/mlir:: definitions must not duplicate ANY name a
#      provider defines (exported dynsym view on installed providers).
#      Measured clean baseline: ZERO duplicates across 6386 LOCAL names on
#      libcudaqMLIR — internalized template copies never collide with
#      provider definitions, so even a small component (libMLIRSupport.a
#      carries ~300 named definitions incl. mlir::openInputFile, exported by
#      libMLIR) is caught exactly, with no population-threshold games. A
#      >20000 LOCAL population stays as a coarse sanity check.
#  (b) Link-input provenance from the build's build.ninja (CUDAQ_BUILD_NINJA)
#      for every consumer — stripped or not (nanobind strips extension DSOs
#      at link by design, -Wl,-s in nanobind-config.cmake:337). The complete
#      link record is scanned, not just the dependency edge: the build
#      statement's explicit and '|' implicit inputs ('||' order-only
#      excluded), the edge's variable block (LINK_LIBRARIES carries most
#      link inputs on CMake/Ninja layouts), the contents of any rspfile the
#      edge references, and inputs conveyed through linker flags (-Wl,...
#      arguments are split; -L/-l forms are resolved through the edge's
#      library path). Missing evidence is a failure, never a pass: no edge,
#      a named-but-absent build.ninja, a stripped consumer with no Ninja
#      evidence at all, unresolved $VARIABLE/.rsp tokens ('$$' is Ninja's
#      literal-$ escape and unescapes first), unresolvable -l names, and
#      inputs that do not exist on disk all fail. Ownership is by content,
#      not by where a path sits: provider-directory components are forbidden
#      wherever they live (a provider root nested under the build tree is
#      still a provider root), and any link-input object or archive whose
#      strong ownership definitions duplicate provider exports is an
#      embedded implementation even when copied into the consumer's own
#      tree. Downstream-owned inputs pass. With the shared providers the
#      producer links support code from libLLVM (see
#      tpls/customizations/llvm/mlir-python-shared-llvm.diff), so no
#      archive exemption exists; the legacy static/monolithic layout (no
#      upstream provider DSOs) keeps the historical Support/Demangle
#      utility-archive skip for upstream's python machinery.
if [ "$(uname)" = "Darwin" ]; then
    exit "$exit_code"
fi

readelf_local_defs() {   # LOCAL-bind defined ownership names
    readelf -sW "$1" 2>/dev/null | awk '
        $7 != "UND" && $5 == "LOCAL" && ($4 == "FUNC" || $4 == "OBJECT") &&
        $8 ~ /^_Z(T[VIS])?N?K?[0-9]*4(llvm|mlir)|^_?mlir[A-Z]/ \
            {sub(/@.*/, "", $8); print $8}'
}

# Ownership namespace: mangled llvm::/mlir:: C++ names AND the unmangled
# upstream MLIR C API (mlirXxx) — a copied MLIRCAPI archive must not slip
# the ownership view. Plain C library names (e.g. libLLVM's bundled zlib
# adler32) are excluded on purpose: they would false-match third-party
# static archives the consumer legitimately links (libz.a).
ownership_symbols='^_Z(T[VIS])?N?K?[0-9]*4(llvm|mlir)|^_?mlir[A-Z]'

# Exported provider names, computed on first use: the ownership view for the
# pre-strip table (a) and the by-content archive classification (b). STRONG
# exports only: a weak/vague-linkage duplicate (vtable, inline template
# instantiation) is the same code emitted per-TU by the compiler, not a
# second implementation — phase 2 already ignores those in the dynamic view;
# a localized weak copy typical of hidden-visibility aggregate builds is no
# different. The small-component attacks this view exists to catch duplicate
# STRONG provider definitions (openInputFile etc.).
provider_exported=""
provider_exports() {
    [ -n "$provider_exported" ] && return
    provider_exported="$({ strong_exports_of "$provider"
        for extra in "${extra_providers[@]+"${extra_providers[@]}"}"; do
            strong_exports_of "$extra"
        done
    } | { grep -E "$ownership_symbols" || true; } | sort -u)"
    if [ -z "$provider_exported" ]; then
        echo "check_mlir_symbols: providers export no strong llvm::/mlir:: names;" \
             "ownership evidence is vacuous" >&2
        exit 1
    fi
}

have_symtab=false
if readelf -sW "$consumer" >/dev/null 2>&1 \
    && readelf -SW "$consumer" 2>/dev/null | grep -q '\.symtab'; then
    have_symtab=true
    # (a) Ownership proof: any LOCAL definition duplicating a provider's
    # STRONG exported name is a second implementation, hidden from the
    # dynamic view. (Localized vague-linkage copies — hidden-visibility
    # aggregates carry these legitimately — collide only against weak
    # provider exports, which stay out of the comparison.)
    provider_exports
    dups="$(comm -12 <(readelf_local_defs "$consumer" | sort -u) \
        <(echo "$provider_exported") || true)"
    if [ -n "$dups" ]; then
        count=$(echo "$dups" | wc -l | tr -d ' ')
        cat >&2 <<EOF

================================================================================
$(basename "$consumer") carries $count LOCAL llvm::/mlir:: definition(s) that
duplicate names exported by the providers ($provider_names):

$(echo "$dups" | head -20 | sed 's/^/  /')

A static MLIR/LLVM implementation (even a small component archive) was
embedded and localized instead of resolving from the providers. The clean
baseline for this comparison is zero.
================================================================================
EOF
        exit_code=1
    fi
    # Coarse sanity check on the localized population.
    hidden_limit=20000
    local_defs=$(readelf_local_defs "$consumer" | wc -l | tr -d ' ')
    if [ "$local_defs" -gt "$hidden_limit" ]; then
        echo "check_mlir_symbols: $(basename "$consumer") carries $local_defs" \
             "LOCAL llvm::/mlir:: definitions (limit $hidden_limit)" >&2
        exit_code=1
    fi
fi

# (b) Link-input provenance (every consumer when Ninja evidence exists).
ninja="${CUDAQ_BUILD_NINJA:-}"
if [ -n "$ninja" ] && [ ! -f "$ninja" ]; then
    echo "check_mlir_symbols: CUDAQ_BUILD_NINJA=$ninja does not exist;" \
         "link-input evidence missing" >&2
    exit_code=1
elif [ -n "$ninja" ]; then
    out_name="$(basename "$consumer")"
    ninja_dir="$(cd "$(dirname "$ninja")" && pwd -P)"
    # Canonical upstream provider directories; membership is tested by
    # canonical path so a provider root nested under the build tree is
    # still caught.
    upstream_dirs=""
    for p in "$provider" "${extra_providers[@]+"${extra_providers[@]}"}"; do
        case "$(basename "$p")" in
        libMLIR*.so*|libLLVM*.so*)
            upstream_dirs="$upstream_dirs $(realpath "$(dirname "$p")")" ;;
        esac
    done
    # Every matching edge block (basenames compared literally), classified
    # with the shared provenance mechanism (scripts/link_audit_lib.sh).
    blocks="$(la_edge_blocks "$ninja" "$out_name")"
    if [ -z "$blocks" ]; then
        echo "check_mlir_symbols: no link edge for $out_name in $ninja;" \
             "link-input evidence missing" >&2
        exit_code=1
    fi
    violations=""
    block=""
    # Content-check candidates accumulate here and are verified in ONE
    # batched nm pass after all edge blocks are scanned (a cudaqMLIR-sized
    # edge carries hundreds of objects; one nm -A reads them all).
    content_candidates=()
    content_display=()
    # classify_input <path-token>: an existing input's ownership, by
    # canonical location and by content. A provider component is forbidden
    # wherever it lives; any input object/archive whose strong ownership
    # definitions duplicate provider exports is an embedded copy, even in
    # the consumer's own tree. Direct objects are content-checked too: a
    # copy renamed into the build tree carries the same definitions.
    classify_input() {
        local tok="$1" resolved d is_upstream dup
        case "$tok" in
        /*) resolved="$tok" ;;
        *)  resolved="$ninja_dir/$tok" ;;
        esac
        if [ ! -e "$resolved" ]; then
            violations="$violations missing-input:$tok"
            return
        fi
        resolved="$(realpath "$resolved")"
        is_upstream=false
        for d in $upstream_dirs; do
            case "$resolved" in
            "$d"|"$d"/*) is_upstream=true ;;
            esac
        done
        if $is_upstream; then
            violations="$violations upstream-input:$tok"
            return
        fi
        if ! nm --defined-only "$resolved" >/dev/null 2>&1; then
            violations="$violations unreadable-input:$tok"
            return
        fi
        case " ${content_candidates[@]+"${content_candidates[@]}"} " in
        *" $resolved "*) ;;
        *)
            content_candidates+=("$resolved")
            content_display+=("$tok") ;;
        esac
    }
    scan_block() {
        [ -n "$block" ] || return 0
        local tok sub resolved link_mode=""
        # Search dirs for -l forms: GNU applies every -L to every -l
        # regardless of textual position, so precollect the complete
        # library path before classifying (LINK_PATH may follow
        # LINK_LIBRARIES in the record). Order among -L dirs kept.
        local -a edge_tokens=() link_dirs=() mode_stack=() implicit_dirs=()
        la_collect_implicit_lib_dirs implicit_dirs "$ninja_dir"
        for tok in $(printf '%s\n' "$block" | la_edge_tokens "$ninja_dir"); do
            edge_tokens+=("$tok")
            for sub in $(la_split_wl "$tok"); do
                case "$sub" in
                -L*) link_dirs+=("${sub#-L}") ;;
                esac
            done
        done
        for tok in "${edge_tokens[@]+"${edge_tokens[@]}"}"; do
            case "$tok" in
            UNRESOLVED:*)
                violations="$violations unresolved-evidence:${tok#UNRESOLVED:}"
                continue ;;
            esac
            if la_unresolved "$tok"; then
                violations="$violations unresolved-evidence:$tok"
                continue
            fi
            # -Wl,foo,bar forwards foo/bar to the linker; -L extends the
            # library search path; -l[:]<name> resolves through it.
            for sub in $(la_split_wl "$tok"); do
                case "$sub" in
                -L*) continue ;;   # already precollected; order kept
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
                # Legacy static/monolithic layout only: upstream's python
                # machinery embeds these utility archives there by design.
                # With shared providers the substitution leaves nothing to
                # exempt — the basenames fail like every other component.
                if [ -z "$upstream_dirs" ]; then
                    case "$sub" in
                    *libLLVMSupport.a|*libLLVMDemangle.a) continue ;;
                    esac
                fi
                case "$sub" in
                -l*)
                    resolved="$(la_resolve_library "$sub" "$link_mode" \
                        "${link_dirs[@]}" "$ninja_dir" $upstream_dirs \
                        "${implicit_dirs[@]}" 2>/dev/null || true)"
                    if [ -z "$resolved" ]; then
                        violations="$violations unresolved-evidence:$sub"
                        continue
                    fi
                    case "$resolved" in
                    *.a) classify_input "$resolved" ;;
                    *) : ;;   # shared library input: fine
                    esac ;;
                *.a|*.o) classify_input "$sub" ;;
                *) : ;;
                esac
            done
        done
    }
    while IFS= read -r line; do
        case "$line" in
        '@@EDGE@@') scan_block; block="" ;;
        *) block="$block$line
" ;;
        esac
    done <<< "$blocks"
    scan_block
    # Batched by-content ownership check over every collected input: one
    # nm -A pass, strong ownership definitions intersected per file. The
    # inspection itself is checked: a failed nm is evidence missing, never
    # a pass.
    if [ "${#content_candidates[@]}" -gt 0 ]; then
        provider_exports
        nm_rc=0
        nm_batch="$(nm -A --defined-only "${content_candidates[@]}" 2>&1)" || nm_rc=$?
        if [ "$nm_rc" -ne 0 ]; then
            violations="$violations unreadable-input:(batch nm failed rc=$nm_rc)"
        fi
        while read -r dup_file dup_name; do
            [ -n "$dup_file" ] || continue
            display="$dup_file"
            for ((ci = 0; ci < ${#content_candidates[@]}; ci++)); do
                case "$dup_file" in
                "${content_candidates[$ci]}"|"${content_candidates[$ci]}:"*)
                    display="${content_display[$ci]}"
                    break ;;
                esac
            done
            violations="$violations embedded-definitions:$display($dup_name)"
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
                <(echo "$provider_exported") -)
    fi
    violations="$(printf '%s\n' $violations | sort -u | sed '/^$/d' || true)"
    if [ -n "$violations" ]; then
        echo "check_mlir_symbols: $(basename "$consumer") links forbidden" \
             "upstream component inputs / unresolved evidence:" \
             "$(echo "$violations" | head -5)" >&2
        exit_code=1
    fi
elif ! $have_symtab; then
    echo "check_mlir_symbols: $(basename "$consumer") is stripped and no" \
         "CUDAQ_BUILD_NINJA link-input evidence was supplied" >&2
    exit_code=1
fi

exit "$exit_code"

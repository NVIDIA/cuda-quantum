#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Shared link-record provenance for the MLIR provider audits. Sourced by
# scripts/check_mlir_symbols.sh (consumer gates) and
# scripts/check_mlir_providers.sh (provider gates). The complete link
# record is harvested, not just the dependency edge: the build statement's
# explicit and '|' implicit inputs ('||' order-only excluded), the edge's
# variable block (LINK_LIBRARIES/LINK_FLAGS carry most real link inputs on
# CMake/Ninja layouts), and the contents of any rspfile the edge
# references.

# la_edge_blocks <build.ninja> <output-basename>
# Print every edge block producing the named output (basenames compared
# literally), blocks separated by @@EDGE@@ lines.
la_edge_blocks() {
    awk -v name="$2" '
        /^build / {
            in_edge = 0
            line = substr($0, 7)
            outs = substr(line, 1, index(line, ":") - 1)
            n = split(outs, o, /[ \t]+/)
            for (i = 1; i <= n; i++) {
                base = o[i]; sub(/.*\//, "", base)
                if (base == name) { in_edge = 1; print "@@EDGE@@"; print; next }
            }
            next
        }
        in_edge && /^[ \t]/ {print; next}
        {in_edge = 0}
    ' "$1"
}

# la_edge_tokens <ninja-dir> <<'BLOCK'
# Emit the block's link-input tokens, one per line. -Wl, arguments are
# split on commas into the underlying linker tokens; Ninja's escapes
# ('$$', '$ ', '$:') unescape before use. Unresolved evidence is reported
# literally as UNRESOLVED:<token> (a remaining '$', a bare .rsp token, or
# a referenced rspfile that cannot be read).
la_edge_tokens() {
    local ninja_dir="$1" block tok probe rp rf
    block="$(cat)"
    {
        echo "$block" | head -1 \
            | sed 's/^build [^:]*: [^ ]* *//' | awk -F'[|][|]' '{print $1}'
        echo "$block" | tail -n +2 \
            | { grep -vE '^[ \t]*rspfile(_content)? *=' || true; } \
            | sed -e 's/^[ \t]*[A-Za-z_][A-Za-z0-9_]* = //' -e 's/^[ \t]+//'
        echo "$block" | sed -n 's/^[ \t]*rspfile_content = //p'
    } | tr ' \t' '\n\n' | sed '/^$/d'
    for rf in $(echo "$block" | sed -n 's/^[ \t]*rspfile = //p'); do
        case "$rf" in
        /*) rp="$rf" ;;
        *)  rp="$ninja_dir/$rf" ;;
        esac
        if [ -f "$rp" ]; then
            tr ' \t' '\n\n' < "$rp" | sed '/^$/d'
        else
            echo "UNRESOLVED:$rf"
        fi
    done
}

# la_all_edge_tokens <build.ninja> <output-basename> <ninja-dir>
# Emit all link-input tokens of every edge producing the named output
# (see la_edge_tokens). Returns 1 when no edge matches.
la_all_edge_tokens() {
    local ninja="$1" out_name="$2" ninja_dir="$3" blocks line block=""
    blocks="$(la_edge_blocks "$ninja" "$out_name")"
    [ -n "$blocks" ] || return 1
    while IFS= read -r line; do
        case "$line" in
        '@@EDGE@@')
            [ -z "$block" ] || printf '%s\n' "$block" | la_edge_tokens "$ninja_dir"
            block="" ;;
        *) block="$block$line
" ;;
        esac
    done <<< "$blocks"
    [ -z "$block" ] || printf '%s\n' "$block" | la_edge_tokens "$ninja_dir"
}

# la_unescape_probe <token> -> 0 if the token is unresolved evidence.
la_unresolved() {
    local probe
    probe="$(printf '%s' "$1" | sed -e 's/\$\$/\x01/g' -e 's/\$ /\x02/g' -e 's/\$:/\x03/g')"
    case "$probe" in
    *'$'*|*.rsp) return 0 ;;
    esac
    return 1
}

# la_implicit_lib_dirs [build-dir]: the effective implicit library search
# path for -l resolution, from the build's own toolchain data rather than
# a hardcoded host tuple: CUDAQ_IMPLICIT_LINK_DIRS (as wired by the build),
# else the sibling CMakeCache's CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES, else
# the default compiler's self-reported search dirs, else a generic
# multiarch fallback.
# One directory per line on stdout, so paths with spaces survive.
la_implicit_lib_dirs() {
    local v=""
    if [ -n "${CUDAQ_IMPLICIT_LINK_DIRS:-}" ]; then
        # Serialized with ':' by the CMake gate env (list boundaries kept).
        echo "${CUDAQ_IMPLICIT_LINK_DIRS}" | tr ':' '\n'
        return
    fi
    # The configured toolchain record (CMakeFiles/<ver>/CMakeCXXCompiler.cmake)
    # is authoritative; CMakeCache.txt only names the compiler, not the dirs.
    local rec
    for rec in "${1:-}"/CMakeFiles/*/CMakeCXXCompiler.cmake; do
        [ -f "$rec" ] || continue
        v="$(sed -n 's/^set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "\(.*\)")/\1/p' \
            "$rec" | head -1)"
        [ -n "$v" ] && { printf '%s\n' "$v" | tr ';' '\n'; return; }
    done
    local cache="${1:-}/CMakeCache.txt"
    if [ -f "$cache" ]; then
        v="$(sed -n 's/^CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES:[^=]*=//p' "$cache" | head -1)"
        [ -n "$v" ] && { printf '%s\n' "$v" | tr ';' '\n'; return; }
        # No implicit-dir record: use the compiler the build actually used.
        v="$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "$cache" | head -1)"
        if [ -n "$v" ] && [ -x "$v" ]; then
            "${v}" -print-search-dirs 2>/dev/null \
                | sed -n 's/^libraries: =//p' | head -1 | tr ':' '\n'
            return
        fi
    fi
    printf '%s\n' /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu \
        /lib /usr/local/lib
}

# la_collect_implicit_lib_dirs <array-name> [build-dir]: read the
# line-per-dir output into a bash array (space-safe).
la_collect_implicit_lib_dirs() {
    local -n _out="$1"; _out=()
    local d
    while IFS= read -r d; do
        [ -n "$d" ] && _out+=("$d")
    done < <(la_implicit_lib_dirs "${2:-}")
}

# la_split_wl: split -Wl,foo,bar into foo bar; pass everything else through.
la_split_wl() {
    local tok="$1"
    case "$tok" in
    -Wl,*) printf '%s\n' "${tok#-Wl,}" | tr ',' '\n' ;;
    *) printf '%s\n' "$tok" ;;
    esac
}

# la_resolve_library <-lfoo|-l:libfoo.a> <static|> <dir...>: print the
# resolved file. GNU ld semantics: the -L directories are searched IN
# ORDER; in each directory libfoo.so is tried first, then libfoo.a as a
# per-directory fallback (the -Bdynamic default does NOT skip archives);
# -Bstatic restricts the whole search to archives. -l: names an exact
# file (mode-agnostic). Empty output when unresolved.
la_resolve_library() {
    local spec="$1" mode="$2"; shift 2
    local name d
    case "$spec" in
    -l:*)
        name="${spec#-l:}"
        for d in "$@"; do
            [ -f "$d/$name" ] && { echo "$d/$name"; return 0; }
        done ;;
    -l*)
        name="lib${spec#-l}"
        for d in "$@"; do
            [ "$mode" != "static" ] && [ -f "$d/$name.so" ] && {
                echo "$d/$name.so"; return 0; }
            [ -f "$d/$name.a" ] && { echo "$d/$name.a"; return 0; }
        done ;;
    esac
    return 1
}

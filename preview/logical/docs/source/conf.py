# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from pygments.lexer import RegexLexer
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Punctuation,
    String,
    Text,
)
from sphinx.highlighting import lexers as _sphinx_lexers

project = "CUDA-Q Logical"
copyright = "Copyright © 2026, NVIDIA Corporation & affiliates. All rights reserved."

# The build is hermetic: no autodoc and no notebook execution, so the docs
# environment never needs a built cudaq.logical package. Shipped examples are
# embedded with literalinclude and executed by pytest/CI instead.
extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.mathjax",
]


# Custom Pygments lexers for the MLIR dialects and text formats used by
# CUDA-Q Logical. Kept in sync with the `qlx`/`lvm`/`fabric` dialects.
class _MLIRLexer(RegexLexer):
    name = "MLIR"
    aliases = ["mlir"]
    filenames = ["*.mlir"]
    tokens = {
        "root": [
            (r"//.*?$", Comment.Single),
            (r'"([^"\\]|\\.)*"', String),
            (r"[%@#!^][\w.$-]*", Name.Variable),
            (
                r"\b(func|module|return|cf|llvm|memref|vector|tensor|affine|"
                r"scf|linalg|gpu|spv|builtin|arith|fabric|qlx|fq)\b",
                Keyword,
            ),
            (r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", Number),
            (r"[{}\[\]()<>,;:=*]", Punctuation),
            (r"[a-zA-Z_][\w.]*", Name),
            (r"\s+", Text),
            (r".", Text),
        ],
    }


class _QLXLexer(RegexLexer):
    name = "QLX"
    aliases = ["qlx", "qlx-asm"]
    filenames = ["*.qlx"]
    tokens = {
        "root": [
            (r"//.*?$", Comment.Single),
            (r'"([^"\\]|\\.)*"', String),
            (r"\.[a-zA-Z_]\w*", Keyword.Reserved),
            (r"%[\w$-]+", Name.Variable),
            (r"[a-zA-Z_][\w-]*(?=\s*=)", Name.Attribute),
            (r"-?\d+(?:\.\d+)?", Number),
            (r"[{}\[\]<>,;=]", Punctuation),
            (r"[a-zA-Z_]\w*", Name),
            (r"\s+", Text),
            (r".", Text),
        ],
    }


class _StimLexer(RegexLexer):
    name = "Stim"
    aliases = ["stim"]
    filenames = ["*.stim"]
    tokens = {
        "root": [
            (r"#.*?$", Comment.Single),
            (r"\b[A-Z][A-Z0-9_]*\b", Keyword),
            (r"-?\d+(?:\.\d+)?", Number),
            (r"[!@*\[\]()]", Punctuation),
            (r"[a-zA-Z_]\w*", Name),
            (r"\s+", Text),
            (r".", Text),
        ],
    }


for _cls in (_MLIRLexer, _QLXLexer, _StimLexer):
    _inst = _cls()
    for _alias in _cls.aliases:
        _sphinx_lexers[_alias] = _inst

# Broken cross-references must fail the build: myst.xref_missing is
# deliberately NOT suppressed. (toc.empty_glob is not suppressible: Sphinx
# emits it without a warning type. Every glob toctree therefore keeps at
# least one real page at each merge point via section-index stubs.)
suppress_warnings = [
    "misc.highlighting_failure",
]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "deflist",
    "tasklist",
    "fieldlist",
]
myst_heading_anchors = 3
myst_title_to_header = True

exclude_patterns = ["_build"]

# Theme parity with the main CUDA-Q documentation (docs/sphinx/conf.py).
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "prev_next_buttons_location": "both",
    "style_nav_header_background": "#76b900",  # NVIDIA green
}


def setup(app):
    return {"parallel_read_safe": True, "parallel_write_safe": True}

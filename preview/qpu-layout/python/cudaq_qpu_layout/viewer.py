# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Bake a trace into a standalone copy of `viewer.html`.

`viewer.html` opens on its own with a sample trace and can load any other from
disk. This writes a copy with a specific trace already embedded, so a run
produces a single file you can open or hand to someone.
"""

import json
import os

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "viewer.html")
OPEN_TAG = '<script id="trace-data" type="application/json">'
CLOSE_TAG = "</script>"


def render(doc, template=TEMPLATE):
    """Return the viewer HTML with `doc` embedded as its trace."""
    with open(template) as f:
        html = f.read()

    start = html.index(OPEN_TAG) + len(OPEN_TAG)
    end = html.index(CLOSE_TAG, start)

    payload = json.dumps(doc, separators=(",", ":"))
    # The payload sits inside a <script> block, so the one sequence that could
    # break out of it must not appear. Trace strings come from MLIR symbol
    # names, so this is belt and braces.
    if "</script" in payload.lower():
        raise ValueError("trace contains a literal '</script'; refusing to embed")

    return html[:start] + payload + html[end:]


def write_viewer(doc, out_path, template=TEMPLATE):
    with open(out_path, "w") as f:
        f.write(render(doc, template))
    return os.path.abspath(out_path)

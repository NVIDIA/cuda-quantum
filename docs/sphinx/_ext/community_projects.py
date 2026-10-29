# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Sphinx directive that discovers community projects.

A project is any folder `community/<name>/` containing an `index.rst`. The
metadata lives in the field list at the top of that page, e.g.::

    :repository: https://github.com/example/project
    :maintainers: Jane Doe
    :license: Apache-2.0
    :example: example.py
    :summary: One-line description shown on the landing page.

    Project Title
    =============

The `community-projects` directive builds the listing and the toctree from
those pages, so adding a project never requires editing an existing file.
"""

from pathlib import Path

from community_metadata import discover
from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective


class CommunityProjects(SphinxDirective):
    has_content = False

    def run(self):
        root = Path(self.env.srcdir) / self.env.docname.rsplit("/", 1)[0]
        projects, errors = discover(root)
        for page in root.glob("*/index.rst"):
            self.env.note_dependency(str(page))
        for e in errors:
            self.state_machine.reporter.warning(e, line=self.lineno)
        docs = [f"{p['name']}/index" for p in projects]

        lines = [".. toctree::", "   :hidden:", ""]
        lines += [f"   {d}" for d in docs]
        lines.append("")
        for p, doc in zip(projects, docs):
            lines += [
                f"* :doc:`{p['title']} <{doc}>` |---| {p['summary']}", "",
                f"  Maintained by {p['maintainers']} "
                f"(`repository <{p['repository']}>`__, {p['license']} license)",
                ""
            ]
        if not projects:
            lines.append("No community projects are listed yet.")

        container = nodes.container()
        self.state.nested_parse(
            StringList(lines, source=self.env.doc2path(self.env.docname)),
            self.content_offset, container)
        return container.children


def setup(app):
    app.add_directive("community-projects", CommunityProjects)
    return {"parallel_read_safe": True, "parallel_write_safe": True}

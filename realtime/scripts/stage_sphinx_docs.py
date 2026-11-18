#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Stage the CUDA-Q Realtime docs for the Sphinx build.

The realtime docs are written once, in `realtime/docs/`, for GitHub: mermaid
fences, SVG images and links relative to the repository. The Sphinx pages in
`docs/sphinx/using/realtime/` only include a staged copy, which this script
writes to `docs/sphinx/_mdgen/realtime/` during `scripts/build_docs.sh`:

  render  Renders every mermaid fence to a file named after the fence's hash,
          with the official `mermaid-cli` Docker image, or the `mmdc` named
          by `CUDAQ_MMDC`. Fences already rendered are skipped. Needs Docker,
          so it runs outside the docs container (on the CI runner, or on the
          host).
  stage   Writes the Sphinx copies: each diagram becomes an `image` directive
          on its rendered SVG, links to other realtime docs become links to
          their Sphinx pages, and links to other repository files become
          GitHub links pinned to the commit being built. Needs only Python.

  stage_sphinx_docs.py render [--out DIR]
  stage_sphinx_docs.py stage [--diagrams DIR] [--out DIR]

DIR defaults to `$CUDAQ_REALTIME_DIAGRAMS`, else `build/docs/realtime-diagrams`.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DOCS_DIR = 'realtime/docs'
# Each engineering doc and the page under `docs/sphinx/using/realtime/` that
# includes its staged copy.
PAGES = {
    'architecture.md': 'architecture',
    'cudaq_realtime_host_api.md': 'host',
    'cudaq_realtime_message_protocol.md': 'protocol',
}
SPHINX_PAGE_DIR = 'using/realtime'
STAGED_DIR = 'docs/sphinx/_mdgen/realtime'
GITHUB = 'https://github.com/NVIDIA/cuda-quantum/blob/'
MERMAID_CLI_IMAGE = (
    'ghcr.io/mermaid-js/mermaid-cli/mermaid-cli:11.17.1@sha256:'
    '7b18341ab7398a0cfd290fdbf066243151c0adb4cbf5b81d8cf074f82019ec6f')
# Rounded and cylinder shapes are drawn with `rough.js`, which is seeded at
# random unless told otherwise.
MERMAID_CONFIG = {'handDrawnSeed': 1}
# `autosectionlabel_maxdepth` in `docs/sphinx/conf.py`: only headings this deep
# have a label `{ref}` can name.
LABEL_DEPTH = 2

FENCE_RE = re.compile(r'^(`{3,}|~{3,})\s*(\S*)')
HEADING_RE = re.compile(r'^(#{1,6})\s+(.*?)\s*#*\s*$')
SVG_IMAGE_RE = re.compile(r'^\s*!\[([^\]]*)\]\(([^)\s]+\.svg)\)\s*$')
LINK_RE = re.compile(r'(?<!!)\[([^\]]*)\]\(([^)\s]+)\)')


def die(msg):
    print(f'stage_sphinx_docs.py: error: {msg}', file=sys.stderr)
    sys.exit(1)


def sha1(text):
    return hashlib.sha1(text.encode()).hexdigest()


def default_diagrams_dir():
    return os.environ.get('CUDAQ_REALTIME_DIAGRAMS') or os.path.join(
        REPO, 'build', 'docs', 'realtime-diagrams')


def fence_end(lines, start, marker, path):
    for i in range(start + 1, len(lines)):
        if lines[i].strip() == marker:
            return i
    die(f'{path}:{start + 1}: unclosed code fence')


def read_lines(path):
    with open(path) as f:
        return f.read().split('\n')


def mermaid_fences(path):
    """[(body, heading above it)] for each mermaid fence of a doc."""
    lines, out, heading, i = read_lines(path), [], '', 0
    while i < len(lines):
        m = FENCE_RE.match(lines[i])
        if m:
            end = fence_end(lines, i, m.group(1), path)
            if m.group(2) == 'mermaid':
                out.append(
                    ('\n'.join(lines[i + 1:end]).rstrip() + '\n', heading or
                     'Diagram'))
            i = end + 1
            continue
        h = HEADING_RE.match(lines[i])
        if h:
            heading = link_text_only(h.group(2))
        i += 1
    return out


# --- render ------------------------------------------------------------------


def render(diagrams_dir):
    """Render the fences of every doc that are not in `diagrams_dir` yet."""
    os.makedirs(diagrams_dir, exist_ok=True)
    todo = {}
    for doc in PAGES:
        for body, _ in mermaid_fences(os.path.join(REPO, DOCS_DIR, doc)):
            key = sha1(body)
            if not os.path.isfile(os.path.join(diagrams_dir, key + '.svg')):
                todo[key] = body
    if not todo:
        print(f'all diagrams already rendered in {diagrams_dir}')
        return
    work = tempfile.mkdtemp(prefix='realtime-diagrams-', dir=diagrams_dir)
    try:
        for key, body in todo.items():
            with open(os.path.join(work, key + '.mmd'), 'w') as f:
                f.write(body)
        with open(os.path.join(work, 'config.json'), 'w') as f:
            json.dump(MERMAID_CONFIG, f)
        mmdc = os.environ.get('CUDAQ_MMDC')
        if mmdc:
            for key in todo:
                run_or_die([
                    mmdc, '-q', '-i',
                    os.path.join(work, key + '.mmd'), '-o',
                    os.path.join(work, key + '.svg'), '-t', 'default', '-b',
                    'transparent', '-I', 'mermaid-svg', '-c',
                    os.path.join(work, 'config.json')
                ])
        else:
            if not shutil.which('docker'):
                die('rendering needs Docker (or CUDAQ_MMDC naming a mermaid-cli '
                    '`mmdc`); run `realtime/scripts/stage_sphinx_docs.py '
                    'render` where Docker is available, then pass the output '
                    'directory as CUDAQ_REALTIME_DIAGRAMS')
            # One container for all fences; `mmdc` runs once per fence in it,
            # with the image's own Chromium (`/puppeteer-config.json`, which
            # its entrypoint passes too).
            loop = ('for f in /data/*.mmd; do mmdc -p /puppeteer-config.json -q'
                    ' -i "$f" -o "${f%.mmd}.svg" -t default -b transparent'
                    ' -I mermaid-svg -c /data/config.json || exit 1; done')
            run_or_die([
                'docker', 'run', '--rm', '-u', f'{os.getuid()}:{os.getgid()}',
                '-e', 'HOME=/tmp', '-v', f'{work}:/data', '--entrypoint', 'sh',
                MERMAID_CLI_IMAGE, '-c', loop
            ])
        for key in todo:
            os.replace(os.path.join(work, key + '.svg'),
                       os.path.join(diagrams_dir, key + '.svg'))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f'rendered {len(todo)} diagram(s) into {diagrams_dir}')


def run_or_die(cmd):
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        die(f'mermaid-cli failed ({cmd[0]}):\n{run.stderr.strip()}')


# --- stage -------------------------------------------------------------------


def link_text_only(text):
    return re.sub(r'(?<!!)\[([^\]]*)\]\([^)]*\)', r'\1', text)


def github_slug(title):
    """GitHub's anchor for a heading, built from its visible text."""
    text = re.sub(r'[^\w\- ]', '', link_text_only(title).lower())
    return text.replace(' ', '-')


def plain_title(title):
    """The heading as Sphinx's `autosectionlabel` names it."""
    return re.sub(r'[`*]', '', link_text_only(title))


def headings(path):
    """[(level, title, slug)] of a Markdown file, outside code fences."""
    lines, out, i = read_lines(path), [], 0
    while i < len(lines):
        m = FENCE_RE.match(lines[i])
        if m:
            i = fence_end(lines, i, m.group(1), path) + 1
            continue
        h = HEADING_RE.match(lines[i])
        if h:
            out.append((len(h.group(1)), h.group(2), github_slug(h.group(2))))
        i += 1
    return out


def section_label(path, anchor):
    """Title of the labeled section an anchor lands in: the heading itself if
    it is at most `LABEL_DEPTH` deep, else the closest such heading above it."""
    hs = headings(path)
    for n, (_, _, slug) in enumerate(hs):
        if slug == anchor:
            for lvl, ttl, _ in reversed(hs[:n + 1]):
                if lvl <= LABEL_DEPTH:
                    return plain_title(ttl)
            return None
    return None


def git(*args):
    run = subprocess.run(['git', '-C', REPO] + list(args),
                         capture_output=True,
                         text=True)
    return run.stdout.strip() if run.returncode == 0 else None


class Stager:

    def __init__(self, diagrams_dir, out_dir):
        self.diagrams_dir = diagrams_dir
        self.out_dir = out_dir
        ref = os.environ.get('CUDAQ_DOCS_SOURCE_REF', 'HEAD')
        self.commit = git('rev-parse', '--verify', f'{ref}^{{commit}}')
        if not self.commit:
            die(f'cannot resolve {ref} to a commit (CUDAQ_DOCS_SOURCE_REF)')
        self.errors = []

    def error(self, msg):
        self.errors.append(msg)

    def link(self, text, url, source):
        """The Sphinx form of one Markdown link of a realtime doc."""
        if re.match(r'^[a-z]+:', url):
            return f'[{text}]({url})'
        path, _, anchor = url.partition('#')
        target = source if not path else os.path.normpath(
            os.path.join(os.path.dirname(source), path))
        rel = os.path.relpath(target, REPO)
        if rel.startswith('..') or not os.path.exists(target):
            self.error(f'{os.path.relpath(source, REPO)}: link {url}: no such '
                       'file in the repository')
            return f'[{text}]({url})'
        doc = os.path.relpath(target, os.path.join(REPO, DOCS_DIR))
        if doc in PAGES:
            page = PAGES[doc]
            if not anchor:
                return f'[{text}]({page}.md)'
            label = section_label(target, anchor)
            if label is None:
                self.error(f'{os.path.relpath(source, REPO)}: link {url}: no '
                           f'heading #{anchor} in {rel}')
                return f'[{text}]({url})'
            return f'{{ref}}`{text} <{SPHINX_PAGE_DIR}/{page}:{label}>`'
        if anchor and rel.endswith('.md') and section_label(target,
                                                            anchor) is None:
            self.error(f'{os.path.relpath(source, REPO)}: link {url}: no '
                       f'heading #{anchor} in {rel}')
        if git('cat-file', '-e', f'{self.commit}:{rel}') is None:
            self.error(f'{os.path.relpath(source, REPO)}: link {url}: {rel} is '
                       f'not in commit {self.commit[:12]}, so GitHub would '
                       'not show it')
        suffix = f'#{anchor}' if anchor else ''
        return f'[{text}]({GITHUB}{self.commit}/{rel}{suffix})'

    def image(self, svg_path, staged_name, alt):
        """An image directive on a copy of `svg_path` in the staged tree."""
        with open(svg_path) as f:
            svg = f.read()
        dest = os.path.join(self.out_dir, staged_name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(svg_path, dest)
        lines = [
            f'```{{image}} /{os.path.relpath(dest, os.path.join(REPO, "docs/sphinx"))}',
            f':alt: {" ".join(alt.split())}'
        ]
        m = re.search(r'viewBox="[-\d.]+ [-\d.]+ ([\d.]+) [\d.]+"', svg)
        if m:
            lines.append(f':width: {math.ceil(float(m.group(1)))}px')
        return lines + ['```']

    def stage(self, doc):
        source = os.path.join(REPO, DOCS_DIR, doc)
        lines, out, heading, i = read_lines(source), [], '', 0
        while i < len(lines):
            line = lines[i]
            m = FENCE_RE.match(line)
            if m:
                end = fence_end(lines, i, m.group(1), source)
                if m.group(2) == 'mermaid':
                    body = '\n'.join(lines[i + 1:end]).rstrip() + '\n'
                    key = sha1(body)
                    svg = os.path.join(self.diagrams_dir, key + '.svg')
                    if os.path.isfile(svg):
                        out.extend(
                            self.image(svg, f'diagrams/{key}.svg', heading or
                                       'Diagram'))
                    else:
                        self.error(f'{doc}: diagram under "{heading}" is not '
                                   f'rendered ({key}.svg missing in '
                                   f'{self.diagrams_dir})')
                else:
                    out.extend(lines[i:end + 1])
                i = end + 1
                continue
            h = HEADING_RE.match(line)
            img = SVG_IMAGE_RE.match(line)
            if h:
                # Sphinx anchors every heading itself; a link in one (usually
                # to itself) would only be a broken cross-reference.
                heading = link_text_only(h.group(2))
                out.append(link_text_only(line))
            elif img:
                svg = os.path.normpath(
                    os.path.join(os.path.dirname(source), img.group(2)))
                if os.path.isfile(svg):
                    out.extend(
                        self.image(
                            svg, os.path.relpath(svg, os.path.dirname(source)),
                            img.group(1) or heading))
                else:
                    self.error(f'{doc}: image {img.group(2)} not found')
            else:
                out.append(
                    LINK_RE.sub(
                        lambda lm: self.link(lm.group(1), lm.group(2), source),
                        line))
            i += 1
        with open(os.path.join(self.out_dir, PAGES[doc] + '.md'), 'w') as f:
            f.write('\n'.join(out))


def stage(diagrams_dir, out_dir):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    missing = any(
        not os.path.isfile(os.path.join(diagrams_dir,
                                        sha1(body) + '.svg'))
        for doc in PAGES
        for body, _ in mermaid_fences(os.path.join(REPO, DOCS_DIR, doc)))
    if missing and (os.environ.get('CUDAQ_MMDC') or shutil.which('docker')):
        render(diagrams_dir)
    stager = Stager(diagrams_dir, out_dir)
    for doc in PAGES:
        stager.stage(doc)
    if stager.errors:
        die('\n  '.join(['cannot stage the realtime docs:'] + stager.errors + [
            'Diagrams are rendered with '
            '`realtime/scripts/stage_sphinx_docs.py render` '
            'where Docker is available.'
        ]))
    print(f'staged {len(PAGES)} realtime docs into {out_dir} (links pinned to '
          f'{stager.commit[:12]})')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('render', help='render the mermaid fences to SVG')
    r.add_argument('--out',
                   default=default_diagrams_dir(),
                   help='directory of rendered SVGs')
    s = sub.add_parser('stage', help='write the Sphinx copies')
    s.add_argument('--diagrams',
                   default=default_diagrams_dir(),
                   help='directory of rendered SVGs')
    s.add_argument('--out',
                   default=os.path.join(REPO, STAGED_DIR),
                   help='staging directory')
    args = p.parse_args()
    if args.command == 'render':
        render(os.path.abspath(args.out))
    else:
        stage(os.path.abspath(args.diagrams), os.path.abspath(args.out))


if __name__ == '__main__':
    main()

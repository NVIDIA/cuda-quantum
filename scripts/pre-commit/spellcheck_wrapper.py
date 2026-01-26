#!/usr/bin/env python3

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
Pre-commit wrapper for pyspelling checks.

Runs pyspelling on individual files passed by pre-commit, using the existing
spellcheck_config.yml configuration.

Note: We use a custom wrapper instead of the native pyspelling hook because
it has `pass_filenames: false` and would check ALL files on every run. This
wrapper enables per-file checking for better performance.
"""

import sys
import subprocess
import os
import argparse


def run_pyspelling(task_name: str, files: list) -> int:
    """
    Run pyspelling on specified files for a given task.

    Args:
        task_name: The pyspelling task name from spellcheck_config.yml
        files: List of file paths to check

    Returns:
        Exit code (0 = success, non-zero = failure)
    """
    if not files:
        print(f"No files to check for task '{task_name}'")
        return 0

    repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'],
                                        text=True).strip()

    config_path = os.path.join(
        repo_root, '.github/workflows/config/spellcheck_config.yml')

    # Check each file individually for better error reporting
    failures = []
    for filepath in files:
        if not os.path.exists(filepath):
            continue

        print(f"Checking {filepath}...")
        result = subprocess.run(
            ['pyspelling', '-n', task_name, '-c', config_path, '-S', filepath],
            capture_output=True,
            text=True)

        if result.returncode != 0:
            failures.append(filepath)
            print(f"FAILED: {filepath}")
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        else:
            print(f"OK: {filepath}")

    if failures:
        print(f"\n{len(failures)} file(s) failed spell check:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"\nAll {len(files)} file(s) passed spell check.")
    return 0


def main():
    valid_tasks = ['markdown', 'rst', 'cxx_headers', 'cxx_examples', 'python']

    parser = argparse.ArgumentParser(
        description='Pre-commit wrapper for pyspelling checks')
    parser.add_argument('task_name',
                        choices=valid_tasks,
                        help='Pyspelling task name from spellcheck_config.yml')
    parser.add_argument('files',
                        nargs='*',
                        help='Files to check (passed by pre-commit)')

    args = parser.parse_args()
    return run_pyspelling(args.task_name, args.files)


if __name__ == '__main__':
    sys.exit(main())

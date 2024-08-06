# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import re
import sys
from pathlib import Path
import subprocess
from shutil import which

if which('jupyter') is None:
    print("Please install jupyter, e.g. with `pip install notebook`.")
    exit(1)


def read_available_backends():
    available_backends = sys.stdin.readlines()
    available_backends = ' '.join(available_backends).split()
    return [backend.strip() for backend in available_backends]


def validate(notebook_filename, available_backends):
    with open(notebook_filename) as f:
        lines = f.readlines()
    for notebook_content in lines:
        match = re.search('set_target[\\\s\(]+"(.+)\\\\"[)]', notebook_content)
        if match and (match.group(1) not in available_backends):
            return False
    for notebook_content in lines:
        match = re.search('--target ([^ ]+)', notebook_content)
        if match and (match.group(1) not in available_backends):
            return False
    return True


def execute(notebook_filename):
    notebook_filename_out = notebook_filename.replace('.ipynb',
                                                      '.nbconvert.ipynb')
    try:
        subprocess.run([
            "jupyter", "nbconvert", "--to", "notebook", "--execute",
            notebook_filename
        ],
                       check=True)
        os.remove(notebook_filename_out)
        return True
    except subprocess.CalledProcessError:
        print('Error executing the notebook "%s".\n\n' % notebook_filename)
        return False


def print_results(success, failed, skipped=[]):
    if success:
        print("Success! The following notebook(s) executed successfully:\n" +
              " ".join(success))

    if failed:
        print(
            "::error::The following notebook(s) raised one or more errors:\n" +
            " ".join(failed))

    if skipped:
        print("::warning::Skipped validation for the following notebook(s):\n" +
              " ".join(skipped))

    if not failed and not skipped:
        print("Success! All the notebook(s) executed successfully.")
    elif failed:
        exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        notebook_filenames = sys.argv[1:]
        notebooks_success, notebooks_failed = ([] for i in range(2))
        for notebook_filename in notebook_filenames:
            if (execute(notebook_filename)):
                notebooks_success.append(notebook_filename)
            else:
                notebooks_failed.append(notebook_filename)
        print_results(notebooks_success, notebooks_failed)
    else:
        available_backends = read_available_backends()
        notebook_filenames = [
            str(fn.relative_to(Path(__file__).parent))
            for fn in Path(__file__).parent.rglob('*.ipynb')
            if not fn.name.endswith('.nbconvert.ipynb')
        ]

        if not notebook_filenames:
            print('Failed! No notebook(s) found.')
            exit(10)

        notebooks_success, notebooks_skipped, notebooks_failed = (
            [] for i in range(3))
        for notebook_filename in notebook_filenames:
            if (validate(notebook_filename, available_backends)):
                if (execute(notebook_filename)):
                    notebooks_success.append(notebook_filename)
                else:
                    notebooks_failed.append(notebook_filename)
            else:
                notebooks_skipped.append(notebook_filename)

        print_results(notebooks_success, notebooks_failed, notebooks_skipped)

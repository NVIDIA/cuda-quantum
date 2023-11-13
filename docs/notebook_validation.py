# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import glob, os, re, sys
from pathlib import Path
import subprocess
from shutil import which
if which('jupyter') is None:
    print("Please install jupyter, e.g. with `pip install notebook`.")
    exit(1)

def available():
    available_backends = sys.stdin.readlines()
    if available_backends:
        available_backends = [
            available_backend.strip()
            for available_backend in available_backends
        ]
        return available_backends

def validate(notebook_filename):
    with open(notebook_filename) as f:
        lines = f.readlines()
    for notebook_content in lines:
        status = True
        match = re.search('set_target[\\\s\(]+"(.+)\\\\"[)]', notebook_content)
        if match:
            if (match.group(1) not in available_backends):
                status = False
                break
    return status

def execute(notebook_filename, run_path='.'):
    notebook_filename_out = notebook_filename.replace('.ipynb',
                                                      '.nbconvert.ipynb')
    success = True
    try:
        subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", notebook_filename], check = True)
        os.remove(notebook_filename_out)
    except subprocess.CalledProcessError:
        print('Error executing the notebook "%s".\n\n' % notebook_filename)
        success = False
    return success


if __name__ == "__main__":
    available_backends = available()
    if len(sys.argv) > 1:
        notebook_filenames = sys.argv[1:]
    else:
        notebook_filenames = [
            fn for fn in glob.glob(f"{Path(__file__).parent}/docs/**/*.ipynb", recursive=True) if not fn.endswith('.nbconvert.ipynb')]

    if notebook_filenames:
        notebooks_failed = []
        for notebook_filename in notebook_filenames:
            if (validate(notebook_filename)):
                notebooks_failed.append(notebook_filename)
            else:
                print(f"Skipped! Missing backends for {notebook_filename}")

        if len(notebooks_failed) > 0:
            print("Failed! The following notebook(s) raised errors:\n" +
                  " ".join(notebooks_failed))
            exit(1)
        else:
            print("Success! All the notebook(s) executed successfully.")
    else:
        print('Failed! No notebook found in the current directory.')
        exit(10)

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from shutil import which

if which('jupyter') is None:
    print("Please install jupyter, e.g. with `pip install notebook`.")
    exit(1)


def read_available_backends():
    available_backends = sys.stdin.readlines()
    available_backends = ' '.join(available_backends).split()
    return [backend.strip() for backend in available_backends]


# Following pattern matches
# `set_target("abc")`
# `set_target( "abc")`
# `set_target("abc", option="xyz")`
# `set_target("abc", option = "xyz")`
# `set_target(\"abc\")`
# `set_target( \"abc\")`
# `set_target(\"abc\", option=\"xyz\")`
# `set_target(\"abc\", option = \"xyz\")`
# `set_target('abc')`
# `set_target( 'abc')`
# `set_target('abc', option='xyz')`
# `set_target('abc', option = 'xyz')`
pattern = r"set_target\(\s*(\\?['\"])([^'\"]+)\1(?:\s*,\s*option\s*=\s*(\\?['\"])([^'\"]+)\3)?\)"

# Platform-dependent skip list: notebooks that require GPU/CUDA libraries
# These cannot run on CPU-only systems even with fallback logic
GPU_REQUIRED_NOTEBOOKS = [
    'afqmc.ipynb',  # AFQMC algorithm, times out on CPU
    'digitized_counterdiabatic_qaoa.ipynb',  # QAOA optimization, times out on CPU
    'qm_mm_pe.ipynb',  # VQE+SCF with many iterations, times out on CPU
    'vqe_advanced.ipynb',  # VQE optimization with `mqpu`, requires multi-GPU
    'edge_detection.ipynb',  # Requires CuPy
    'entanglement_acc_hamiltonian_simulation.ipynb',  # Requires CuPy
    'skqd.ipynb',  # Requires CuPy
    'divisive_clustering_coresets.ipynb',  # Multi-GPU MPI demo
    'quantum_pagerank.ipynb',  # Requires dynamics target
]


def validate(notebook_filename, available_backends):
    """
    Validate if a notebook can run with the available backends.
    
    This function is fallback-aware: if a notebook has multiple set_target()
    calls (e.g., `nvidia` with `qpp-cpu` fallback), it will return True if ANY
    of the targets is available.
    """
    with open(notebook_filename) as f:
        lines = f.readlines()

    # Check platform-dependent skip list
    base_name = os.path.basename(notebook_filename)
    has_gpu = 'nvidia' in available_backends
    if not has_gpu and base_name in GPU_REQUIRED_NOTEBOOKS:
        return False

    # Collect all set_target calls
    targets_found = []
    for notebook_content in lines:
        # Skip commented lines
        if re.search(r'^\s*#', notebook_content) or re.search(
                r'^\s*"#', notebook_content):
            continue

        match = re.search(pattern, notebook_content)
        if match:
            targets_found.append(match.group(2))

    # Also check --target flags in shell commands
    for notebook_content in lines:
        match = re.search('--target ([^ ]+)', notebook_content)
        if match:
            targets_found.append(match.group(1))

    # If no targets specified, notebook uses default (`qpp-cpu`) - allow it
    if not targets_found:
        return True

    # If ANY target is available (including fallback), allow the notebook
    return any(target in available_backends for target in targets_found)


def execute(notebook_filename, jupyter_kernel=None, timeout_seconds=600):
    """Execute a notebook with timeout."""
    notebook_filename_out = notebook_filename.replace('.ipynb',
                                                      '.nbconvert.ipynb')
    notebook_basename = os.path.basename(notebook_filename)
    try:
        start_time = time.perf_counter()
        cmd = [
            "jupyter", "nbconvert", "--to", "notebook", "--execute",
            f"--ExecutePreprocessor.timeout={timeout_seconds}",
            notebook_filename
        ]
        if jupyter_kernel:
            cmd.extend(["--ExecutePreprocessor.kernel_name", jupyter_kernel])

        subprocess.run(cmd, check=True)
        elapsed = time.perf_counter() - start_time
        print(f"  ✓  {notebook_basename}: {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError:
        elapsed = time.perf_counter() - start_time
        print(f"  ✗  {notebook_basename}: FAILED after {elapsed:.1f}s")
        return False
    finally:
        if os.path.exists(notebook_filename_out):
            os.remove(notebook_filename_out)


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
    # Check for optional Jupyter kernel argument
    jupyter_kernel = None
    args_to_process = sys.argv[1:]

    # Check if first argument is a kernel name (not a `.ipynb` file)
    if args_to_process and not args_to_process[0].endswith('.ipynb'):
        jupyter_kernel = args_to_process[0]
        args_to_process = args_to_process[1:]

    if args_to_process:
        # Direct notebook file execution mode
        notebook_filenames = args_to_process
        notebooks_success, notebooks_failed = ([] for i in range(2))
        for notebook_filename in notebook_filenames:
            if (execute(notebook_filename, jupyter_kernel=jupyter_kernel)):
                notebooks_success.append(notebook_filename)
            else:
                notebooks_failed.append(notebook_filename)
        print_results(notebooks_success, notebooks_failed)
    else:
        available_backends = read_available_backends()
        notebook_filenames = [
            str(fn)  # Use absolute paths for robustness
            for fn in Path(__file__).parent.rglob('*.ipynb')
            if not fn.name.endswith('.nbconvert.ipynb')
        ]

        if not notebook_filenames:
            print('Failed! No notebook(s) found.')
            exit(10)

        notebooks_success, notebooks_skipped, notebooks_failed = (
            [] for i in range(3))

        ## `quantum_transformer`:
        ## See: https://github.com/NVIDIA/cuda-quantum/issues/2689
        notebooks_skipped = [
            'quantum_transformer.ipynb', 'logical_aim_sqale.ipynb',
            'hybrid_quantum_neural_networks.ipynb',
            'unitary_compilation_diffusion_models.ipynb', 'qsci.ipynb'
        ]

        for notebook_filename in notebook_filenames:
            base_name = os.path.basename(notebook_filename)
            if base_name in notebooks_skipped:
                continue  # Already skipped, no need to re-check
            if not validate(notebook_filename, available_backends):
                notebooks_skipped.append(base_name)
                continue
            if execute(notebook_filename, jupyter_kernel=jupyter_kernel):
                notebooks_success.append(notebook_filename)
            else:
                notebooks_failed.append(notebook_filename)

        print_results(notebooks_success, notebooks_failed, notebooks_skipped)

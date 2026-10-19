# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Install an EDEM wheel, relocate its package, and build native consumers."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def inspect_package(python, directory=None):
    code = """
import ctypes
from importlib.metadata import version
import json
from pathlib import Path
import sys

if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])
import cudaq_edem
from cudaq_edem import _core

assert cudaq_edem.__version__ == version('cudaq-edem')
assert _core.__file__.endswith(('.so', '.pyd'))
for name in ('LICENSE.nanobind', 'LICENSE.robin-map'):
    assert (Path(cudaq_edem.__file__).parent / 'share' / 'cudaq-edem' /
            'licenses' / name).is_file()
cmake = Path(cudaq_edem.get_cmake_dir())
assert (cmake / 'cudaq-edem-config.cmake').is_file()
library_dir = cmake.parent.parent
libraries = []
for name in ('cudaq-edem', 'cudaq-gf2'):
    pattern = f'lib{name}*.dylib' if sys.platform == 'darwin' else f'lib{name}.so*'
    libraries.append(ctypes.CDLL(str(next(library_dir.glob(pattern)))))
print(json.dumps({'package': str(Path(cudaq_edem.__file__).parent),
                  'cmake': str(cmake), 'version': cudaq_edem.__version__}))
"""
    command = [str(python), "-I", "-c", code]
    if directory is not None:
        command.append(str(directory))
    return json.loads(subprocess.check_output(command, text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    consumer = Path(__file__).resolve().parent / "consumer"

    with tempfile.TemporaryDirectory(
            prefix="cudaq-edem-wheel-test-") as temporary:
        work = Path(temporary)
        environment = work / "venv"
        subprocess.run([sys.executable, "-m", "virtualenv",
                        str(environment)],
                       check=True)
        python = environment / "bin/python"
        subprocess.run([
            str(python), "-m", "pip", "install", "--no-index", "--no-deps",
            str(wheel)
        ],
                       check=True,
                       cwd=work)
        installed = inspect_package(python)
        relocated = work / "relocated"
        relocated.mkdir()
        shutil.move(installed["package"], relocated / "cudaq_edem")
        package = inspect_package(python, relocated)
        assert Path(package["package"]) == relocated / "cudaq_edem"

        for path in Path(package["cmake"]).glob("*.cmake"):
            if installed["package"] in path.read_text():
                raise RuntimeError(f"Package retains its old prefix: {path}")
        build = work / "consumer"
        commands = [[
            "cmake", "-S",
            str(consumer), "-B",
            str(build), "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_COMPILE_WARNING_AS_ERROR=ON",
            f"-Dcudaq-edem_DIR:PATH={package['cmake']}",
            f"-DCUDAQ_EDEM_TEST_VERSION:STRING={package['version']}"
        ],
                    [
                        "cmake", "--build",
                        str(build), "--config", "Release", "--parallel", "2"
                    ],
                    [
                        "ctest", "--test-dir",
                        str(build), "-C", "Release", "--output-on-failure"
                    ]]
        for command in commands:
            subprocess.run(command, check=True, cwd=work)
    print(
        "EDEM wheel import, native library loading, relocation, and consumers passed"
    )


if __name__ == "__main__":
    main()

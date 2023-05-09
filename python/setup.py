import os

from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

__version__ = "0.0.1"

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        # self.sourcedir = os.fspath(Path(sourcedir).resolve())
        self.name = name
        self.sourcedir = "/cuda-quantum/python/cudaq/*.so"
        # Trying to include the python header files.
        # Can likely remove this because it didn't help:
        # self.include_dirs = [
        #     "/cuda-quantum/python/src/",
        #     "/cuda-quantum/python/src/_cudaq.cpp",
        # ]
        self.depends = ["/cuda-quantum/python/cudaq/*.so"]
        self.runtime_library_dirs = ["/cuda-quantum/python/cudaq/*.so"]

class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        pass 
        # # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        # full_extension_path = Path.cwd() / self.get_ext_fullpath(ext.name)
        # extension_directory = full_extension_path.parent.resolve()

        # # TODO: This is where I can handle all of the cmake building of the python
        # # extension. Not sure if that's the method we will wind up using, but this
        # # is where we'd do it.

        # debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        # cfg = "Debug" if debug else "Release"
        # cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

setup(
    name="cuda-quantum",
    version=__version__,    
    url="https://github.com/NVIDIA/cuda-quantum",
    long_description="",
    ext_modules=[CMakeExtension("_pycudaq")],
    # cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.10.6",

    # NOTE: This succesfully creates an importable module name "cudaq" 
    # despite the above being named "_cudaq"!! It has the same `PyInit__cudaq`
    # error but it's at least promising.
    packages=["cudaq"],
    package_data={"cudaq": ["/cuda-quantum/python/cudaq/*.so", "/cuda-quantum/python/src/_cudaq.cpp"]},
    
    # extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    install_requires=[
        "numpy",
        "pytest",
        "setuptools"
    ],
)
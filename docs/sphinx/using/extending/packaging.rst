**********************************************
Package & Distribute a Backend Plugin
**********************************************

This guide covers how to package a CUDA-Q backend implementation as an
installable Python package, distribute it to end users of the target, and make
it discoverable by both the Python runtime and ``nvq++`` after they install it.
For how to *implement* a backend (``ServerHelper`` or ``QPU`` subclass), see
:doc:`backend`.


Plugin Package Layout
=====================

Every plugin follows the same directory convention:

.. code-block:: text

    my-backend/
    ├── targets/
    │   └── my-backend.yml          # Target YAML configuration
    ├── lib/
    │   └── libcudaq-serverhelper-my-backend.so  # Backend shared library
    ├── data/                        # Optional auxiliary files
    │   └── topology.txt
    ├── pyproject.toml               # Python package metadata
    ├── __init__.py                  # Entry-point registration
    └── __main__.py                  # --install-nvqpp hook

The ``targets/`` and ``lib/`` directories are required. The ``data/`` directory
is optional and holds any auxiliary files your backend needs at runtime.


Target YAML Reference (Plugin Fields)
======================================

The target YAML uses the same schema as in-tree targets with these
plugin-relevant fields:

``%PLUGIN_ROOT%``
-----------------

A substitution token expanded to the plugin's root directory at YAML parse
time:

.. code-block:: yaml

    config:
      jit-mid-level-pipeline: "qubit-mapping{device=file(%PLUGIN_ROOT%/data/topology.txt)}"

    target-arguments:
      - key: device
        type: string
        default: "%PLUGIN_ROOT%/data/topology.txt"
        platform-arg: device

Use this to reference files shipped inside the plugin without hard-coding
install paths.

``target-arguments``
--------------------

Declares per-target parameters that surface as CLI flags for ``nvq++`` and
keyword arguments for ``cudaq.set_target()``:

.. code-block:: yaml

    target-arguments:
      - key: api-key
        required: true
        type: string
        platform-arg: api_key
        help-string: "API key for the backend."
      - key: shots
        required: false
        type: integer
        platform-arg: shots
        help-string: "Default shot count."

These appear as ``--my-backend-api-key <value>`` on the command line and
``cudaq.set_target("my-backend", api_key=...)`` in Python.


Building with ``CUDAQ_EXTERNAL_PROJECTS``
==========================================

During development, build your plugin inside the CUDA-Q source tree without
modifying any in-tree files:

.. code-block:: bash

    cmake -B build \
      -DCUDAQ_EXTERNAL_PROJECTS="my-backend" \
      -DCUDAQ_EXTERNAL_MY_BACKEND_SOURCE_DIR=/path/to/my-backend

    ninja -C build cudaq-serverhelper-my-backend   # or cudaq-qpu-my-backend

The plugin output lands in ``build/external/my-backend/`` with the standard
``targets/``, ``lib/``, and packaging files ready to use.

Multiple plugins can be built together:

.. code-block:: bash

    cmake -B build \
      -DCUDAQ_EXTERNAL_PROJECTS="foo;bar" \
      -DCUDAQ_EXTERNAL_FOO_SOURCE_DIR=/path/to/foo \
      -DCUDAQ_EXTERNAL_BAR_SOURCE_DIR=/path/to/bar


Python Packaging
================

A plugin ships as a standard Python package with a ``cudaq.backends`` entry
point that makes it discoverable at ``import cudaq`` time. The package includes
the target YAML and shared library, allowing the plugin author to build a wheel
and publish it to a package index or distribute it directly. End users can then
install that package into their own CUDA-Q environments without building the
plugin from source.

Because the package contains a native shared library, it is platform-specific.
Plugin authors must build and distribute a separate package for each operating
system and architecture required by their users. When distributing wheels,
ensure that each wheel has the platform tag matching the platform targeted by
its shared library; do not distribute the package as a platform-agnostic wheel.

``pyproject.toml``
------------------

.. code-block:: toml

    [build-system]
    requires = ["setuptools>=64", "wheel"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "my-backend-cudaq"
    version = "0.1.0"
    description = "My Backend plugin for CUDA-Q."
    requires-python = ">=3.9"

    [project.entry-points."cudaq.backends"]
    my-backend = "my_backend_cudaq:register"

    [tool.setuptools]
    packages = ["my_backend_cudaq"]

    [tool.setuptools.package-dir]
    "my_backend_cudaq" = "."

    [tool.setuptools.package-data]
    "my_backend_cudaq" = ["targets/*.yml", "lib/*"]

The critical piece is the ``[project.entry-points."cudaq.backends"]`` section.
The key (``my-backend``) is a free-form identifier; the value points to the
``register()`` function in your package.

``__init__.py``
---------------

.. code-block:: python

    from importlib.resources import files


    def register():
        import cudaq
        cudaq.register_backend_path(str(files(__name__)))

When ``import cudaq`` runs, it discovers all ``cudaq.backends`` entry points
and calls each one. Your ``register()`` function calls
``cudaq.register_backend_path()`` with the package root, which scans
``targets/`` and makes your YAML-defined targets available.

If your entry point raises an exception, CUDA-Q logs a warning with the
entry-point name and traceback and continues — other plugins still load.

``__main__.py`` (``--install-nvqpp`` hook)
------------------------------------------

.. code-block:: python

    PACKAGE_NAME = "my_backend_cudaq"
    PLUGIN_NAME = "my-backend"


    def main() -> int:
        from cudaq.plugins import install_plugin_for_nvqpp_main
        return install_plugin_for_nvqpp_main(PACKAGE_NAME, PLUGIN_NAME)


    if __name__ == "__main__":
        raise SystemExit(main())

This lets users run ``python -m my_backend_cudaq --install-nvqpp`` to symlink
the installed package into the ``nvq++`` user plugin scope.


Installing the Plugin for End Users
===================================

In this section, "installation" means installation by an end user who wants to
use the target. It is separate from the preceding build and packaging steps
performed by the plugin author. The steps an end user follows depend on how the
plugin is distributed and whether they access the target from Python or C++:

``pip install`` (Python — zero config)
--------------------------------------

.. code-block:: bash

    pip install my-backend-cudaq

This installs the Python package previously published or otherwise distributed
by the plugin author. After installation in the end user's Python environment,
``import cudaq`` discovers the entry point automatically. No further
configuration is needed — ``cudaq.set_target("my-backend")`` works immediately.

``--install-nvqpp`` (make visible to ``nvq++``)
-----------------------------------------------

The Python entry-point mechanism only fires inside a Python process. To make
the Python package that the end user installed above visible to the ``nvq++``
compiler driver as well:

.. code-block:: bash

    python -m my_backend_cudaq --install-nvqpp

This symlinks the installed package directory into the user plugin scope.
After this, ``nvq++ --target=my-backend`` resolves the target.

``cudaq-install-plugin`` (C++-only workflows)
----------------------------------------------

For plugins distributed without Python packaging (built from source,
distributed as a tarball):

.. code-block:: bash

    # Install to user scope (default, no sudo required)
    cudaq-install-plugin /path/to/my-backend

    # Install to system scope (shared across users)
    sudo cudaq-install-plugin --system /path/to/my-backend

    # Copy instead of symlink
    cudaq-install-plugin --copy /path/to/my-backend

    # List installed plugins
    cudaq-install-plugin --list

    # Remove a plugin
    cudaq-install-plugin --uninstall my-backend


Discovery Mechanics
===================

``nvq++`` target resolution
---------------------------

When ``nvq++ --target=my-backend`` is invoked, the compiler resolves the
target YAML in this order:

1. **In-tree**: ``${install_dir}/targets/my-backend.yml``
2. **User scope**: ``${CUDAQ_PLUGIN_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/cudaq/plugins}/*/targets/my-backend.yml``
3. **System scope**: ``${install_dir}/plugins/*/targets/my-backend.yml``

The first match wins. User-scope plugins take precedence over system-scope.

When a plugin target is resolved, ``nvq++`` automatically adds the plugin's
``lib/`` directory to the linker search path (``-L``) and runtime path
(``-Wl,-rpath``).

Python target resolution
------------------------

At ``import cudaq``, the runtime:

1. Scans ``${install_dir}/targets/`` (in-tree targets)
2. Calls each ``cudaq.backends`` entry point, which registers additional
   target directories via ``cudaq.register_backend_path()``

After this, ``cudaq.set_target("my-backend")`` works for any registered target.

Environment variables
---------------------

.. list-table::
   :header-rows: 1

   * - Variable
     - Purpose
   * - ``CUDAQ_PLUGIN_ROOT``
     - Overrides the user-scope plugin directory for ``nvq++`` and
       ``cudaq-install-plugin``. Useful for CI, sandboxed builds, or
       shared filesystems.
   * - ``XDG_DATA_HOME``
     - Standard XDG variable. When ``CUDAQ_PLUGIN_ROOT`` is unset, the
       user scope defaults to ``${XDG_DATA_HOME}/cudaq/plugins``.


Reference Plugins
=================

CUDA-Q ships a reference plugin under ``docs/sphinx/examples/plugins/``
that demonstrates the full plugin lifecycle:

.. list-table::
   :header-rows: 1

   * - Plugin
     - Shape
     - What it demonstrates
   * - `mock_rest <https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/examples/plugins/mock_rest>`_
     - REST
     - ``ServerHelper`` subclass, ``remote_rest`` QPU, mock server testing

Use this as a starter template for new plugins. It includes a complete
build configuration, Python packaging, lit tests, and documentation.


Quick-Start Checklist
=====================

.. code-block:: text

    □ Implement ServerHelper subclass
    □ Create targets/<name>.yml with target configuration
    □ Create CMakeLists.txt (build with CUDAQ_EXTERNAL_PROJECTS)
    □ Add pyproject.toml with cudaq.backends entry point
    □ Add __init__.py with register() function
    □ Add __main__.py with --install-nvqpp hook
    □ Add tests/ with at least one end-to-end test
    □ Verify: pip install . && python -c "import cudaq; assert cudaq.has_target('<name>')"
    □ Verify: python -m <pkg> --install-nvqpp && nvq++ --target=<name> ...

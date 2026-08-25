# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import argparse
import os
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Optional, Sequence


def user_plugin_scope() -> Path:
    if os.environ.get("CUDAQ_PLUGIN_ROOT"):
        return Path(os.environ["CUDAQ_PLUGIN_ROOT"]).expanduser()
    if os.environ.get("XDG_DATA_HOME"):
        return Path(
            os.environ["XDG_DATA_HOME"]).expanduser() / "cudaq" / "plugins"
    return Path.home() / ".local" / "share" / "cudaq" / "plugins"


def install_plugin_for_nvqpp(package_name: str,
                             plugin_name: str,
                             *,
                             copy: bool = False) -> Path:
    package_root = Path(str(files(package_name))).resolve()
    destination = user_plugin_scope() / plugin_name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(package_root, destination, symlinks=True)
    else:
        destination.symlink_to(package_root, target_is_directory=True)
    return destination


def install_plugin_for_nvqpp_main(package_name: str,
                                  plugin_name: str,
                                  argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install this CUDA-Q plugin for nvq++ discovery.")
    parser.add_argument(
        "--install-nvqpp",
        action="store_true",
        help="link this package into the user nvq++ plugin scope")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="copy instead of symlinking when installing for nvq++")
    args = parser.parse_args(argv)

    if not args.install_nvqpp:
        parser.print_help()
        return 1

    try:
        print(
            install_plugin_for_nvqpp(package_name, plugin_name, copy=args.copy))
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc
    return 0

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq


def getSVGstring(kernel, *args):
    from subprocess import check_output, STDOUT
    from tempfile import TemporaryDirectory

    latex_string = cudaq.draw("latex", kernel, *args)
    with TemporaryDirectory() as tmpdirname:
        with open(tmpdirname + "/cudaq-trace.tex", "w") as f:
            f.write(latex_string)
        # this needs `latex` and `quantikz` to be installed, e.g. through `apt install texlive-latex-extra`
        check_output(["latex", "cudaq-trace"], cwd=tmpdirname, stderr=STDOUT)
        check_output(["dvisvgm", "cudaq-trace"], cwd=tmpdirname, stderr=STDOUT)
        with open(tmpdirname + "/cudaq-trace.svg", "rb") as f:
            return str(f.read(), encoding="utf-8")


def displaySVG(kernel, *args):
    from IPython.display import SVG, display

    display(SVG(data=getSVGstring(kernel, *args)))

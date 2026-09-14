# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Minimal local entry points. CI calls sphinx-build directly with the same
# arguments; see README.md.
SPHINXOPTS  ?= -W
SPHINXBUILD ?= sphinx-build
SOURCEDIR   = source
BUILDDIR    ?= _build

.PHONY: html clean

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

clean:
	@rm -rf "$(BUILDDIR)"

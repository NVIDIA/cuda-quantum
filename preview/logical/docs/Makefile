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

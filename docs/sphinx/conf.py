# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'NVIDIA CUDA-Q'
copyright = '2026, NVIDIA Corporation & Affiliates'
author = 'NVIDIA Corporation & Affiliates'

# The version info for the project you're documenting, acts as replacement for
# |version| used in various places throughout the docs.

# The short X.Y version.
version = os.getenv("CUDA_QUANTUM_VERSION", "latest")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # 'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.autodoc',  # to get documentation from python doc comments
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.doctest',  # test example codes in docs
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',  # support google/numpy style docstrings
    #'sphinx.ext.linkcode',
    'sphinx_reredirects',
    'breathe',
    'sphinx_tags',
    'enum_tools.autoenum',  # for pretty-print Python enums
    'myst_parser',  # for including markdown files
    'sphinx_inline_tabs',  # showing code blocks in multiple languages
    'nbsphinx',  # for supporting jupyter notebooks
    'sphinx_copybutton',  # allows for copy/paste of code cells
    "sphinx_gallery.load_style",
    "IPython.sphinxext.ipython_console_highlighting",
]

html_static_path = ['_static']
html_js_files = ['filter.js']
# Disable source links in the HTML output.
html_show_sourcelink = False

# Enable dynamic filtering in sphinx-needs
needs_include_needs = True  # Enable needs processing
needs_types = [{
    'directive': 'notebook',
    'title': 'Notebook',
    'prefix': 'NB_',
    'color': '#BFD8D2',
    'style': 'node'
}]

tags_create_tags = True  # Automatically generate tag pages
nbsphinx_allow_errors = False

needs_extra_options = ['description', 'nb_links']

nbsphinx_thumbnails = {
    # Default thumbnail if the notebook does not define a cell tag to specify the thumbnail.
    # See also: https://nbsphinx.readthedocs.io/en/latest/subdir/gallery.html
    '**': '_static/cuda_quantum_icon.svg',
    'examples/python/tutorials/hybrid_qnns': '_images/hybrid.png',
    'examples/python/tutorials/multi_gpu_workflows': '_images/circsplit.png',
}

imgmath_latex_preamble = r'\usepackage{braket}'

imgmath_image_format = 'svg'
imgmath_font_size = 14
#imgmath_dvipng_args = ['-gamma', '1.5', '-D', '110', '-bg', 'Transparent']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '**/_*', '.DS_Store', 'examples/python/building_kernels.ipynb',
    'examples/python/measuring_kernels.ipynb',
    'examples/python/executing_kernels.ipynb', 'examples/python/operators.ipynb'
]

#redirect links
redirects = {"backends/dynamics": "../dynamics.html"}

# Generate OpenAPI spec for the REST API
import ruamel.yaml
import json

current_path = os.path.dirname(os.path.realpath(__file__))
# YAML file defines the spec
yaml_file = current_path + '/api/rest/.openapi.yaml'
yaml = ruamel.yaml.YAML(typ='safe')
with open(yaml_file) as fpi:
    data = yaml.load(fpi)
json_spec = json.dumps(data, indent=2)

# Auto-generate the HTML file with the embedded OpenAPI spec.
html_template_file = current_path + '/_templates/openapi.html'
with open(html_template_file, 'r') as file:
    filedata = file.read()
# Replace the placeholder with the spec
filedata = filedata.replace("##SPEC_JSON##", json_spec)
html_output_file = current_path + '/Output/openapi.html'
os.makedirs(os.path.dirname(html_output_file), exist_ok=True)
# Write the file out again
with open(html_output_file, 'w') as file:
    file.write(filedata)
# Add this to `html_extra_path` so that it is copied to the deployed docs.
html_extra_path = ['Output/openapi.html']

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = 'code'  # NOTE: the following may be a better choice to error on the side of flagging anything that is referenced but but not declared
#default_role = 'cpp:any' # see https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'lightbulb'

# autosummary is buggy: this must be py instead of cpp so that the domain setting
# can be propagated to the autogen'd rst files.
# primary_domain = 'py'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": False,
    "prev_next_buttons_location": "both",
    "style_nav_header_background":
        "#76b900"  # Set upper left search bar to NVIDIA green
}

html_css_files = ['_static/cudaq_override.css', 'custom.css']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'cudaqDoc'


def setup(app):
    app.add_css_file('cudaq_override.css')


# -- Options for BREATHE -------------------------------------------------

breathe_projects = {"cudaq": "_doxygen/xml"}

breathe_default_project = "cudaq"

breathe_show_enumvalue_initializer = True

# -- Other options -------------------------------------------------

autosummary_generate = True

# This is unfortunately not sufficient for docs generation, due to the use of Union type;
# see also https://github.com/sphinx-doc/sphinx/issues/11211.
# autodoc_mock_imports = ['cuquantum', 'cupy']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

redirects = {"versions": "../latest/releases.html"}

nitpick_ignore = [
    ('cpp:identifier', 'GlobalRegisterName'),
    ('cpp:identifier', 'CountsDictionary::iterator'),
    ('cpp:identifier', 'CountsDictionary::const_iterator'),
    ('cpp:identifier', 'State'),
    ('cpp:identifier', 'pauli'),
    ('cpp:identifier', 'Job'),
    ('cpp:identifier', 'mlir'),
    ('cpp:identifier', 'mlir::Value'),
    ('cpp:identifier', 'mlir::Type'),
    ('cpp:identifier', 'mlir::MLIRContext'),
    ('cpp:identifier', 'mlir::ImplicitLocOpBuilder'),
    ('cpp:identifier', 'BinarySymplecticForm'),
    ('cpp:identifier', 'CountsDictionary'),
    ('cpp:identifier', 'QuakeValueOrNumericType'),
    ('cpp:identifier', 'cudaq::ctrl'),
    ('py:class', 'function'),
    ('py:class', 'type'),
    ('py:class', 'numpy.ndarray[]'),
    # FIXME: remove these after adding proper documentation
    # (also reexamine why some of the ones above are ignored)
    ('py:class', 'cudaq::sum_op<cudaq::spin_handler>'),
    ('py:class', 'SpinOperatorTerm'),
    ('cpp:identifier', 'cudaq::detail'),
    ('cpp:identifier', 'cudaq::detail::EigenSparseMatrix'),
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2
copybutton_copy_empty_lines = False

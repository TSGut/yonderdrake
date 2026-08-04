"""Sphinx configuration for the Yonderdrake documentation."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

project = "Yonderdrake"
author = "Timon S. Gutleb and contributors"
copyright = "2026, Timon S. Gutleb and contributors"

try:
    release = package_version("yonderdrake")
except PackageNotFoundError:
    release = "development"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
master_doc = "index"
exclude_patterns = [
    "theory/time-derivatives.md",
    "theory/diffusive-representations.md",
    "theory/direct-time-methods.md",
    "theory/exponential-memory.md",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = True

intersphinx_mapping = {
    "firedrake": ("https://www.firedrakeproject.org/", None),
    "petsc": ("https://petsc.org/release/", None),
    "python": ("https://docs.python.org/3/", None),
    "ufl": ("https://docs.fenicsproject.org/ufl/main/", None),
}

html_theme = "furo"
html_title = "Yonderdrake documentation"
html_baseurl = "https://timon.gutleb.com/yonderdrake/"
html_logo = "_static/yonderdrake-logo.png"
html_favicon = "_static/yonderdrake-favicon.png"
html_static_path = ["_static"]
html_css_files = ["yonderdrake.css"]
html_show_sourcelink = True
html_show_sphinx = False
html_use_index = False
html_domain_indices = False
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#146c6e",
        "color-brand-content": "#146c6e",
    },
    "dark_css_variables": {
        "color-brand-primary": "#79c8b8",
        "color-brand-content": "#79c8b8",
    },
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/TSGut/yonderdrake/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "top_of_page_buttons": ["view", "edit"],
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# Publisher and host endpoints that reject automated requests. Each target was
# checked manually and resolves in a browser; only the robot access is blocked.
linkcheck_ignore = [
    r"https://brainweb\.bic\.mni\.mcgill\.ca/.*",
    r"https://doi\.org/10\.1115/1\.1448322",
    # epubs.siam.org answers every automated request with 403
    r"https://doi\.org/10\.1137/.*",
    r"https://doi\.org/10\.3390/math10081245",
    r"https://doi\.org/10\.1145/2566630",
]

"""Sphinx configuration for the Spatialize Examples documentation gallery."""

import shutil
import sys
from datetime import datetime
from pathlib import Path

# -- Path setup --------------------------------------------------------------

DOCS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = DOCS_DIR.parent
EXAMPLES_SRC = REPO_ROOT / "examples"
EXAMPLES_DST = DOCS_DIR / "examples"

sys.path.insert(0, str(DOCS_DIR))
import gallery  # noqa: E402  (docs/gallery.py: the manifest docs/notebooks.yaml)


def _sync_example_sources() -> None:
    """Copy the notebooks of the manifest (and their sibling assets) into the Sphinx source tree,
    overwriting on every build so the docs build never drifts from ``examples/``. Sphinx cannot
    include files that live outside its source directory, so this makes the copies idempotent and
    keeps ``docs/examples/`` out of version control (see docs/.gitignore). The landing page
    ``index.md`` is written from the same manifest.
    """
    for nb in gallery.notebooks():
        src = EXAMPLES_SRC / f"{nb['path']}.ipynb"
        dst = EXAMPLES_DST / f"{nb['path']}.ipynb"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        for asset in nb.get("assets", []):
            shutil.copyfile(src.parent / asset, dst.parent / asset)
    gallery.write_index(DOCS_DIR / "index.md")


_sync_example_sources()

# -- Project information -----------------------------------------------------

project = "Spatialize Examples"
author = "ALGES Lab"
copyright = f"{datetime.now().year}, {author}"

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
]

# Enables cross-references from notebook prose into the main spatialize API
# docs (e.g. linking a mention of `esi_griddata` to its reference page), now
# that the main project's RTD site is live.
intersphinx_mapping = {
    "spatialize": ("https://spatialize.readthedocs.io/en/latest/", None),
}

suppress_warnings = []

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

templates_path = ["_templates"]

# -- MyST / myst-nb configuration ---------------------------------------------

myst_enable_extensions = ["colon_fence", "dollarmath"]

nb_execution_mode = "off"

# -- HTML output ---------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
# Drop Sphinx's default "<project> documentation" long form in the
# sidebar/browser-tab title. No version concept for this notebook gallery
# (unlike the main spatialize docs), so just "Spatialize".
html_title = "Spatialize"

# This is a standalone Sphinx project (an RTD subproject of the main
# spatialize docs), so its navbar is independent by default and has no
# built-in way back to the parent site. Top navbar links (Home,
# Documentation, Examples) are rendered by docs/_templates/navbar-nav.html,
# which overrides the theme's stock component (see that file for why: it
# avoids flooding the header with a link per hidden-toctree entry, and uses
# `pathto()` for this site's own "Examples" self-link so it resolves
# correctly at any page depth without leaving this site — Home/Documentation
# stay real absolute URLs since they point at a genuinely different Sphinx
# project/host).
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/alges/spatialize-examples",
            "icon": "fa-brands fa-github",
        },
    ],
}

# Every page in this project is a leaf (a single notebook, or the landing
# index) with no child pages in its local toctree, so pydata_sphinx_theme's
# left "Section Navigation" sidebar (sidebar-nav-bs) always rendered empty —
# just a heading with nothing under it. Drop it from every page; the
# in-page table of contents (right-hand "On this page" panel, powered by
# the notebook's own headings) already provides real, working navigation.
html_sidebars = {
    "**": [],
}

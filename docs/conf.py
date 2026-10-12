"""Sphinx configuration for the Spatialize Examples documentation gallery."""

import shutil
from datetime import datetime
from pathlib import Path

# -- Path setup --------------------------------------------------------------

DOCS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = DOCS_DIR.parent
EXAMPLES_SRC = REPO_ROOT / "examples"
EXAMPLES_DST = DOCS_DIR / "examples"

# Explicit list of the 16 tracked notebooks to publish, as (category, name)
# pairs. Keep this list in sync with the toctree in index.md.
#
# NOTE: examples/esi_implementations/esi_kriging.ipynb and
# examples/introductory/esi_overview.py are intentionally EXCLUDED (untracked
# work-in-progress files) and must never be added here.
NOTEBOOKS = [
    ("introductory", "esi_overview"),
    ("introductory", "ess_overview"),
    ("introductory", "spa_overview"),
    ("esi_fundamentals", "esi_griddata"),
    ("esi_fundamentals", "esi_nongriddata"),
    ("esi_fundamentals", "esi_hparams_search"),
    ("esi_fundamentals", "esi_pareto_optimization"),
    ("esi_fundamentals", "esi_precision"),
    ("esi_implementations", "esi_2.5d"),
    ("esi_implementations", "adaptive_esi_2d"),
    ("esi_implementations", "adaptive_esi_2.5d"),
    ("esi_implementations", "categorical_esi"),
    ("how_to", "custom_esi_precision"),
    ("utilities", "empirical_tools"),
    ("utilities", "evaluation_tools"),
    ("utilities", "visualization_tools"),
]

# Sibling non-notebook assets (e.g. images) that notebooks reference via
# relative paths, and therefore must be copied alongside them.
EXTRA_ASSETS = [
    ("introductory", "ESI_procedure.png"),
]


def _sync_example_sources() -> None:
    """Copy the published notebooks (and their sibling assets) into the
    Sphinx source tree, overwriting on every build so the docs build never
    drifts from ``examples/``. Sphinx cannot include files that live outside
    its source directory, so this makes the copies idempotent and keeps
    ``docs/examples/`` out of version control (see docs/.gitignore).
    """
    for category, name in NOTEBOOKS:
        src = EXAMPLES_SRC / category / f"{name}.ipynb"
        dst_dir = EXAMPLES_DST / category
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst_dir / f"{name}.ipynb")

    for category, filename in EXTRA_ASSETS:
        src = EXAMPLES_SRC / category / filename
        dst_dir = EXAMPLES_DST / category
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst_dir / filename)


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

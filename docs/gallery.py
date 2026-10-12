"""The gallery manifest (docs/notebooks.yaml), shared by docs/conf.py and
docs/scripts/generate_thumbnails.py so that the list of notebooks is kept in one place."""
from pathlib import Path

import yaml

DOCS_DIR = Path(__file__).parent.resolve()
MANIFEST = DOCS_DIR / "notebooks.yaml"


def manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text())


def notebooks() -> list:
    """Every notebook entry, in reading order."""
    return [nb for section in manifest()["sections"] for nb in section["notebooks"]]


def thumbnail(path: str) -> str:
    """File name of a notebook's thumbnail under _static/thumbnails/ (unique per path)."""
    return path.replace("/", "__") + ".png"


def write_index(target: Path) -> None:
    """Write the landing page: one grid of cards per section, then a hidden toctree."""
    m = manifest()
    lines = ["# Spatialize Examples", "", m["intro"].strip(), ""]
    for section in m["sections"]:
        lines += [f"## {section['title']}", "", "::::{grid} 1 2 2 3", ":gutter: 3", ""]
        for nb in section["notebooks"]:
            lines += [f":::{{grid-item-card}} {nb['title']}",
                      f":link: examples/{nb['path']}",
                      ":link-type: doc",
                      f":img-top: _static/thumbnails/{thumbnail(nb['path'])}",
                      "",
                      nb["description"],
                      ":::",
                      ""]
        lines += ["::::", ""]
    lines += ["```{toctree}", ":maxdepth: 1", ":hidden:", ""]
    lines += [f"examples/{nb['path']}" for nb in notebooks()]
    lines += ["```", ""]
    text = "\n".join(lines)
    if not target.exists() or target.read_text() != text:   # an unchanged page keeps Sphinx's cache
        target.write_text(text)

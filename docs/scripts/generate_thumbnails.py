#!/usr/bin/env python3
"""Generate gallery thumbnails for the docs build.

For each notebook of the manifest (docs/notebooks.yaml), extracts the last image/png output
found in any cell and writes it to ``docs/_static/thumbnails/`` under the name given by
``gallery.thumbnail``.
Notebooks with no image output get a copy of a generic placeholder thumbnail.

Idempotent: safe to re-run, always overwrites with fresh output.
"""

import base64
import sys
from pathlib import Path

import nbformat
from PIL import Image

DOCS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = DOCS_DIR.parent
EXAMPLES_SRC = REPO_ROOT / "examples"
THUMBS_DIR = DOCS_DIR / "_static" / "thumbnails"
PLACEHOLDER_PATH = THUMBS_DIR / "_placeholder.png"

sys.path.insert(0, str(DOCS_DIR))
import gallery  # noqa: E402  (docs/gallery.py: the manifest docs/notebooks.yaml)


def _ensure_placeholder() -> None:
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    if not PLACEHOLDER_PATH.exists():
        img = Image.new("RGB", (400, 300), color=(200, 200, 200))
        img.save(PLACEHOLDER_PATH)
        print(f"Created placeholder thumbnail: {PLACEHOLDER_PATH}")


def _last_png_output(notebook_path: Path) -> bytes | None:
    nb = nbformat.read(notebook_path, as_version=4)
    last_png = None
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data:
                last_png = data["image/png"]
    if last_png is None:
        return None
    return base64.b64decode(last_png)


def main() -> None:
    _ensure_placeholder()

    generated, placeholders = [], []
    for nb in gallery.notebooks():
        name = nb["path"]
        notebook_path = EXAMPLES_SRC / f"{name}.ipynb"
        dest = THUMBS_DIR / gallery.thumbnail(name)

        png_bytes = _last_png_output(notebook_path)
        if png_bytes is not None:
            dest.write_bytes(png_bytes)
            generated.append(name)
        else:
            dest.write_bytes(PLACEHOLDER_PATH.read_bytes())
            placeholders.append(name)

    print(f"Real thumbnails generated ({len(generated)}): {', '.join(generated) or 'none'}")
    print(f"Placeholder thumbnails used ({len(placeholders)}): {', '.join(placeholders) or 'none'}")


if __name__ == "__main__":
    main()

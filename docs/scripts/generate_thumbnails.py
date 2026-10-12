#!/usr/bin/env python3
"""Generate gallery thumbnails for the docs build.

For each of the 16 published notebooks, extracts the last image/png output
found in any cell and writes it to ``docs/_static/thumbnails/<name>.png``.
Notebooks with no image output get a copy of a generic placeholder thumbnail.

Idempotent: safe to re-run, always overwrites with fresh output.
"""

import base64
from pathlib import Path

import nbformat
from PIL import Image

DOCS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = DOCS_DIR.parent
EXAMPLES_SRC = REPO_ROOT / "examples"
THUMBS_DIR = DOCS_DIR / "_static" / "thumbnails"
PLACEHOLDER_PATH = THUMBS_DIR / "_placeholder.png"

# Keep in sync with the NOTEBOOKS list in docs/conf.py.
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
    for category, name in NOTEBOOKS:
        notebook_path = EXAMPLES_SRC / category / f"{name}.ipynb"
        dest = THUMBS_DIR / f"{name}.png"

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

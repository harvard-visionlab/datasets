"""Convert a jupytext percent-format .py into an .ipynb (no jupytext dependency): `python scripts/py2nb.py notebooks/x.py`."""
import re
import sys
from pathlib import Path

import nbformat

src = Path(sys.argv[1]); dst = src.with_suffix(".ipynb")
text = src.read_text()
text = re.sub(r"\A# ---\n.*?# ---\n", "", text, flags=re.S)             # drop the jupytext header
cells = []
for block in re.split(r"^# %%", text, flags=re.M):
    if not block.strip():
        continue
    head, _, body = block.partition("\n")
    body = body.strip("\n")
    if "[markdown]" in head:
        md = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(md))
    else:
        cells.append(nbformat.v4.new_code_cell(body))
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}})
nbformat.write(nb, dst)
print(f"{dst}: {len(cells)} cells")

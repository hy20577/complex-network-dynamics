"""Make the links written inside the tutorial notebooks work on the site.

In a notebook, another tutorial is linked by its file name
(`02_network_types.ipynb`) and an API page by `../api/datasets.md`, which is
what works in Jupyter. mkdocs does not rewrite links inside notebooks, and
each tutorial is served from its own folder (`tutorials/01_.../`), so on the
site those links would be broken. This hook points them to the built pages.
"""
import re


def on_page_content(html, page, config, files):
    if not page.file.src_path.endswith(".ipynb"):
        return html
    html = re.sub(r'href="(\d\d_\w+)\.ipynb(#[^"]*)?"', lambda m: f'href="../{m.group(1)}/{m.group(2) or ""}"', html)
    html = re.sub(r'href="\.\./(api/\w+|publications|index)\.md(#[^"]*)?"',
                  lambda m: f'href="../../{m.group(1)}/{m.group(2) or ""}"', html)
    return html

"""Add a content hash to the extra stylesheets, so browsers load a changed file
at once instead of keeping the cached copy (GitHub Pages caches for 10 minutes)."""
import hashlib
from pathlib import Path


def on_config(config):
    docs = Path(config["docs_dir"])
    stamped = []
    for css in config["extra_css"]:
        path = docs / css
        if path.is_file():
            digest = hashlib.md5(path.read_bytes()).hexdigest()[:8]
            css = f"{css}?v={digest}"
        stamped.append(css)
    config["extra_css"] = stamped
    return config

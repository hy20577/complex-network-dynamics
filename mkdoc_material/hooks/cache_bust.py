"""Add a content hash to the local extra stylesheets and scripts, so browsers
load a changed file at once instead of keeping the cached copy (GitHub Pages
caches for 10 minutes)."""
import hashlib
from pathlib import Path


def _stamp(docs, entry):
    path = docs / str(entry)
    if not path.is_file():
        return entry  # Remote URLs (the MathJax CDN) are left as they are
    digest = hashlib.md5(path.read_bytes()).hexdigest()[:8]
    stamped = f"{entry}?v={digest}"
    if isinstance(entry, str):
        return stamped
    entry.path = stamped  # extra_javascript items can be script objects
    return entry


def on_config(config):
    docs = Path(config["docs_dir"])
    config["extra_css"] = [_stamp(docs, css) for css in config["extra_css"]]
    config["extra_javascript"] = [_stamp(docs, js) for js in config["extra_javascript"]]
    return config

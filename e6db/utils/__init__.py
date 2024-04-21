"Python only utils (no dependencies)"
from pathlib import Path
import gzip
import json

tag_categories = [
    "general",
    "artist",
    "copyright",
    "character",
    "species",
    "invalid",
    "meta",
    "lore",
    "pool",
]
tag_category2id = {v: k for k, v in enumerate(tag_categories)}
tag_categories_colors = [
    "#b4c7d9",
    "#f2ac08",
    "#d0d",
    "#0a0",
    "#ed5d1f",
    "#ff3d3d",
    "#fff",
    "#282",
    "wheat",
]
tag_categories_alt_colors = [
    "#2e76b4",
    "#fbd67f",
    "#ff5eff",
    "#2bff2b",
    "#f6b295",
    "#ffbdbd",
    "#666",
    "#5fdb5f",
    "#d0b27a",
]


def load_tags(data_dir):
    data_dir = Path(data_dir)
    with gzip.open(data_dir / "tags.txt.gz", "rt") as fd:
        idx2tag = fd.read().split("\n")
        if not idx2tag[-1]:
            idx2tag = idx2tag[:-1]
    with gzip.open(data_dir / "tag2idx.json.gz", "rb") as fp:
        tag2idx = json.load(fp)
    with gzip.open(data_dir / "tags_categories.bin.gz", "rb") as fp:
        tag_categories = fp.read()
    return tag2idx, idx2tag, tag_categories

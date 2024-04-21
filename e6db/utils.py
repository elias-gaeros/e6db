from pathlib import Path
import gzip
import json

# FIXME: split this module by dependencies
import numpy as np
import scipy.sparse
import torch
import polars as pl
from polars import col

id2tagcat = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "species",
    6: "invalid",
    7: "meta",
    8: "lore",
}
tagcat2id = {v: k for k, v in id2tagcat.items()}


def load_tags(data_dir):
    data_dir = Path(data_dir)
    with gzip.open(data_dir / "tags.txt.gz", 'rt') as fd:
        idx2tag = fd.read().split("\n")
        if not idx2tag[-1]:
            idx2tag = idx2tag[:-1]
    with gzip.open(data_dir / "tag2idx.json.gz", "rb") as fp:
        tag2idx = json.load(fp)
    return tag2idx, idx2tag


def filter_tag_list(column, vocab_size):

    return col(column).list.eval(pl.element().filter(pl.element() < vocab_size))


def tags_to_csr(df_posts, column="stripped_tags"):
    from polars import col

    indices = df_posts[column].explode().drop_nulls().to_numpy()
    indptr = df_posts.select(
        offset=col(column).list.len().cum_sum().shift(1, fill_value=0)
    )["offset"].to_numpy()
    return indices, indptr


def tags_to_scipy_csr(df_posts, column="stripped_tags", vocab_size=None):
    indices, indptr = tags_to_csr(df_posts, column=column)
    indptr = np.r_[indptr, len(indices)]
    R = scipy.sparse.csr_array(
        (np.ones(indices.shape[0], dtype=np.uint8), indices, indptr),
        shape=(len(df_posts), vocab_size or indices.max() + 1),
    )
    return R


def tags_to_tensors(df_posts, column="stripped_tags", device=None, dtype=torch.int32):
    indices, indptr = tags_to_csr(df_posts, column=column)
    tags = torch.tensor(indices, dtype=dtype, device=device)
    offsets = torch.tensor(indptr, dtype=dtype, device=device)
    return tags, offsets

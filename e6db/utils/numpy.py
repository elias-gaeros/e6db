import numpy as np
from . import load_tags as py_load_tags

def tags_to_scipy_csr(df_posts, column="stripped_tags", vocab_size=None):
    from .polars import tags_to_csr
    import scipy.sparse

    indices, indptr = tags_to_csr(df_posts, column=column)
    indptr = np.r_[indptr, len(indices)]
    R = scipy.sparse.csr_array(
        (np.ones(indices.shape[0], dtype=np.uint8), indices, indptr),
        shape=(len(df_posts), vocab_size or indices.max() + 1),
    )
    return R


def load_tags(data_dir):
    tag2idx, idx2tag, tag_categories = py_load_tags(data_dir)
    idx2tag = np.array(idx2tag)
    tag_categories = np.frombuffer(tag_categories, dtype=np.uint8)
    return tag2idx, idx2tag, tag_categories

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


def tag_rank_to_freq(rank: np.ndarray) -> np.ndarray:
    """Approximate the frequency of a tag given its rank"""
    return np.exp(26.4284 * np.tanh(2.93505 * rank ** (-0.136501)) - 11.492)


def tag_freq_to_rank(freq: np.ndarray) -> np.ndarray:
    """Approximate the rank of a tag given its frequency"""
    log_freq = np.log(freq)
    return np.exp(
        -7.57186
        * (0.0465456 * log_freq - 1.24326)
        * np.log(1.13045 - 0.0720383 * log_freq)
        + 12.1903
    )

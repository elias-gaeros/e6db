import polars as pl
from polars import col


def filter_tag_list(column, vocab_size):
    return col(column).list.eval(pl.element().filter(pl.element() < vocab_size))


def tags_to_csr(df_posts, column="stripped_tags"):
    indices = df_posts[column].explode().drop_nulls().to_numpy()
    indptr = df_posts.select(
        offset=col(column).list.len().cum_sum().shift(1, fill_value=0)
    )["offset"].to_numpy()
    return indices, indptr

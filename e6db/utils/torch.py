import torch


def tags_to_tensors(df_posts, column="stripped_tags", device=None, dtype=torch.int32):
    from .polars import tags_to_csr

    indices, indptr = tags_to_csr(df_posts, column=column)
    tags = torch.tensor(indices, dtype=dtype, device=device)
    offsets = torch.tensor(indptr, dtype=dtype, device=device)
    return tags, offsets

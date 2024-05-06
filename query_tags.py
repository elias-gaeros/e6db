#!/usr/bin/python
from pathlib import Path
import argparse

import numpy as np
import safetensors
from sklearn.decomposition import PCA

from e6db.utils.numpy import load_tags
from e6db.utils import (
    tag_category2id,
    tag_categories_colors,
    tag_freq_to_rank,
    tag_rank_to_freq,
)


def dothething(args):
    with safetensors.safe_open(
        args.data_dir / "implicit_tag_factors.safetensors", framework="numpy"
    ) as st:
        X = st.get_tensor("tag_factors")
    N_vocab = X.shape[0]

    tags2id, idx2tag, tag_categories = load_tags(args.data_dir)
    idx2tag = idx2tag[:N_vocab]
    tag_categories = tag_categories[:N_vocab]

    X /= np.linalg.norm(X, axis=1)[:, None]
    if args.first_pca and args.first_pca < X.shape[1]:
        pca = PCA(args.first_pca)
        Xt = pca.fit_transform(X)
        Xt /= np.linalg.norm(Xt, axis=1)[:, None]
        del X
    else:
        Xt = X

    sel_idxs = (tags2id.get(t, N_vocab) for t in args.tags)
    sel_idxs = np.array(sorted(i for i in sel_idxs if i < N_vocab))
    sel_tags = idx2tag[sel_idxs]
    print("Query tags:", " ".join(sel_tags))

    # Select neighboring tags similar to the input tags
    global_topk = args.global_topk
    top_k = args.topk
    if top_k is None:
        top_k = int(1.5 * global_topk / len(sel_idxs))
    rank_tresh = min(N_vocab, int(tag_freq_to_rank(args.min_frequency)))

    # Score and filter
    scores = Xt[:rank_tresh] @ Xt[sel_idxs].T
    scores[sel_idxs[sel_idxs < rank_tresh], :] = float("-inf")  # Mask self-matches
    if args.category:
        categories = [tag_category2id[cat] for cat in args.category]
        scores[~np.isin(tag_categories[:rank_tresh], categories), :] = float("-inf")

    # Per query top-k
    neigh_idxs = np.argpartition(-scores, top_k, axis=0)[:top_k]

    for i, t in enumerate(sel_tags):
        order = np.argsort(scores[neigh_idxs[:, i], i])[::-1]
        idxs = neigh_idxs[order[: args.display_topk], i]
        tag_list = " ".join(
            f"{idx2tag[i]} ({format_tagfreq(tag_rank_to_freq(i))})" for i in idxs
        )
        print(f"* {t} ({format_tagfreq(tag_rank_to_freq(sel_idxs[i]))}): {tag_list}")

    if not args.plot_out:
        return
    from matplotlib import pyplot as plt

    # Deduplicate, global top-k
    neigh_idxs = np.unique(neigh_idxs)
    scores = scores[neigh_idxs, :].sum(axis=1)
    if len(neigh_idxs) > global_topk:
        neigh_idxs = neigh_idxs[np.argpartition(-scores, global_topk)[:global_topk]]

    idxs = np.concatenate([sel_idxs, neigh_idxs])
    query_slice = slice(None, len(sel_idxs))
    target_slice = slice(len(sel_idxs), len(sel_idxs) + args.display_topk)
    colors = np.array(tag_categories_colors)[tag_categories[idxs]]

    # Local PCA
    X2 = Xt[idxs]
    del Xt
    X2 = X2 - X2.mean(0)
    X2 /= np.linalg.norm(X2, axis=1)[:, None]
    X2t = PCA(2).fit_transform(X2)[:, ::-1]

    f, ax = plt.subplots(figsize=(12, 12), facecolor="#152f56")
    ax.axis("off")

    dx = 0.01
    ax.scatter(*X2t[query_slice].T, c=colors[query_slice], linewidth=0, s=20)
    for t, pos, c in zip(
        idx2tag[idxs[query_slice]], X2t[query_slice, :], colors[query_slice]
    ):
        ax.annotate(t, tuple(pos + [dx * 0.5, dx * 0.25]), color=c)

    ax.scatter(
        *X2t[target_slice].T,
        c=colors[target_slice],
        linewidth=0,
        s=10,
        alpha=0.5,
    )
    for t, pos, c in zip(
        idx2tag[idxs[target_slice]], X2t[target_slice, :], colors[target_slice]
    ):
        ax.annotate(
            t,
            tuple(pos + [dx * 0.3, dx * 0.3 / 2]),
            color=c,
            fontsize=8,
            alpha=2 / 3,
        )

    f.tight_layout()
    if str(args.plot_out) == "-":
        plt.show(block=True)
    else:
        f.savefig(args.plot_out, facecolor="auto")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query similar tags and plots a local PCA.\nUse `-o -` to get an interactive plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "tags",
        metavar="<tag>",
        type=str,
        nargs="+",
        help="query tags (use underscore format)",
    )
    parser.add_argument(
        "-c",
        "--category",
        choices=tag_category2id.keys(),
        action="append",
        help="restrict the output to the specified tag category",
    )
    parser.add_argument(
        "-f",
        "--min_frequency",
        type=int,
        default=100,
        help="minimal number of posts tagged for a tag to be considered",
    )
    parser.add_argument(
        "-N",
        "--display-topk",
        type=int,
        default=24,
        help="set the number of neighboring tags to display",
    )
    parser.add_argument(
        "-n",
        "--global-topk",
        type=int,
        default=64,
        help="selects the global top-k neighbors for the local PCA",
    )
    parser.add_argument(
        "-k",
        "--topk",
        type=int,
        default=None,
        help="Number of neighbors to consider for each query tag. When not specified, is set to 1.5 * GLOBAL_TOPK / <number of query tags>",
    )
    parser.add_argument(
        "-d",
        "--first-pca",
        type=int,
        default=None,
        help="truncation rank for the global PCA meant to smooth all embeddings",
    )
    parser.add_argument(
        "-o",
        "--plot-out",
        type=Path,
        default=None,
        help="Where to write the PCA plot (use '-' to display an interactive plot)",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="./data",
        help="Directory containing the data",
    )

    return parser.parse_args()


def format_tagfreq(count):
    count = int(count)
    if count < 1000:
        return str(count)
    elif count < 1000_000:
        return f"{count*1e-3:.1f}k"
    return f"{count*1e-6:.1f}m"


if __name__ == "__main__":
    args = parse_args()
    dothething(args)

#!/usr/bin/python
from pathlib import Path
import argparse

import numpy as np
import safetensors


from sklearn.decomposition import PCA
from e6db.utils.numpy import load_tags
from e6db.utils import (
    tag_categories,
    tag_category2id,
    tag_categories_colors,
    tag_categories_alt_colors,
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

    sel_idxs = np.array(sorted(tags2id[t] for t in args.tags if t in tags2id))
    sel_tags = idx2tag[sel_idxs]
    print("Query tags:", " ".join(sel_tags))

    # Select neighboring tags similar to the input tags
    n_neighbors = args.global_topk
    top_k = args.topk
    if top_k is None:
        top_k = int(1.5 * args.global_topk // len(sel_idxs))

    # Score and filter
    scores = Xt @ Xt[sel_idxs].T
    scores[sel_idxs, :] = float("-inf")
    if args.category is not None:
        scores[tag_categories != tag_category2id[args.category], :] = float("-inf")

    # Per query top-k
    neigh_idxs = np.argpartition(-scores, top_k, axis=0)[:top_k]

    for i, t in enumerate(sel_tags):
        o = np.argsort(scores[neigh_idxs[:, i], i])[::-1]
        o = neigh_idxs[o[: args.display_topk], i]
        tag_list = " ".join(idx2tag[o])
        print(f"{t}: {tag_list}")

    # Deduplicate, global top-k
    neigh_idxs = np.unique(neigh_idxs)
    scores = scores[neigh_idxs, :].sum(axis=1)
    if len(neigh_idxs) > n_neighbors:
        neigh_idxs = neigh_idxs[np.argpartition(-scores, n_neighbors)[:n_neighbors]]

    if not args.plot_out:
        return

    from matplotlib import pyplot as plt

    idxs = np.concatenate([sel_idxs, neigh_idxs])
    query_slice = slice(None, len(sel_idxs))
    target_slice = slice(len(sel_idxs), len(sel_idxs) + args.display_topk)
    colors = tag_categories_alt_colors if args.no_dark else tag_categories_colors
    colors = np.array(colors)[tag_categories[idxs]]

    # Local PCA
    X2 = Xt[idxs]
    del Xt
    X2 = X2 - X2.mean(0)
    X2 /= np.linalg.norm(X2, axis=1)[:, None]
    X2t = PCA(2).fit_transform(X2)[:, ::-1]

    f, ax = plt.subplots(
        figsize=(15, 15), facecolor="white" if args.no_dark else "black"
    )
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
        description="Query similar tags and plots a local PCA",
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
        choices=tag_categories,
        default=None,
        help="restrict the output to the specified tag category",
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
        "-d",
        "--first-pca",
        type=int,
        default=None,
        help="truncation rank for the global PCA applied to all vectors for smoothing them",
    )
    parser.add_argument(
        "-k",
        "--topk",
        type=int,
        default=None,
        help="Number of neighbors to consider for each query tag",
    )
    parser.add_argument(
        "-o",
        "--plot-out",
        type=Path,
        default=None,
        help="Where to write the PCA plot",
    )
    parser.add_argument(
        "--no-dark",
        action="store_true",
        help="Invert colors of the plot",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="./data",
        help="Directory containing the data",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dothething(args)

import re
import datetime
import logging
from pathlib import Path

from tqdm import tqdm

import polars as pl
from polars import col


def convert_db_export_to_parquet(dumps_path, min_freq=2):
    dumps_path = Path(dumps_path)
    paths = get_csv_paths(dumps_path)
    parquets_dir = dumps_path.parent / "parquets"
    parquets_dir.mkdir(exist_ok=True)

    logging.info("Reading tag CSVs...")
    tags, aliases, impls = read_tags_csvs(paths)
    post_parquet_paths, tag_freqs = read_posts_csv(paths["posts"], parquets_dir)
    logging.info("Normalizing tags...")
    used_tags, tag2index, impl_mapped, rejtag_impls_csq_mapped = normalize_tag_list(
        tag_freqs, tags, aliases, impls, min_freq=min_freq
    )

    used_tags.with_columns(col("tag").cast(pl.String)).write_parquet(
        parquets_dir / "tags.parquet", compression="zstd"
    )
    tag2index.write_parquet(parquets_dir / "tag2index.parquet", compression="zstd")

    all_posts = post_process_posts(
        post_parquet_paths, tag2index, rejtag_impls_csq_mapped, impl_mapped
    )

    logging.info("Writing posts.parquet...")
    all_posts.write_parquet(parquets_dir / "posts.parquet", compression="zstd")

    return used_tags, all_posts


@pl.StringCache()
def read_tags_csvs(paths, alias_implications=True):
    """Reads tags, tag_aliases, tag_implications CSVs"""
    tags = pl.read_csv(
        paths["tags"],
        dtypes=[pl.Categorical, pl.UInt8],
        columns=["name", "category"],
    )

    aliases = (
        pl.scan_csv(paths["tag_aliases"])
        .filter(
            col("status") == "active",
            col("antecedent_name") != col("consequent_name"),
        )
        .select(expr_cast_tocat("antecedent_name"), expr_cast_tocat("consequent_name"))
        .sort("antecedent_name", "consequent_name")
        .collect()
    )

    impls = (
        pl.scan_csv(paths["tag_implications"])
        .filter(
            col("status") == "active",
            col("antecedent_name") != col("consequent_name"),
        )
        .select(expr_cast_tocat("antecedent_name"), expr_cast_tocat("consequent_name"))
    )

    # Map aliased tag on both side of the implication edges
    impls = (
        (
            impls.join(
                aliases.lazy(),
                how="left",
                left_on="consequent_name",
                right_on="antecedent_name",
                suffix="_alias",
            )
            .select(
                "antecedent_name",
                consequent_name=col("consequent_name_alias").fill_null(
                    col("consequent_name")
                ),
            )
            .join(aliases.lazy(), how="left", on="antecedent_name", suffix="_alias")
            .select(
                col("consequent_name_alias")
                .fill_null(col("antecedent_name"))
                .alias("antecedent_name"),
                "consequent_name",
            )
            .filter(col("antecedent_name") != col("consequent_name"))
            .collect()
        )
        if alias_implications
        else impls.collect()
    )
    # transitive closure
    impls = transitive_closure(impls)
    impls = (
        impls.lazy()
        .group_by("antecedent_name", "consequent_name")
        .min()
        .sort("antecedent_name", "consequent_name")
        .collect()
    )

    return tags, aliases, impls


def normalize_tag_list(tag_freqs, tags, aliases, impls, min_freq=2):
    """Filter, augment and normalize the (tag, freq) list accumulated from the posts table"""
    # Filter tags that have actual use
    used_tags = (
        tag_freqs.lazy()
        .join(aliases.lazy(), how="left", left_on="tag", right_on="antecedent_name")
        # Map aliased if any (found orc_humanoid->orc)
        .select(tag=col("consequent_name").fill_null(col("tag")), freq="freq")
        .group_by("tag")
        .sum()
        # Join categories
        .join(tags.lazy(), how="left", left_on="tag", right_on="name", validate="1:1")
        .sort("freq", descending=True)
        .filter(col("freq") >= min_freq)
        .collect()
    )

    # tag string -> tag id translation table
    tag2index = (
        pl.concat(
            [
                used_tags.lazy().select("tag").with_row_index(),
                # Adds aliased tags mapping to their consequent/aliased_to tag id
                aliases.lazy()
                .select("consequent_name", tag="antecedent_name")
                .join(
                    used_tags.lazy().select("tag").with_row_index(),
                    how="inner",
                    left_on="consequent_name",
                    right_on="tag",
                )
                .select("index", "tag"),
            ]
        )
        .sort("tag")
        .collect()
    )

    # Maps implication to tag ids. First on the consequent side:
    impls_csq_mapped = (
        impls.lazy()
        .join(tag2index.lazy(), how="inner", left_on="consequent_name", right_on="tag")
        .select(antecedent_name="antecedent_name", consequent="index", depth="depth")
        .collect()
        .lazy()
    )

    # Some rejected tag could imply tags in the selected set.
    # To recover them latter, store the implications from rejected tags.
    rejtag_impls_csq_mapped = impls_csq_mapped.join(
        tag2index.lazy(), how="anti", left_on="antecedent_name", right_on="tag"
    ).collect()

    # Maps the antecedent side of implications to tag ids
    impl_mapped = (
        impls_csq_mapped.join(
            tag2index.lazy(), how="inner", left_on="antecedent_name", right_on="tag"
        )
        .select(antecedent="index", consequent="consequent", depth="depth")
        .sort("antecedent", "consequent")
        .collect()
    )
    return used_tags, tag2index, impl_mapped, rejtag_impls_csq_mapped


@pl.StringCache()
def read_posts_csv(posts_csv_path, out_path, batch_size=1 << 18, write_parquets=True):
    """First pass on posts csv.

    Parse textual data from CSV chunks and store them as parquet files.
    Accumulate tag frequencies.
    """
    schema = dict(
        id=pl.UInt32,
        uploader_id=pl.UInt32,
        created_at=pl.String,
        md5=pl.String,
        source=pl.String,
        rating=pl.String,
        image_width=pl.Int32,
        image_height=pl.Int32,
        tag_string=pl.String,
        locked_tags=pl.String,
        fav_count=pl.UInt32,
        file_ext=pl.String,
        parent_id=pl.UInt32,
        change_seq=pl.UInt32,
        approver_id=pl.UInt32,
        file_size=pl.UInt32,
        comment_count=pl.Int16,
        description=pl.String,
        duration=pl.String,
        updated_at=pl.String,
        is_deleted=pl.String,
        is_pending=pl.String,
        is_flagged=pl.String,
        score=pl.Int16,
        up_score=pl.UInt16,
        down_score=pl.UInt16,
        is_rating_locked=pl.String,
        is_status_locked=pl.String,
        is_note_locked=pl.String,
    )
    column_selections = [
        "id",
        "uploader_id",
        "created_at",
        "md5",
        # "source",
        "rating",
        "image_width",
        "image_height",
        "tag_string",
        "fav_count",
        "file_ext",
        "file_size",
        "comment_count",
        # "description",
        "is_deleted",
        "score",
        "up_score",
    ]
    columns_remaps = [
        col("created_at").str.to_datetime("%Y-%m-%d %H:%M:%S%.f"),
        col("md5").str.decode("hex"),
        col("image_width").cast(pl.UInt16),
        col("image_height").cast(pl.UInt16),
        col("tag_string").str.split(" ").cast(pl.List(pl.Categorical)),
        col("fav_count").cast(pl.UInt16),
        col("comment_count").cast(pl.UInt16),
        col("up_score").cast(pl.UInt16),
    ]
    reader = pl.read_csv_batched(
        posts_csv_path, columns=column_selections, dtypes=schema, batch_size=batch_size
    )

    tag_freqs = None
    batch_idx = 0
    parquet_paths = []
    progress = tqdm(desc=f"Reading {posts_csv_path.name}")
    while True:
        batch = reader.next_batches(1)
        if not batch:
            break
        (chunk_df,) = batch
        del batch

        chunk_df = (
            chunk_df.lazy()
            # Filtering
            .filter(col("file_ext").is_in(("jpg", "png", "webp")), is_deleted="f").drop(
                "is_deleted"
            )
            # Projection
            .with_columns(*columns_remaps)
            .collect(streaming=True)
        )

        if write_parquets:
            parquet_path = out_path / f"posts-{batch_idx:03}.parquet"
            parquet_paths.append(parquet_path)
            chunk_df.write_parquet(parquet_path, compression="zstd")

        # Count tag in the batch, accumulate frequencies
        chunk_tag_freqs = (
            chunk_df.lazy()
            .select(tag="tag_string")
            .explode("tag")
            .group_by("tag")
            .len()
            .select("tag", freq="len")
            .collect()
        )
        del chunk_df
        if tag_freqs is None:
            tag_freqs = chunk_tag_freqs
        else:
            tag_freqs = (
                tag_freqs.lazy()
                .join(
                    chunk_tag_freqs.lazy(),
                    on="tag",
                    how="outer_coalesce",  # validate='1:1' <- needed for streaming, wth?
                )
                .select(
                    "tag",
                    freq=col("freq").fill_null(0) + col("freq_right").fill_null(0),
                )
                .collect(streaming=False)
            )
        del chunk_tag_freqs

        batch_idx += 1
        progress.update()

    progress.close()
    return parquet_paths, tag_freqs.rechunk()


def post_process_posts(
    post_parquet_paths, tag2index, rejtag_impls_csq_mapped, impl_mapped
):
    all_posts = None
    for f in tqdm(post_parquet_paths, desc="Consolidating posts"):
        posts = pl.scan_parquet(f)
        post_tags = (
            posts.select("tag_string", postid="id")
            .explode("tag_string")
            .join(tag2index.lazy(), how="left", left_on="tag_string", right_on="tag")
            .cache()
        )

        post_implied_tags = (
            post_tags.lazy()
            .join(
                impl_mapped.lazy(), how="inner", left_on="index", right_on="antecedent"
            )
            .select("postid", implied_tags="consequent")
            .merge_sorted(
                post_tags.lazy()
                .join(
                    rejtag_impls_csq_mapped.lazy(),
                    how="inner",
                    left_on="tag_string",
                    right_on="antecedent_name",
                )
                .select("postid", implied_tags="consequent"),
                "postid",
            )
            .group_by("postid", maintain_order=True)
            .agg(col("implied_tags").unique().sort())
        )

        post_tags = (
            post_tags.filter(~col("index").is_null())
            .group_by("postid", maintain_order=True)
            .agg(tags=col("index").sort())
            .join(post_implied_tags, how="left", on="postid")
            .select(
                "postid",
                "tags",
                col("implied_tags").fill_null([]),
            )
            .select(
                "postid",
                "implied_tags",
                stripped_tags=col("tags").list.set_difference("implied_tags"),
            )
        )
        del post_implied_tags

        posts = (
            posts.lazy()
            .drop("tag_string")
            .join(post_tags, how="left", left_on="id", right_on="postid")
            .collect()
        )
        del post_tags
        all_posts = posts if all_posts is None else all_posts.vstack(posts)
        del posts

        f.unlink()
    return all_posts


def transitive_closure(
    edges, antecedent="antecedent_name", consequent="consequent_name", max_depth=16
):
    edges = edges.select("antecedent_name", "consequent_name")
    all_edges = edges.with_columns(depth=pl.lit(0, pl.UInt8))
    supp_edges = edges
    for i in range(1, max_depth):
        supp_edges = supp_edges.lazy().join(
            edges.lazy(),
            how="inner",
            left_on=consequent,
            right_on=antecedent,
        )
        supp_edges = supp_edges.select(
            antecedent,
            consequent_name=f"{consequent}_right",
            depth=pl.lit(i, pl.UInt8),
        )
        supp_edges = supp_edges.collect()
        if len(supp_edges) == 0:
            break
        all_edges.vstack(supp_edges, in_place=True)
    else:
        logging.warning(
            "transitive_closure: reached max_depth=%d, len(supp_edges)=%d",
            max_depth,
            len(supp_edges),
        )
    return all_edges


RE_DUMP_PATH = re.compile(r"(.*?/)*?(\w+)-(\d+-\d+-\d+)\.csv\.?.*")
DUMP_NAMES = {"posts", "tags", "tag_aliases", "tag_implications"}


def get_csv_paths(dump_dir):
    "return the paths to the latests dump files"
    dump_entries = {k: None for k in DUMP_NAMES}
    for path in dump_dir.glob("*.csv*"):
        if m := RE_DUMP_PATH.match(str(path)):
            name, date = m.groups()[-2:]
            if name not in dump_entries:
                continue
            date = datetime.date.fromisoformat(date)
            if old_entry := dump_entries[name]:
                if date < old_entry[1]:
                    continue
            dump_entries[name] = (path, date)

    dump_paths = {k: e[0] for k, e in dump_entries.items() if e is not None}
    not_found = DUMP_NAMES - dump_paths.keys()
    if not_found:
        raise RuntimeError(f"CSVs {not_found!s} not found in {dump_dir!s}")

    return dump_paths


def expr_cast_tocat(c):
    return col(c).cast(pl.Categorical())


if __name__ == "__main__":
    import sys

    logging.root.setLevel(logging.INFO)
    data_dir = sys.argv.get(1, "./data")
    convert_db_export_to_parquet(data_dir)

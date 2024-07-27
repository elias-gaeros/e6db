---
license: cc-by-sa-4.0
task_categories:
  - token-classification
language:
  - en
size_categories:
  - 10K<n<100K
tags:
  - not-for-all-audiences
---

# E6DB

This a dataset is compiled from the e621 database. It currently provides
utilities and indexes for normalizing tags and embeddings for finding similar
tags.

## Installation

Please clone with `git` after having installed
[`git-lfs`](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).
Do not download the github zip, it doesn't contain the data files.

## Utilities

### `query_tags.py`

A small command-line utility that finds related tags and generates 2D plots
illustrating tag relationships through local PCA projection. It utilizes
collaborative filtering embeddings, [computed with alternating least
squares](./notebooks/AltLstSq.ipynb).

When only tags are provided as arguments, it displays the top-k most similar
tags (where k is set with `-k`). By using `-o plot.png` or `-o -`, it saves or
displays a 2D plot showing the local projection of the query and related tags.

Tag categories are represented with the e621 color scheme. Results can be
filtered based on one or more categories using the `-c` flag once or multiple
times. The `-f` flag sets a post count threshold.

Filtering occurs in the following sequence:

- Tags used fewer than twice are excluded from the dataset; tags with a post
  count lower than the `-f` threshold are also discarded.
- If a category filter is specified, only matching tags are retained.
- For each query tag, the most `-k` similar neighboring tags are selected.
- The per-query neighbors are printed, and if no plot is being generated,
  filtering halts at this point.
- Similarity scores are aggregated across queries, and the `-n` tags closest to
  all queries are chosen for the PCA.
- Only the highest `-N` scoring tags are displayed in the plot.

### `normalize_tags.py`

Tag Normalizer is a powerful command-line tool designed to clean, standardize,
and normalize e621 tags in text files. By applying a set of customizable rules,
Tag Normalizer helps maintain consistency and improve the quality of your tag
data.

#### Usage

```
python normalize_tags.py <path to input dataset> <path to output normalized tags> -s unknown
```

Find all `*.txt` and `*.cap*` files in `<path to input dataset>` and write the
normalized files in `<path to output normalized tags>` while reproducing the
folder hierarchy. Additionally, `-s unknown` writes to the standard output the
the top 100 unrecognized tags. `-s meta` will show the top 100 meta tags, while
`-k 50` alone will show the top 50 for all categories.

You can specify the same folder for input and output, and use `-f` to skip
confirmation.

#### Configuration

Tag Normalizer uses a TOML configuration file to customize its behavior. By
default, it looks for `normalize.toml` in the following locations:

1. The path specified by the `-c` or `--config` option
2. The output directory
3. The input directory
4. The current directory (most likely the example one)

Here's a brief overview of the main configuration options:

- `blacklist`: List of tags to remove
- `blacklist_regexp`: Regular expressions for blacklisting tags
- `keep_underscores`: List of tags where underscores should be preserved
- `blacklist_categories`: Categories of tags to remove entirely
- `remove_parens_suffix_for_categories`: Categories where parenthetical suffixes
  should be removed
- `aliases`: Define tag aliases
- `aliases_overrides`: Define aliases that can override existing tag meanings
- `renames`: Specify tags to be renamed in the output
- `use_underscores`: Whether to use underscores or spaces in output tags
- `keep_implied`: Whether to keep or remove implied tags, may also be a list of
  tags to keep even when implied by another one
- `on_alias_conflict`: How to handle conflicts when creating aliases
- `artist_by_prefix`: Whether to add "by\_" prefix to artist tags
- `blacklist_implied`: Whether to also blacklist tags implied by blacklisted
  tags

For a detailed explanation of each option, refer to the comments in the
`normalize.toml` file. It contains the default values with some opinionated
changes.

#### Command-line Options

- `-c`, `--config`: Specify a custom configuration file (default: looks for
  `normalize.toml` in output dir, input dir, or current dir)
- `-v`, `--verbose`: Enable verbose logging
- `-f`, `--force`: Don't ask for confirmation when overwriting input files
- `-b`, `--print-blacklist`: Print the effective list of blacklisted tags
- `-k`, `--print-topk [N]`: Print the N most common tags (default: 100 if no
  value provided)
- `-s`, `--stats-categories <cat>`: Restrict tag count printing to specific
  categories (or `unknown` for non e621 tags)
- `-j`, `--print-implied-topk [N]`: Print the N most common implied tags
  (default: 100 if no value provided)

### `e6db.utils`

This [module](./e6db/utils/__init__.py) contains utilities for loading the provided data and use it to normalize tag sets.

- `load_tags` and `load_implications` loads the tags indexes and implications,
- `TagNormalizer` allows to adapt the spellings of tag and alias for working with various datasets. It can normalize spellings by converting tag strings to numerical ids and back,
- `TagSetNormalizer` uses the above class along tag implications to normalize tag sets and strip implied tags.

See [this example notebook](./notebooks/Normalize%20tags%20T2I%20dataset.ipynb) that cleans T2I datasets in the sd-script format.

Additionally, the `e6db.utils.numpy` and `e6db.utils.torch` modules provides
functions to construct post-tag interaction matrices in sparse matrix format. For this
you'll need to generate the `posts.parquet` file from CSVs.

### `importdb`

Reads the CSVs from [e621 db export](https://e621.net/db_export/)

`python -m e6db.importdb ./data` reads tags, aliases, implications and posts CSV
files. The following operations are performed:

- Tags used at least twice are assigned numerical ids based on their rank.
- Computes the transitive closure of implications,
- For each post, split the tags into direct and implied tag.
- Write parquets files for tags and posts (~500MB) and convert tags indexes using simple formats described in the next section.

The CSV files must be decompressed beforehand.

## Dataset content

This dataset currently focus on tags alone using simple file formats that are
easily parsed without additional dependencies.

- `tag2idx.json.gz`: a dictionary mapping tag strings and aliases to numerical id (tag rank),
- `tags.txt.gz`: list of tags sorted by rank, can be indexed by the ids given by `tag2idx.json.gz`,
- `tags_categories.bin.gz`: a raw array of bytes representing tag categories in the same order than `tags.txt.gz`,
- `implications.json.gz`: maps tags id to implied tag ids (including transitive implications),
- `implications_rej.json.gz`: maps tag strings to a list of implied numerical
  ids. Keys in implications_rej are tags that have a very little usage (less
  than 2 posts) and don't have numerical ids associated with them.
- `implicit_tag_factors.safetensors`: Tag embedding computed by [alternating least squares](./notebooks/AltLstSq.ipynb).

No post data is currently included, since this wouldn't add any useful
information compared to what's inside the CSVs. If you want the post parquet
files with normalized tags, you can download the CSVs and run the
[`e6db.importdb`](#importdb) script yourself.

I plan to compile more post data in the future, such as aesthetic predictions,
adjusted favcounts, etc. Utilities will then be added to assists with the
selection of a subset of posts for specific ML tasks.

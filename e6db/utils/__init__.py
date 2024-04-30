"Python only utils (no dependencies)"
from pathlib import Path
import gzip
import json
import warnings
import math
from typing import Callable, Iterable

tag_categories = [
    "general",
    "artist",
    None,  # Invalid catid
    "copyright",
    "character",
    "species",
    "invalid",
    "meta",
    "lore",
    "pool",
]
tag_category2id = {v: k for k, v in enumerate(tag_categories) if v}
tag_categories_colors = [
    "#b4c7d9",
    "#f2ac08",
    None,  # Invalid catid
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
    None,  # Invalid catid
    "#ff5eff",
    "#2bff2b",
    "#f6b295",
    "#ffbdbd",
    "#666",
    "#5fdb5f",
    "#d0b27a",
]


def load_tags(data_dir):
    """
    Load tag data, returns a tuple `(tag2idx, idx2tag, tag_categories)`

    * `tag2idx`: dict mapping tag and aliases to numerical ids
    * `idx2tag`: list mapping numerical id to tag string
    * `tag_categories`: byte string mapping numerical id to categories
    """
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


def load_implications(data_dir):
    """
    Load implication mappings. Returns a tuple `(implications, implications_rej)`

    * `implications`: dict mapping numerical ids to a list of implied numerical
      ids. Contains transitive implications.
    * `implications_rej`: dict mapping tag to a list of implied numerical ids
    keys in implications_rej are tag that have a very little usage (less than 2
    posts) and don't have numerical ids associated with them.
    """
    with gzip.open(data_dir / "implications.json.gz", "rb") as fp:
        implications = json.load(fp)
    implications = {int(k): v for k, v in implications.items()}
    with gzip.open(data_dir / "implications_rej.json.gz", "rb") as fp:
        implications_rej = json.load(fp)
    return implications, implications_rej


def tag_rank_to_freq(rank: int) -> float:
    """Approximate the frequency of a tag given its rank"""
    return math.exp(26.4284 * math.tanh(2.93505 * rank ** (-0.136501)) - 11.492)


def tag_freq_to_rank(freq: int) -> float:
    """Approximate the rank of a tag given its frequency"""
    log_freq = math.log(freq)
    return math.exp(
        -7.57186
        * (0.0465456 * log_freq - 1.24326)
        * math.log(1.13045 - 0.0720383 * log_freq)
        + 12.1903
    )


MapFun = Callable[[str, int | None], str | list[str]]


class TagNormalizer:
    """
    Map tag strings to numerical ids, and vice versa.

    Multiple strings can be mapped to a single id, while each id maps to a
    single string. As a result, the encode/decode process can be used to
    normalize tags to canonical spelling.

    See `add_input_mappings` for adding aliases, and `rename_output` for setting
    the canonical spelling of a tag.
    """

    def __init__(self, path_or_data: str | Path | tuple[dict, list, bytes]):
        if isinstance(path_or_data, (Path, str)):
            data = load_tags(path_or_data)
        else:
            data = path_or_data
        self.tag2idx, self.idx2tag, self.tag_categories = data

    def get_category(self, tag: int | str, as_string=True) -> int:
        if isinstance(tag, str):
            tag = self.encode(tag)
        cat = self.tag_categories[tag]
        if as_string:
            return tag_categories[cat]
        return cat

    def encode(self, tag: str, default=None):
        "Convert tag string to numerical id"
        return self.tag2idx.get(tag, default)

    def decode(self, tag: int | str):
        "Convert numerical id to tag string"
        if isinstance(tag, str):
            return tag
        return self.idx2tag[tag]

    def get_reverse_mapping(self):
        """Return a list mapping id -> [ tag strings ]"""
        res = [[] for i in range(len(self.idx2tag))]
        for tag, tid in self.tag2idx.items():
            res[tid].append(tag)
        return res

    def add_input_mappings(
        self, tags: str | Iterable[str], to_tid: int | str, on_conflict="raise"
    ):
        """Associate tag strings to an id for recognition by `encode`

        `on_conflict` defines what to do when the tag string is already mapped
        to a different id:

        * "raise": raise an ValueError (default)
        * "warn": raise a warning
        * "overwrite_rarest": make the tag point to the most frequently used tid
        * "overwrite": silently overwrite the mapping
        * "silent", or any other string: don't set the mapping
        """
        tag2idx = self.tag2idx
        if not isinstance(to_tid, int):
            to_tid = tag2idx[to_tid]
        if isinstance(tags, str):
            tags = (tags,)
        for tag in tags:
            conflict = tag2idx.get(tag, to_tid)
            if conflict != to_tid:
                msg = f"mapping {tag!r}->{self.idx2tag[to_tid]!r}({to_tid}) conflicts with previous mapping {tag!r}->{self.idx2tag[conflict]!r}({conflict})."
                if on_conflict == "raise":
                    raise ValueError(msg)
                elif on_conflict == "warn":
                    warnings.warn(msg)
                elif on_conflict == "overwrite_rarest" and to_tid > conflict:
                    continue
                elif on_conflict != "overwrite":
                    continue
            tag2idx[tag] = to_tid

    def rename_output(self, orig: int | str, dest: str):
        """Change the tag string associated with an id. Used by `decode`."""
        if not isinstance(orig, int):
            orig = self.tag2idx[orig]
        self.idx2tag[orig] = dest

    def map_inputs(self, mapfun: MapFun, on_conflict="raise") -> "TagNormalizer":
        res = type(self)(({}, self.idx2tag, self.tag_categories))
        for tag, tid in self.tag2idx.items():
            res.add_input_mappings(mapfun(tag, tid), tid, on_conflict=on_conflict)
        return res

    def map_outputs(self, mapfun: MapFun) -> "TagNormalizer":
        idx2tag_gen = (mapfun(t, i) for i, t in enumerate(self.idx2tag))
        idx2tag = [t if isinstance(t, str) else t[0] for t in idx2tag_gen]
        return type(self)((self.tag2idx, idx2tag, self.tag_categories))

    def get(self, key: int | str, default=None):
        """
        Returns the string tag associated with a numerical id, or conversely,
        the id associated with a tag.
        """
        if isinstance(key, int):
            idx2tag = self.idx2tag
            if key >= len(idx2tag):
                return default
            return idx2tag[key]
        return self.tag2idx.get(key, default)


class TagSetNormalizer:
    def __init__(self, path_or_data: str | Path | tuple[TagNormalizer, dict, dict]):
        if isinstance(path_or_data, (Path, str)):
            data = TagNormalizer(path_or_data), *load_implications(path_or_data)
        else:
            data = path_or_data
        self.tag_normalizer, self.implications, self.implications_rej = data

    def map_implicaitons_rej(
        self, mapfun: MapFun, on_conflict="raise"
    ) -> "TagSetNormalizer":
        implications_rej: dict[str, list[str]] = {}
        for tag_string, implied_ids in self.implications_rej.items():
            for new_tag_string in mapfun(tag_string, None):
                conflict = implications_rej.get(new_tag_string, implied_ids)
                if conflict != implied_ids:
                    msg = f"mapping {tag_string!r}->{implied_ids} conflicts with previous mapping {tag_string!r}->{conflict}."
                    if on_conflict == "raise":
                        raise ValueError(msg)
                    elif on_conflict == "warn":
                        warnings.warn(msg)
                    elif on_conflict != "overwrite":
                        continue
                implications_rej[new_tag_string] = implied_ids

        return type(self)((self.tag_normalizer, self.implications, implications_rej))

    def map_tags(
        self, mapfun: MapFun, map_input=True, map_output=True, on_conflict="raise"
    ) -> "TagSetNormalizer":
        """Apply a function to all tag strings.

        The provided function will be run on:

        * The of list output tag strings,
        * Keys from the dictionary mapping strings to ids, contains canonical
          tag and aliases,
        * Implication source tags that are not used frequently enough to get an
          id assigned (less than twice).

        The function should return a list, where the first string is the
        canonical tag used in the output, the others are additional aliases
        used for recognizing the tag.
        """
        tag_normalizer = self.tag_normalizer
        if map_input:
            tag_normalizer = tag_normalizer.map_inputs(mapfun, on_conflict=on_conflict)
        if map_output:
            tag_normalizer = tag_normalizer.map_outputs(mapfun)
        res = type(self)((tag_normalizer, self.implications, self.implications_rej))
        if map_input:
            res = res.map_implicaitons_rej(mapfun, on_conflict=on_conflict)
        return res

    def encode(self, tags: Iterable[str], keep_implied=False):
        """
        Encode a list of string as numerical ids and strip implied tags.

        Unknown tags are returned as strings.

        Returns :

        * a list of tag ids and unknown tag strings,
        * a list of implied tag ids.
        """
        implied = set()
        res = []
        for tag in tags:
            tag = self.tag_normalizer.encode(tag, tag)
            implied.update(
                self.implications.get(tag, ())
                if isinstance(tag, int)
                else self.implications_rej.get(tag, ())
            )
            res.append(tag)
        if not keep_implied:
            res = [t for t in res if t not in implied]
        return res, implied

    def decode(self, tags):
        return [self.tag_normalizer.decode(t) for t in tags]

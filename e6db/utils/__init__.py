"Python only utils (no dependencies)"
import gzip
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

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
    with gzip.open(data_dir / "tags.txt.gz", "rt", encoding="utf-8") as fd:
        idx2tag = fd.read().split("\n")
        if not idx2tag[-1]:
            idx2tag = idx2tag[:-1]
    with gzip.open(data_dir / "tag2idx.json.gz", "rb") as fp:
        tag2idx = json.load(fp)
    with gzip.open(data_dir / "tags_categories.bin.gz", "rb") as fp:
        tag_categories = fp.read()
    logging.info(f"Loaded {len(idx2tag)} tags, {len(tag2idx)} tag2id mappings")
    return tag2idx, idx2tag, tag_categories


def load_implications(data_dir):
    """
    Load implication mappings. Returns a tuple `(implications, implications_rej)`

    * `implications`: dict mapping numerical ids to a list of implied numerical
      ids. Contains transitive implications.
    * `implications_rej`: dict mapping tag strings to a list of implied
      numerical ids. keys in implications_rej are tags that have a very little
      usage (less than 2 posts) and don't have numerical ids associated with
      them.
    """
    with gzip.open(data_dir / "implications.json.gz", "rb") as fp:
        implications = json.load(fp)
    implications = {int(k): v for k, v in implications.items()}
    with gzip.open(data_dir / "implications_rej.json.gz", "rb") as fp:
        implications_rej = json.load(fp)
    logger.info(
        f"Loaded {len(implications)} implications + {len(implications_rej)} implication from tags without id"
    )
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


InMapFun = Callable[[str, int | None], list[str]]
OutMapFun = Callable[[str], list[str]]


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
                    logger.warning(msg)
                elif on_conflict == "overwrite_rarest" and to_tid > conflict:
                    continue
                elif on_conflict != "overwrite":
                    continue
            tag2idx[tag] = to_tid

    def remove_input_mappings(self, tags: str | Iterable[str]):
        """Remove tag strings from the mapping"""
        if isinstance(tags, str):
            tags = (tags,)
        for tag in tags:
            if tag in self.tag2idx:
                del self.tag2idx[tag]
            else:
                logger.warning(f"tag {tag!r} is not a valid tag")

    def rename_output(self, orig: int | str, dest: str):
        """Change the tag string associated with an id. Used by `decode`."""
        if not isinstance(orig, int):
            orig = self.tag2idx[orig]
        self.idx2tag[orig] = dest

    def map_inputs(
        self, mapfun: InMapFun, prepopulate=True, on_conflict="raise"
    ) -> "TagNormalizer":
        tag2idx = self.tag2idx.copy() if prepopulate else {}
        res = type(self)((tag2idx, self.idx2tag, self.tag_categories))
        for tag, tid in self.tag2idx.items():
            res.add_input_mappings(mapfun(tag, tid), tid, on_conflict=on_conflict)
        return res

    def map_outputs(self, mapfun: OutMapFun) -> "TagNormalizer":
        idx2tag = [mapfun(t, i) for i, t in enumerate(self.idx2tag)]
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

    def map_inputs(self, mapfun: InMapFun, on_conflict="raise") -> "TagSetNormalizer":
        tag_normalizer = self.tag_normalizer.map_inputs(mapfun, on_conflict=on_conflict)

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

        res = type(self)((tag_normalizer, self.implications, implications_rej))
        return res

    def map_outputs(self, mapfun: OutMapFun) -> "TagSetNormalizer":
        tag_normalizer = self.tag_normalizer.map_outputs(mapfun)
        return type(self)((tag_normalizer, self.implications, self.implications_rej))

    def get_implied(self, tag: int | str) -> list[int]:
        if isinstance(tag, int):
            return self.implications.get(tag, ())
        else:
            return self.implications_rej.get(tag, ())

    def encode(
        self,
        tags: list[str],
        keep_implied: bool | set[int] = False,
        max_antecedent_rank: int | None = None,
        drop_antecedent_rank: int | None = None,
    ) -> tuple[list[int | str], set[int]]:
        """
        Encode a list of string as numerical ids and strip implied tags.

        Unknown tags are returned as strings.

        Returns :

        * a list of tag ids and unknown tag strings,
        * a list of implied tag ids.
        """
        tag2idx = self.tag_normalizer.tag2idx
        N = len(tag2idx)
        max_antecedent_rank = max_antecedent_rank or N + 1
        drop_antecedent_rank = drop_antecedent_rank or N + 1
        get_implied = self.implications.get
        get_implied_rej = self.implications_rej.get

        stack = [tag2idx.get(tag, tag) for tag in tags[::-1]]
        implied = set()
        res = dict()  # dict as a cheap ordered set
        while stack:
            tag = stack.pop()
            if isinstance(tag, int):
                antecedent_rank = tag
                consequents = get_implied(tag)
            else:
                # the tag might be a very rare antecedent (less than two posts)
                # that doesn't have a tag id
                antecedent_rank = N
                consequents = get_implied_rej(tag)
            if consequents:
                if antecedent_rank < max_antecedent_rank:
                    implied.update(consequents)
                else:
                    # The implied tags from low frequency antecedent (high rank)
                    # are added to the list and instead the antecedent may be
                    # dropped
                    stack.extend(consequents)
                    if antecedent_rank >= drop_antecedent_rank:
                        continue
            res[tag] = None
        res = res.keys()

        if not keep_implied:
            res = [t for t in res if t not in implied]
        elif isinstance(keep_implied, set):
            res = [t for t in res if t not in implied or t in keep_implied]
        else:
            res = list(res)
        return res, implied

    def decode(self, tags: Iterable[int | str]) -> list[str]:
        idx2tag = self.tag_normalizer.idx2tag
        return [idx2tag[t] if isinstance(t, int) else t for t in tags]

"""
Python utilities for tag normalization, mapping, and frequency analysis.

This module provides:
1. Tag normalization infrastructure to convert between tag strings and numerical IDs
2. Tag implication handling (when one tag implies the presence of another)
3. Tag frequency/rank conversion utilities
4. Tag category management

Key concepts:
- Tag normalization: Converting variant spellings and aliases to canonical forms
- Tag implications: Relationships where one tag implies another (e.g., "dog" implies "mammal")
- Tag frequency: How often a tag appears in the dataset
- Tag rank: Position in frequency order (lower rank = more frequent)

The numerical IDs assigned to tags are based on frequency - the most common tags
have the lowest IDs. This allows efficient representation and facilitates
filtering based on frequency thresholds.
"""
import gzip
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

# Tag categories and their numerical IDs
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
# Map category names to their IDs for easy lookup
tag_category2id = {v: k for k, v in enumerate(tag_categories) if v}
# UI colors for tag categories (light and dark variants)
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
    Load tag data from data files in the specified directory.
    
    This function loads three key components:
    1. idx2tag: List mapping numerical ids to canonical tag strings
    2. tag2idx: Dictionary mapping all tag variants/aliases to their numerical ids
    3. tag_categories: Byte string mapping numerical ids to category ids
    
    Args:
        data_dir: Directory containing the tag data files
        
    Returns:
        A tuple (tag2idx, idx2tag, tag_categories) containing the loaded tag data
    """
    data_dir = Path(data_dir)
    
    # Load canonical tag strings (idx2tag)
    with gzip.open(data_dir / "tags.txt.gz", "rt", encoding="utf-8") as fd:
        idx2tag = fd.read().split("\n")
        if not idx2tag[-1]:
            idx2tag = idx2tag[:-1]
            
    # Load tag-to-id mapping dictionary (tag2idx)
    with gzip.open(data_dir / "tag2idx.json.gz", "rb") as fp:
        tag2idx = json.load(fp)
        
    # Load category data for each tag
    with gzip.open(data_dir / "tags_categories.bin.gz", "rb") as fp:
        tag_categories = fp.read()
        
    logging.info(f"Loaded {len(idx2tag)} tags, {len(tag2idx)} tag2id mappings")
    return tag2idx, idx2tag, tag_categories


def load_implications(data_dir):
    """
    Load tag implication data from the specified directory.
    
    Tag implications represent relationships where one tag (antecedent) implies 
    the presence of another tag (consequent). For example, "dog" implies "mammal".
    
    This function loads two types of implications:
    1. Regular implications (numerical id to list of implied ids)
    2. Rejected tag implications (rare tags without ids to list of implied ids)
    
    Args:
        data_dir: Directory containing the implication data files
        
    Returns:
        A tuple (implications, implications_rej) containing:
        - implications: Dict mapping tag ids to lists of implied tag ids
        - implications_rej: Dict mapping rare tag strings to lists of implied tag ids
    """
    # Load regular implications (id -> list of implied ids)
    with gzip.open(data_dir / "implications.json.gz", "rb") as fp:
        implications = json.load(fp)
    implications = {int(k): v for k, v in implications.items()}
    
    # Load implications from rejected tags (string -> list of implied ids)
    # These are tags so rare they don't have their own ids
    with gzip.open(data_dir / "implications_rej.json.gz", "rb") as fp:
        implications_rej = json.load(fp)
        
    logger.info(
        f"Loaded {len(implications)} implications + {len(implications_rej)} implication from tags without id"
    )
    return implications, implications_rej


def tag_rank_to_freq(rank: int) -> float:
    """
    Approximate the frequency of a tag given its rank.
    
    Tags follow a power-law distribution (Zipf's law), where frequency drops
    exponentially as rank increases. This function models that relationship.
    
    Args:
        rank: The rank of the tag (lower rank = more frequent)
        
    Returns:
        Estimated frequency (number of occurrences) of the tag
    """
    return math.exp(26.4284 * math.tanh(2.93505 * rank ** (-0.136501)) - 11.492)


def tag_freq_to_rank(freq: int) -> float:
    """
    Approximate the rank of a tag given its frequency.
    
    This is the inverse of tag_rank_to_freq, estimating where in the
    frequency ranking a tag with the given number of occurrences would fall.
    
    Args:
        freq: The frequency (number of occurrences) of the tag
        
    Returns:
        Estimated rank position of the tag
    """
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

    The normalizer maintains three main data structures:
    - tag2idx: A dictionary mapping tag strings (including aliases) to their numerical ids
    - idx2tag: A list mapping numerical ids to their canonical tag strings
    - tag_categories: A byte string mapping numerical ids to their categories

    The tag2idx mapping can have many-to-one relationships (multiple strings to the same id),
    which enables alias handling and normalization. The idx2tag mapping is one-to-one,
    representing the canonical form of each tag.

    See `add_input_mappings` for adding aliases, and `rename_output` for setting
    the canonical spelling of a tag.
    """

    def __init__(self, path_or_data: str | Path | tuple[dict, list, bytes]):
        """
        Initialize a TagNormalizer from either a data directory path or pre-loaded data.

        Args:
            path_or_data: Either:
                - A path to a directory containing tag data files
                - A tuple of (tag2idx, idx2tag, tag_categories) where:
                  * tag2idx: Dict mapping tag strings to numerical ids
                  * idx2tag: List mapping numerical ids to canonical tag strings
                  * tag_categories: Byte string mapping numerical ids to categories
        """
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
        """
        Convert a tag string to its numerical id.
        
        This is one of the core functions of the normalizer, converting a tag string
        to its canonical numerical id for processing.
        
        Args:
            tag: The tag string to encode
            default: Value to return if the tag is not found in the mapping
            
        Returns:
            The numerical id of the tag, or the default value if not found
        """
        return self.tag2idx.get(tag, default)

    def decode(self, tag: int | str):
        """
        Convert a numerical id to its canonical tag string.
        
        This is the inverse operation of encode, retrieving the canonical representation
        of a tag from its id.
        
        Args:
            tag: The numerical id to decode, or a string to return as-is (pass-through)
            
        Returns:
            The canonical string representation of the tag
        """
        if isinstance(tag, str):
            return tag
        return self.idx2tag[tag]

    def get_reverse_mapping(self):
        """
        Return a list mapping each id to all the tag strings that encode to it.
        
        This effectively inverts the tag2idx dictionary, grouping all alias strings
        by their shared numerical id.
        
        Returns:
            A list where each element at index i is a list of all strings that map to id i
        """
        res = [[] for i in range(len(self.idx2tag))]
        for tag, tid in self.tag2idx.items():
            res[tid].append(tag)
        return res

    def add_input_mappings(
        self, tags: str | Iterable[str], to_tid: int | str, on_conflict="raise"
    ):
        """
        Associate tag strings to an id for recognition by `encode`.
        
        This function creates or updates mappings in the tag2idx dictionary, which is used
        to convert tag strings to numerical ids. It allows creating aliases (multiple strings
        mapping to the same id) and handling potential conflicts with existing mappings.
        
        When a tag string is already mapped to a different id, the conflict is resolved
        according to the on_conflict parameter.

        Args:
            tags: One or more tag strings to be mapped to the target id
            to_tid: The target id (or a tag string whose id will be used as the target)
            on_conflict: How to handle conflicts when a tag string is already mapped:
                * "raise": Raise a ValueError (default)
                * "warn": Log a warning but don't change the mapping
                * "overwrite_rarest": Keep the mapping to the most frequent tag (lower id = more frequent)
                * "overwrite": Silently replace the existing mapping
                * "silent" or any other value: Silently keep the existing mapping
        """
        # Get the tag2idx mapping dictionary
        tag2idx = self.tag2idx
        
        # If to_tid is a string, look up its numerical id
        if not isinstance(to_tid, int):
            to_tid = tag2idx[to_tid]
            
        # Handle single string case
        if isinstance(tags, str):
            tags = (tags,)
            
        # Process each tag string
        for tag in tags:
            # Check if this tag already maps to a different id (conflict)
            conflict = tag2idx.get(tag, to_tid)
            if conflict != to_tid:
                # Prepare conflict message
                msg = f"mapping {tag!r}->{self.idx2tag[to_tid]!r}({to_tid}) conflicts with previous mapping {tag!r}->{self.idx2tag[conflict]!r}({conflict})."
                
                # Handle conflict based on on_conflict parameter
                if on_conflict == "raise":
                    raise ValueError(msg)
                elif on_conflict == "warn":
                    logger.warning(msg)
                elif on_conflict == "overwrite_rarest" and to_tid > conflict:
                    # Skip if the existing tag has a lower id (more frequent)
                    continue
                elif on_conflict != "overwrite":
                    # Skip if we're not explicitly overwriting
                    continue
                    
            # Create or update the mapping
            tag2idx[tag] = to_tid

    def remove_input_mappings(self, tags: str | Iterable[str]):
        """
        Remove tag string mappings from tag2idx.
        
        This function removes the specified tag strings from the input mapping,
        making them unrecognizable by the encode method.
        
        Args:
            tags: One or more tag strings to remove from the mapping
        """
        # Handle single string case
        if isinstance(tags, str):
            tags = (tags,)
            
        # Process each tag string
        for tag in tags:
            if tag in self.tag2idx:
                del self.tag2idx[tag]
            else:
                logger.warning(f"tag {tag!r} is not a valid tag")

    def rename_output(self, orig: int | str, dest: str):
        """
        Change the canonical tag string associated with an id.
        
        This modifies the idx2tag list, changing the canonical string representation
        that the decode method will return for a given id.
        
        Args:
            orig: The id to modify (or a tag string whose id will be modified)
            dest: The new canonical string representation for this id
        """
        # If orig is a string, look up its numerical id
        if not isinstance(orig, int):
            orig = self.tag2idx[orig]
            
        # Update the canonical representation
        self.idx2tag[orig] = dest

    def map_inputs(
        self, mapfun: InMapFun, prepopulate=True, on_conflict="raise"
    ) -> "TagNormalizer":
        """
        Create a new TagNormalizer with transformed input mappings.
        
        This applies a mapping function to each tag string in the current normalizer,
        allowing for batch creation of new aliases or remapping of existing tags.
        
        Args:
            mapfun: A function that takes (tag_string, tag_id) and returns a list of 
                   new tag strings to map to that id
            prepopulate: Whether to start with a copy of the current tag2idx or an empty dict
            on_conflict: How to handle conflicts (see add_input_mappings)
            
        Returns:
            A new TagNormalizer with the transformed mappings
        """
        # Create a new tag2idx mapping, either copied or empty
        tag2idx = self.tag2idx.copy() if prepopulate else {}
        
        # Create a new normalizer with the same idx2tag and tag_categories
        res = type(self)((tag2idx, self.idx2tag, self.tag_categories))
        
        # Apply the mapping function to each tag in the current normalizer
        for tag, tid in self.tag2idx.items():
            # Get new tag strings for this tag and add them as mappings to the same id
            res.add_input_mappings(mapfun(tag, tid), tid, on_conflict=on_conflict)
            
        return res

    def map_outputs(self, mapfun: OutMapFun) -> "TagNormalizer":
        """
        Create a new TagNormalizer with transformed canonical tag strings.
        
        This applies a mapping function to each canonical tag string in idx2tag,
        allowing for batch renaming of canonical representations.
        
        Args:
            mapfun: A function that takes (tag_string, tag_id) and returns a new
                   canonical string for that id
                   
        Returns:
            A new TagNormalizer with the transformed canonical strings
        """
        # Apply the mapping function to each canonical tag
        idx2tag = [mapfun(t, i) for i, t in enumerate(self.idx2tag)]
        
        # Create a new normalizer with the transformed idx2tag
        return type(self)((self.tag2idx, idx2tag, self.tag_categories))

    def get(self, key: int | str, default=None):
        """
        Bidirectional lookup: get tag string from id or id from tag string.
        
        This is a convenience method that works in both directions:
        - If given an integer id, returns the canonical tag string
        - If given a string tag, returns its numerical id
        
        Args:
            key: Either a numerical id or a tag string
            default: Value to return if the key is not found
            
        Returns:
            Either the tag string or id corresponding to the key, or default if not found
        """
        if isinstance(key, int):
            idx2tag = self.idx2tag
            if key >= len(idx2tag):
                return default
            return idx2tag[key]
        return self.tag2idx.get(key, default)


class TagSetNormalizer:
    """
    Normalize sets of tags by handling aliases and implications between tags.
    
    This class extends TagNormalizer with the ability to process implications,
    where the presence of one tag implies the presence of other tags. It can
    be used to expand tag sets or filter out redundant tags.
    """
    
    def __init__(self, path_or_data: str | Path | tuple[TagNormalizer, dict, dict]):
        """
        Initialize a TagSetNormalizer from either a data directory or pre-loaded data.
        
        Args:
            path_or_data: Either:
                - A path to a directory containing tag data files
                - A tuple of (tag_normalizer, implications, implications_rej) where:
                  * tag_normalizer: A TagNormalizer instance
                  * implications: Dict mapping tag ids to lists of implied tag ids
                  * implications_rej: Dict mapping tag strings to lists of implied tag ids
                    (for tags without their own ids)
        """
        if isinstance(path_or_data, (Path, str)):
            # Load from data directory
            data = TagNormalizer(path_or_data), *load_implications(path_or_data)
        else:
            # Use provided data
            data = path_or_data
            
        # Unpack the data tuple
        self.tag_normalizer, self.implications, self.implications_rej = data

    def map_inputs(self, mapfun: InMapFun, on_conflict="raise") -> "TagSetNormalizer":
        """
        Create a new TagSetNormalizer with transformed input mappings.
        
        This applies a mapping function to both the tag normalizer's mappings
        and the implications_rej dictionary (for tags without ids).
        
        Args:
            mapfun: A function that takes (tag_string, tag_id) and returns a list of 
                   new tag strings to map to that id
            on_conflict: How to handle conflicts
            
        Returns:
            A new TagSetNormalizer with the transformed mappings
        """
        # Transform the tag normalizer
        tag_normalizer = self.tag_normalizer.map_inputs(mapfun, on_conflict=on_conflict)

        # Transform the implications_rej dictionary (for tags without ids)
        implications_rej: dict[str, list[str]] = {}
        for tag_string, implied_ids in self.implications_rej.items():
            # Apply the mapping function to get new tag strings
            for new_tag_string in mapfun(tag_string, None):
                # Check for conflicts with existing mappings
                conflict = implications_rej.get(new_tag_string, implied_ids)
                if conflict != implied_ids:
                    msg = f"mapping {tag_string!r}->{implied_ids} conflicts with previous mapping {tag_string!r}->{conflict}."
                    if on_conflict == "raise":
                        raise ValueError(msg)
                    elif on_conflict == "warn":
                        warnings.warn(msg)
                    elif on_conflict != "overwrite":
                        continue
                        
                # Add the new mapping
                implications_rej[new_tag_string] = implied_ids

        # Create a new normalizer with the transformed components
        res = type(self)((tag_normalizer, self.implications, implications_rej))
        return res

    def map_outputs(self, mapfun: OutMapFun) -> "TagSetNormalizer":
        """
        Create a new TagSetNormalizer with transformed canonical tag strings.
        
        This transforms the canonical tag strings in the tag normalizer component.
        
        Args:
            mapfun: A function that takes (tag_string, tag_id) and returns a new
                   canonical string for that id
                   
        Returns:
            A new TagSetNormalizer with the transformed canonical strings
        """
        # Transform only the tag normalizer component
        tag_normalizer = self.tag_normalizer.map_outputs(mapfun)
        
        # Create a new normalizer with the transformed tag normalizer
        return type(self)((tag_normalizer, self.implications, self.implications_rej))

    def get_implied(self, tag: int | str) -> list[int]:
        """
        Get the list of tag ids implied by the given tag.
        
        Args:
            tag: Either a tag id or a tag string
            
        Returns:
            A list of tag ids that are implied by the given tag,
            or an empty tuple if there are no implications
        """
        if isinstance(tag, int):
            # Look up implications for a tag id
            return self.implications.get(tag, ())
        else:
            # Look up implications for a tag string (without id)
            return self.implications_rej.get(tag, ())

    def encode(
        self,
        tags: list[str],
        keep_implied: bool | set[int] = False,
        max_antecedent_rank: int | None = None,
        drop_antecedent_rank: int | None = None,
    ) -> tuple[list[int | str], set[int]]:
        """
        Encode a list of tags as numerical IDs and filter out implied tags based on configured rules.
        
        This method performs several key operations:
        1. Converts known tags to their numerical IDs
        2. Preserves unknown tags as strings (they won't be lost)
        3. Processes tag implications based on frequency thresholds
        4. Optionally filters out tags that are implied by other tags
        
        Args:
            tags: List of string tags to encode and process
            keep_implied: Controls which implied tags to keep:
                - False: Remove all implied tags
                - True: Keep all implied tags
                - set[int]: Keep only the specified implied tag IDs
            max_antecedent_rank: Only consider implications from tags with rank <= this value.
                                 Higher rank = less frequent tag. None means consider all.
            drop_antecedent_rank: Don't drop antecedent tags with rank <= this value,
                                 even if they only exist to imply other tags.
                                 None means don't drop any antecedents.
        
        Returns:
            A tuple containing:
            - A list of tag IDs (integers) for known tags and strings for unknown tags
            - A set of implied tag IDs that were filtered (or would have been filtered)
        
        Note:
            Unknown tags (not found in tag2idx) are preserved as strings throughout the
            entire process and returned as strings in the final result.
        """
        tag2idx = self.tag_normalizer.tag2idx
        N = len(tag2idx)
        max_antecedent_rank = max_antecedent_rank or N + 1
        drop_antecedent_rank = drop_antecedent_rank or N + 1
        get_implied = self.implications.get
        get_implied_rej = self.implications_rej.get

        # Convert tags to IDs where possible, keeping unknown tags as strings
        # tag2idx.get(tag, tag) returns the ID if tag exists in mapping, otherwise returns the original string
        stack = [tag2idx.get(tag, tag) for tag in tags[::-1]]
        implied = set()  # Set to track all implied tags encountered
        res = dict()  # dict as a cheap ordered set to maintain insertion order while removing duplicates
        
        # Process all tags and their implications
        while stack:
            tag = stack.pop()
            if isinstance(tag, int):
                # Known tag (converted to integer ID)
                antecedent_rank = tag  # For known tags, rank = tag ID
                consequents = get_implied(tag)  # Get tags implied by this tag
            else:
                # Unknown tag (still a string) - might be a very rare tag without an ID
                antecedent_rank = N  # Treat unknown tags as having maximum rank (least frequent)
                consequents = get_implied_rej(tag)  # Check if this unknown tag implies any known tags
            
            if consequents:  # If this tag implies other tags
                if antecedent_rank < max_antecedent_rank:
                    # If tag is common enough (low rank), mark its implications to be filtered
                    implied.update(consequents)
                else:
                    # If tag is uncommon (high rank), keep its implications and consider dropping the tag itself
                    # Add implied tags to stack to ensure they're included in result
                    stack.extend(consequents)
                    # If tag is very rare (higher than drop threshold), don't include it in result
                    # This prevents keeping tags that only exist to imply other tags
                    if antecedent_rank >= drop_antecedent_rank:
                        continue
            
            # Add this tag to result set
            res[tag] = None
        
        # Convert the result dict keys back to a list
        res = res.keys()

        # Filter implied tags based on keep_implied parameter
        if not keep_implied:
            # Remove all implied tags
            res = [t for t in res if t not in implied]
        elif isinstance(keep_implied, set):
            # Keep only specified implied tags
            res = [t for t in res if t not in implied or t in keep_implied]
        else:
            # Keep all tags (including implied ones)
            res = list(res)
            
        return res, implied

    def decode(self, tags: Iterable[int | str]) -> list[str]:
        idx2tag = self.tag_normalizer.idx2tag
        return [idx2tag[t] if isinstance(t, int) else t for t in tags]

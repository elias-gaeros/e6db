#!/usr/bin/env python3

import argparse
import logging
import re
import sys
import time
from collections import Counter
from functools import cache
from itertools import chain
from pathlib import Path

import e6db
from e6db.utils import TagSetNormalizer, tag_categories, tag_category2id

data_dir = Path(__file__).resolve().parent / "data"


def make_tagset_normalizer(warn_conflict=True) -> TagSetNormalizer:
    """
    Create a TagSetNormalizer for encoding/decoding tags to and from integers.
    Pre-configures it with more aliases and customize the spelling of some tags in the output.
    """
    tagset_normalizer = TagSetNormalizer(data_dir)
    tagid2cat = tagset_normalizer.tag_normalizer.tag_categories

    cat_artist = tag_category2id["artist"]
    cat_lore = tag_category2id["lore"]

    @cache
    def tag_mapfun(tag_underscores, tid):
        """
        Maps raw e621 tags to more natural forms. The input will be from:

        * The list output tag strings,
        * Keys from the dictionary mapping strings to ids, contains canonical
          tag and aliases,
        * Implication source tags that are not used frequently enough to get an
          id.

        Returns a list, where the first string is the canonical tag used in the
        output, the others are additional aliases used for recognizing the tag.
        """
        cat = tagid2cat[tid] if tid is not None else -1
        tag = tag_underscores.replace("_", " ")
        tags = [tag, tag_underscores]
        if cat == cat_artist:
            if not tag.startswith("by "):
                # 'by ' is used in the output tags
                tag_without_suffix = tag.removesuffix(" (artist)")
                tags.insert(0, f"by {tag_without_suffix}")
            if not tag.endswith("(artist)"):
                artist = tag.removeprefix("by ")
                tags.append(f"{artist} (artist)")
        elif cat == cat_lore and not tag.endswith(" (lore)"):
            tags.append(f"{tag} (lore)")

        # Recognize tags where ':' were replaced by a space (aspect ratio)
        if ":" in tag:
            tags.append(tag.replace(":", " "))

        # Recognize tags that have escaped parentheses
        for t in tags.copy():
            escaped = t.replace("(", "\\(").replace(")", "\\)")
            if escaped != t:
                tags.append(escaped)

        # Example of debugging:
        # if "digital media" in tag:
        #     print(tags)
        return tags

    tagset_normalizer = tagset_normalizer.map_tags(
        tag_mapfun,
        # on_conflictc choices: "silent", "overwrite", "overwrite_rarest",
        # warn", "raise", use "warn" to debug conflicts.
        on_conflict="warn" if warn_conflict else "overwrite_rarest",
    )

    # Add some underscores back in the output, for example "rating explicit"
    # will be exported as "rating_explicit"
    tag_normalizer = tagset_normalizer.tag_normalizer
    tag_normalizer.rename_output("rating explicit", "rating_explicit")
    tag_normalizer.rename_output("rating questionable", "rating_questionable")
    tag_normalizer.rename_output("rating safe", "rating_safe")

    # Custom mappings, for example "explicit" will be interpreted as
    # "rating_explicit"
    tag_normalizer.add_input_mappings("explicit", "rating_explicit")
    tag_normalizer.add_input_mappings("score_explicit", "rating_explicit")
    tag_normalizer.add_input_mappings("safe", "rating_safe", on_conflict="overwrite")
    tag_normalizer.add_input_mappings("score_safe", "rating_safe")
    tag_normalizer.add_input_mappings(
        "questionable", "rating_questionable", on_conflict="overwrite"
    )
    tag_normalizer.add_input_mappings("score_questionable", "rating_questionable")

    return tagset_normalizer


def make_blacklist(
    tagset_normalizer: TagSetNormalizer,
    additional_tags=None,
    additional_regexps=None,
    override_base=False,
):
    if override_base:
        blacklist = set()
        re_blacklist = set()
    else:
        # Base blacklist
        blacklist = {
            "invalid tag",
            "by conditional dnp",
            "hi res",
            "absurd res",
            "superabsurd res",
            "4k",
            "uncensored",
            "ambiguous gender",
            "translation edit",
            "story in description",
            "non- balls",
            "non- nipples",
            "non- breasts",
            "feet out of frame",
        }
        # Base regexp
        re_blacklist = {r"(\d+|\d+:\d+)"}

    # Add additional tags and regexps
    if additional_tags:
        blacklist.update(additional_tags)
    if additional_regexps:
        re_blacklist.update(additional_regexps)

    all_tags = tagset_normalizer.tag_normalizer.idx2tag

    # Apply regexp blacklist
    for pattern in re_blacklist:
        re_pattern = re.compile(pattern)
        blacklist.update(t for t in all_tags if re_pattern.fullmatch(t))

    # blacklist tags ending with ' at source'
    blacklist.update(t for t in all_tags if t.endswith(" at source"))

    # Encode the blacklist to ids
    blacklist, implied = tagset_normalizer.encode(blacklist)

    # Also blacklist tags implied by blacklisted tags
    blacklist = set(blacklist) | implied

    return blacklist


RE_SEP = re.compile(r"[,\n]")  # Split on commas and newlines


def load_caption(fp: Path):
    """
    Load caption from file.
    Caption are formatted like this: tag1, tag2, caption1., caption2.
    """
    tags, captions = [], []
    with open(fp, "rt") as fd:
        for chunk in RE_SEP.split(fd.read()):
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.endswith("."):
                captions.append(chunk)
            else:
                tags.append(chunk)
    return tags, captions


def process_directory(
    dataset_root: Path,
    output_dir: Path,
    tagset_normalizer: TagSetNormalizer,
    blacklist: set = set(),
    keep_implied=True,
):
    counter = Counter()
    implied_counter = Counter()
    processed_files = 0
    skipped_files = 0
    for file in chain(dataset_root.glob("**/*.txt"), dataset_root.glob("**/*.cap*")):
        if "sample-prompts" in file.name:
            skipped_files += 1
            continue
        tags, captions = load_caption(file)
        orig_tags = tags

        # Convert tags to ids, separate implied tags
        tags, implied = tagset_normalizer.encode(tags, keep_implied=keep_implied)
        tags = [t for t in tags if t not in blacklist]

        # Count tags
        counter.update(tags)
        implied_counter.update(implied)

        # Convert back to strings
        tags = tagset_normalizer.decode(tags)
        if tags == orig_tags:
            skipped_files += 1
            continue

        # Write output
        output_file = output_dir / file.relative_to(dataset_root)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result = ", ".join(chain(tags, captions))
        with open(output_file, "wt") as fd:
            fd.write(result)
        processed_files += 1

    return counter, implied_counter, processed_files, skipped_files


def print_topk(counter, tagset_normalizer, n=10, categories=None, implied=False):
    if implied:
        implied = "implied "
    else:
        implied = ""
    if categories:
        category_names = ", ".join(categories)
        print(f"\n🔝 Top {n} most common {implied}tags in categories: {category_names}")
    else:
        print(f"\n🔝 Top {n} most common {implied}tags:")

    filtered_counter = counter
    if categories:
        filtered_counter = Counter()
        for tag, count in counter.items():
            if isinstance(tag, int):
                cat = tag_categories[
                    tagset_normalizer.tag_normalizer.tag_categories[tag]
                ]
                if cat in categories:
                    filtered_counter[tag] = count
            elif "unknown" in categories:
                filtered_counter[tag] = count

    for tag, count in filtered_counter.most_common(n):
        if isinstance(tag, int):
            tag_string = tagset_normalizer.tag_normalizer.decode(tag)
            cat = tag_categories[tagset_normalizer.tag_normalizer.tag_categories[tag]]
            print(f"   {tag_string:<30} count={count:<7} (e621:{cat})")
        else:
            print(f"   {tag:<30} count={count:<7} (unknown)")


def print_blacklist(blacklist, tagset_normalizer):
    print("\n🚫 Blacklisted tags:")
    for tag_str in sorted(tagset_normalizer.decode(blacklist)):
        print(f"   {tag_str}")


def setup_logger(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="🏷️  Tag Normalizer - Clean and normalize your tags with ease!"
    )
    parser.add_argument(
        "input_dir", type=Path, help="Input directory containing tag files"
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for normalized tag files"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "-i", "--keep-implied", action="store_true", help="Keep implied tags"
    )
    parser.add_argument(
        "-b",
        "--additional-blacklist",
        action="append",
        help="Additional tags to add to the blacklist",
    )
    parser.add_argument(
        "-r",
        "--additional-blacklist-regexp",
        action="append",
        help="Additional regular expressions for blacklisting tags",
    )
    parser.add_argument(
        "-O",
        "--override-base-blacklist",
        action="store_true",
        help="Override the base blacklist and regexp with only the provided additions",
    )
    parser.add_argument(
        "--print-blacklist",
        action="store_true",
        help="Print the effective list of blacklisted tags",
    )
    parser.add_argument(
        "-k",
        "--print-topk",
        type=int,
        nargs="?",
        const=100,
        help="Print the N most common tags (default: 100 if flag is used without a value)",
    )
    parser.add_argument(
        "-j",
        "--print-implied-topk",
        type=int,
        nargs="?",
        const=100,
        help="Print the N most common implied tags (default: 100 if flag is used without a value)",
    )
    parser.add_argument(
        "-c",
        "--stats-categories",
        action="append",
        choices=list(tag_category2id.keys()) + ["unknown"],
        help="Restrict tag count printing to specific categories or 'unknown'",
    )
    parser.add_argument(
        "--print-conflicts",
        action="store_true",
        help="Print the conflicts encountered during the construction of the normalization mapping (useful for debugging it)",
    )
    args = parser.parse_args()

    logger = setup_logger(args.verbose)

    logger.info("🚀 Starting Tag Normalizer")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    logger.info("🔧 Initializing tag normalizer...")
    start_time = time.time()
    tagset_normalizer = make_tagset_normalizer(warn_conflict=args.print_conflicts)
    logging.info(f"  Data loaded in {time.time() - start_time:.2f} seconds")

    logger.info("🚫 Creating blacklist...")
    blacklist = make_blacklist(
        tagset_normalizer,
        additional_tags=args.additional_blacklist,
        additional_regexps=args.additional_blacklist_regexp,
        override_base=args.override_base_blacklist,
    )
    logger.info(f"Blacklist size: {len(blacklist)} tags")
    if args.print_blacklist:
        print_blacklist(blacklist, tagset_normalizer)

    logger.info("🔍 Processing files...")
    start_time = time.time()
    counter, implied_counter, processed_files, skipped_files = process_directory(
        args.input_dir,
        args.output_dir,
        tagset_normalizer,
        blacklist=blacklist,
        keep_implied=args.keep_implied,
    )

    logger.info(
        f"✅ Processing complete! Time taken: {time.time() - start_time:.2f} seconds"
    )
    logger.info(f"Files processed: {processed_files}")
    logger.info(f"Files skipped (no changes): {skipped_files}")
    logger.info(f"Total unique tags: {len(counter)}")
    logger.info(f"Total tag occurrences: {sum(counter.values())}")
    if args.print_topk:
        print_topk(
            counter,
            tagset_normalizer,
            args.print_topk,
            args.stats_categories,
        )
    if args.print_implied_topk:
        print_topk(
            implied_counter,
            tagset_normalizer,
            args.print_implied_topk,
            implied=True,
        )

    logger.info("👋 Tag Normalizer finished. Have a great day!")


if __name__ == "__main__":
    main()

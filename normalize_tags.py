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


def make_tagset_normalizer(warn_conflict=True):
    """
    Create a TagSetNormalizer for encoding/decoding tags to and from integers.
    Pre-configures it with more aliases and customize the spelling of some tags in the output.
    """
    cat_artist = e6db.utils.tag_category2id["artist"]
    cat_lore = e6db.utils.tag_category2id["lore"]

    tagset_normalizer = e6db.utils.TagSetNormalizer(data_dir)
    tagid2cat = tagset_normalizer.tag_normalizer.tag_categories

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


def make_blacklist(tagset_normalizer):
    # Manual blacklist: a list of e621 tags or unknown tag strings
    blacklist = r"""
    invalid tag, by conditional dnp, 
    hi res, absurd res, superabsurd res, 4k,
    uncensored, ambiguous gender,
    translation edit, story in description,
    non- balls, non- nipples, non- breasts, feet out of frame
    """
    blacklist = (t.strip() for t in blacklist.split(","))
    blacklist = set(t for t in blacklist if len(t) > 0)
    # multiline is ok, but don't forget the comma on line endings
    assert not any("\n" in t for t in blacklist)

    # blacklist years, digits only tags, and aspect ratios
    all_tags = tagset_normalizer.tag_normalizer.idx2tag
    RE_BLACKLIST = re.compile(r"(\d+|\d+:\d+)")
    blacklist.update(t for t in all_tags if RE_BLACKLIST.fullmatch(t))

    # blacklist tags ending with ' at source'
    blacklist.update(t for t in all_tags if t.endswith(" at source"))

    # Encode the blacklist to ids
    blacklist, implied = tagset_normalizer.encode(blacklist)

    # Also blacklist tags implied by blacklisted tags
    blacklist = set(blacklist) | implied

    return blacklist


def load_caption(fp):
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


RE_SEP = re.compile(r"[,\n]")  # Split on commas and newlines


def process_directory(dataset_root, output_dir, tagset_normalizer, blacklist):
    counter = Counter()
    processed_files = 0
    skipped_files = 0
    for file in chain(dataset_root.glob("**/*.txt"), dataset_root.glob("**/*.cap*")):
        if "sample-prompts" in file.name:
            skipped_files += 1
            continue
        tags, captions = load_caption(file)
        orig_tags = tags

        # Convert tags to ids, separate implied tags
        tags, implied = tagset_normalizer.encode(tags)
        tags = [t for t in tags if t not in blacklist]

        # Count tags
        counter.update(tags)

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

    return counter, processed_files, skipped_files


def print_stats(counter, tagset_normalizer, n=10, print_common=False, categories=None):
    total_tags = sum(counter.values())
    unique_tags = len(counter)
    print(f"\n📊 Tag Statistics:")
    print(f"   Total tags processed: {total_tags}")
    print(f"   Unique tags: {unique_tags}")

    if print_common:
        if categories:
            category_names = ", ".join(categories)
            print(f"\n🔝 Top {n} most common tags in categories: {category_names}")
        else:
            print(f"\n🔝 Top {n} most common tags:")

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
                cat = tag_categories[
                    tagset_normalizer.tag_normalizer.tag_categories[tag]
                ]
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
        "--print-blacklist",
        action="store_true",
        help="Print the list of blacklisted tags",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print tag statistics after processing",
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
        "-c",
        "--stats-categories",
        nargs="+",
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

    try:
        start_time = time.time()

        logger.info("🔧 Initializing tag normalizer...")
        tagset_normalizer = make_tagset_normalizer(warn_conflict=args.print_conflicts)

        logger.info("🚫 Creating blacklist...")
        blacklist = make_blacklist(tagset_normalizer)

        logger.info(f"Blacklist size: {len(blacklist)} tags")

        if args.print_blacklist:
            print_blacklist(blacklist, tagset_normalizer)

        logger.info("🔍 Processing files...")
        counter, processed_files, skipped_files = process_directory(
            args.input_dir, args.output_dir, tagset_normalizer, blacklist
        )

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"✅ Processing complete! Time taken: {duration:.2f} seconds")
        logger.info(f"Files processed: {processed_files}")
        logger.info(f"Files skipped (no changes): {skipped_files}")
        logger.info(f"Total unique tags: {len(counter)}")
        logger.info(f"Total tag occurrences: {sum(counter.values())}")

        if args.print_stats or args.print_topk:
            print_stats(
                counter,
                tagset_normalizer,
                args.print_topk,
                bool(args.print_topk),
                args.stats_categories,
            )

    except Exception as e:
        logger.error(f"❌ An error occurred: {str(e)}")
        sys.exit(1)

    logger.info("👋 Tag Normalizer finished. Have a great day!")


if __name__ == "__main__":
    main()

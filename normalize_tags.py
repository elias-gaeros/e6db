#!/usr/bin/env python3

import argparse
import logging
import math
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

CONFIG_OPEN_MODE = "rb"
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        CONFIG_OPEN_MODE = "rt"
        import toml as tomllib

from e6db.utils import (
    TagSetNormalizer,
    tag_categories,
    tag_category2id,
    tag_freq_to_rank,
)

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent / "data"
RE_PARENS_SUFFIX = re.compile(r"_\([^)]+\)$")


def make_tagset_normalizer(config: dict) -> TagSetNormalizer:
    """
    Create a TagSetNormalizer for encoding/decoding tags to and from integers.
    Configures it based on the provided config.
    """
    # This loads all the aliases and implications
    tagset_normalizer = TagSetNormalizer(DATA_DIR)

    tagid2cat = tagset_normalizer.tag_normalizer.tag_categories
    cat_artist = tag_category2id["artist"]
    cat2suffix = {
        tag_category2id["character"]: "_(character)",
        tag_category2id["lore"]: "_(lore)",
        tag_category2id["species"]: "_(species)",
        tag_category2id["copyright"]: "_(copyright)",
    }

    # Create additional aliases for tags using simple rules
    def input_map(tag, tid):
        # Make an alias without parentheses, it might conflict but we'll handle
        # it depending on `on_alias_conflict` config value.
        without_suffix = RE_PARENS_SUFFIX.sub("", tag)
        had_suffix = tag != without_suffix
        if had_suffix:
            yield without_suffix

        # Add an alias with the suffix (special case for artist)
        cat = tagid2cat[tid] if tid is not None else -1
        if cat == cat_artist:
            artist = without_suffix.removeprefix("by_")
            if artist != without_suffix:
                yield artist
                if not had_suffix:
                    yield f"{artist}_(artist)"
            else:
                yield f"by_{artist}"
                if not had_suffix:
                    yield f"by_{artist}_(artist)"
        elif not had_suffix:
            suffix = cat2suffix.get(cat)
            if suffix is not None:
                yield f"{without_suffix}{suffix}"

        # Recognize tags where ':' were replaced by a space (aspect ratio)
        if ":" in tag:
            yield tag.replace(":", "_")

    on_alias_conflict = config.get("on_alias_conflict", None)
    tagset_normalizer = tagset_normalizer.map_inputs(
        input_map,
        # on_conflict choices: "silent", "overwrite", "overwrite_rarest",
        # "warn", "raise", use "warn" to debug conflicts.
        on_conflict=on_alias_conflict or "ignore",
    )
    tag_normalizer = tagset_normalizer.tag_normalizer
    tag2id = tag_normalizer.tag2idx

    # Apply custom input mappings
    for antecedent, consequent in config.get("aliases", {}).items():
        antecedent = antecedent.replace(" ", "_")
        consequent = consequent.replace(" ", "_")
        tag_normalizer.add_input_mappings(
            antecedent, consequent, on_conflict=on_alias_conflict or "warn"
        )
    for antecedent, consequent in config.get("aliases_overrides", {}).items():
        antecedent = antecedent.replace(" ", "_")
        consequent = consequent.replace(" ", "_")
        tag_normalizer.add_input_mappings(
            antecedent, consequent, on_conflict="overwrite"
        )

    # Apply custom output renames as opposite aliases to ensure
    # idempotence:
    output_renames = {
        old.replace(" ", "_"): new.replace(" ", "_")
        for old, new in config.get("renames", {}).items()
    }
    for old, new in output_renames.items():
        tag_normalizer.add_input_mappings(new, old)

    # Remove specified aliases
    for tag in config.get("remove_aliases", []):
        tag = tag.replace(" ", "_")
        tag_normalizer.remove_input_mappings(tag)

    # Apply rule based output renames
    remove_parens = config.get("remove_parens", True)
    artist_by_prefix = config.get("artist_by_prefix", True)

    def map_output(tag, tid):
        cat = tagid2cat[tid] if tid is not None else -1
        if remove_parens:
            without_suffix = tag.removesuffix(f"_({tag_categories[cat]})")
            if without_suffix != tag and tag2id.get(without_suffix) == tid:
                tag = without_suffix
        if cat == cat_artist and artist_by_prefix and not tag.startswith("by_"):
            tag_wby = f"by_{tag}"
            if tag2id.get(tag_wby) == tid:
                tag = tag_wby
        return tag

    tagset_normalizer = tagset_normalizer.map_outputs(map_output)
    tag_normalizer = tagset_normalizer.tag_normalizer
    tag2id = tag_normalizer.tag2idx

    # Apply custom output renames
    for old, new in output_renames.items():
        if tag2id[old] == tag2id[new]:
            tag_normalizer.rename_output(old, new)
        else:
            logger.warning(
                f"Cannot rename {old} -> {new}: old tag id={tag2id[old]} vs. new tag id={tag2id[new]})"
            )

    return tagset_normalizer


def make_blacklist(
    tagset_normalizer: TagSetNormalizer,
    config: dict,
    print_blacklist=False,
):
    if print_blacklist:
        print("\n🚫 Blacklisted tags:")

    all_tags = tagset_normalizer.tag_normalizer.idx2tag
    encode = tagset_normalizer.tag_normalizer.encode
    decode = tagset_normalizer.tag_normalizer.decode
    get_implied = tagset_normalizer.get_implied

    blacklist = set()
    for tag in config.get("blacklist", ["invalid tag"]):
        tag = tag.replace(" ", "_")
        encoded_tag = encode(tag, tag)
        blacklist.add(encoded_tag)
        if print_blacklist:
            decoded_tag = decode(encoded_tag)
            if tag != decoded_tag:
                print(f"   {tag} -> {decoded_tag}")
            else:
                print(f"   {tag}")

    for regexp in config.get("blacklist_regexp", []):
        regexp = regexp.replace(" ", "_")
        cregexp = re.compile(regexp)
        for tid, tag in enumerate(all_tags):
            if cregexp.fullmatch(tag):
                blacklist.add(tid)
                if print_blacklist:
                    print(f'   {tag} (r"{regexp})"')

    implied = set()
    if config.get("blacklist_implied", True):
        for tag in blacklist:
            tag_implied = get_implied(tag)
            implied.update(tag_implied)
            if print_blacklist:
                for implied_tag in tag_implied:
                    print(f"   {decode(implied_tag)} (implied by {decode(tag)})")
    blacklist |= implied

    tagid2cat = tagset_normalizer.tag_normalizer.tag_categories
    blacklist_categories = {
        tag_category2id[c] for c in config.get("blacklist_categories", ["pool"])
    }
    if blacklist_categories:
        for tid, cat in enumerate(tagid2cat):
            if cat in blacklist_categories:
                blacklist.add(tid)
                if print_blacklist:
                    print(f"   {tagid2cat[tid]} (cat:{tag_categories[cat]})")

    return blacklist


RE_SEP = re.compile(r"[,\n]")  # Split on commas and newlines
RE_ESCAPES = re.compile(r"\\+?(?=[():])")  # Match backslash escapes before :()


def process_directory(
    dataset_root: Path,
    output_dir: Path,
    tagset_normalizer: TagSetNormalizer,
    config: dict,
    blacklist: set = set(),
):
    n_tags = len(tagset_normalizer.tag_normalizer.tag2idx)
    use_underscores = config.get("use_underscores", False)
    keep_underscores = set(config.get("keep_underscores", ()))

    keep_implied = config.get("keep_implied", False)
    if isinstance(keep_implied, list):
        encode = tagset_normalizer.tag_normalizer.encode
        keep_implied = {encode(t, t) for t in keep_implied}
    max_antecedent_rank = n_tags + 1
    min_antecedent_freq = config.get("min_antecedent_freq", 0)
    if min_antecedent_freq >= 1.0:
        max_antecedent_rank = math.ceil(tag_freq_to_rank(min_antecedent_freq))
    drop_antecedent_rank = n_tags + 1
    drop_antecedent_freq = config.get("drop_antecedent_freq", 0)
    if drop_antecedent_freq >= 1.0:
        drop_antecedent_rank = math.ceil(tag_freq_to_rank(drop_antecedent_freq))
        if drop_antecedent_rank < max_antecedent_rank:
            logger.warning(
                "drop_antecedents_freq must be smaller or equal to min_antecedent_freq"
            )
    logger.debug(f"{keep_implied=} {max_antecedent_rank=} {drop_antecedent_rank=}")

    logger.debug(f"🔍 Gathering file list...")
    files = walk_directory(dataset_root, config)
    logger.info("💾 Processing %d files...", len(files))

    # Running stats
    counter = Counter()
    implied_counter = Counter()
    processed_files = 0
    skipped_files = 0
    blacklist_instances = 0
    implied_instances = 0
    for file in tqdm(files):
        try:
            with open(file, "rt", encoding="utf-8") as fd:
                content = fd.read()
        except ValueError as e:
            logging.warning('Failed to read "%s": %s', file, e)
            continue

        orig_tags = tags = []
        for chunk in RE_SEP.split(content):
            chunk = chunk.strip()
            if not chunk:
                continue
            tags.append(chunk)
        original_len = len(tags)

        # Convert tags to ids, separate implied tags
        tags = [RE_ESCAPES.sub("", t.lower().replace(" ", "_")) for t in tags]

        # Encode to integer ids and strip implied tags
        tags, implied = tagset_normalizer.encode(
            tags,
            keep_implied=keep_implied,
            max_antecedent_rank=max_antecedent_rank,
            drop_antecedent_rank=drop_antecedent_rank,
        )
        implication_filtered_len = len(tags)
        implied_instances += original_len - implication_filtered_len

        # Remove blacklisted tags
        tags = [t for t in tags if t not in blacklist]
        blacklist_instances += implication_filtered_len - len(tags)

        # Count tags
        counter.update(tags)
        implied_counter.update(implied)

        # Convert back to strings
        tags = tagset_normalizer.decode(tags)
        if not use_underscores:
            tags = [
                t.replace("_", " ") if t not in keep_underscores else t for t in tags
            ]
        if tags == orig_tags:
            skipped_files += 1
            continue

        # Write output
        output_file = output_dir / file.relative_to(dataset_root)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "wt", encoding="utf-8") as fd:
            fd.write(", ".join(tags))
        processed_files += 1

    return dict(
        counter=counter,
        implied_counter=implied_counter,
        processed_files=processed_files,
        skipped_files=skipped_files,
        blacklist_instances=blacklist_instances,
        implied_instances=implied_instances,
    )


def walk_directory(dataset_root: Path, config: dict):
    exclude_re = re.compile(
        config.get("exclude_filename_regexp", r".*samples?-prompts?.*")
    )
    include_re = re.compile(config.get("include_filename_regexp", r".*?\.(txt|cap.*)$"))
    res = []
    for root, dirs, files in os.walk(dataset_root, followlinks=True):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if files:
            root = Path(root)
        for file in files:
            if not include_re.fullmatch(file) or exclude_re.fullmatch(file):
                continue
            res.append(root / file)
    return res


def print_topk(
    counter: Counter,
    tagset_normalizer: TagSetNormalizer,
    config: dict,
    n=10,
    categories=None,
    implied=False,
):
    if implied:
        implied = "implied "
    else:
        implied = ""
    if categories:
        category_names = ", ".join(categories)
        print(f"\nTop {n} most common {implied}tags in categories: {category_names}")
    else:
        print(f"\nTop {n} most common {implied}tags:")

    use_underscores = config.get("use_underscores", True)
    keep_underscores = config.get("keep_underscores", set())

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
            elif "unknown" in categories and tag[-1] != ".":
                filtered_counter[tag] = count

    for tag, count in filtered_counter.most_common(n):
        if isinstance(tag, int):
            tag_string = tagset_normalizer.tag_normalizer.decode(tag)
            cat = tag_categories[tagset_normalizer.tag_normalizer.tag_categories[tag]]
            source = f"e621:{cat}"
        else:
            tag_string = tag
            source = "unknown"
        if not use_underscores and tag_string not in keep_underscores:
            tag_string = tag_string.replace("_", " ")
        print(f"   {tag_string:<30} count={count:<7} ({source})")


def main():
    parser = argparse.ArgumentParser(
        description="🏷️  Tag Normalizer - Clean and normalize your tags with ease!"
    )
    parser.add_argument(
        "input_dir", type=Path, help="Input directory containing tag files"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for normalized tag files",
        nargs="?",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Toml configuration file, defaults to output_dir/normalize.toml, input_dir/normalize.toml or ./normalize.toml",
        default=None,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Don't ask for confirmation for clobbering input files",
    )
    parser.add_argument(
        "-b",
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
        "-s",
        "--stats-categories",
        action="append",
        choices=list(tag_category2id.keys()) + ["unknown"],
        help="Restrict tag count printing to specific categories or 'unknown'",
    )
    args = parser.parse_args()
    setup_logger(args.verbose)

    # Validate input/output directories
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        exit(1)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create output directory {output_dir}: {e}")
        exit(1)
    logger.info("🚀 Starting Tag Normalizer")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Check for overwriting input
    if input_dir == output_dir:
        input_lookup = output_lookup = find_files_up_hierarchy(
            input_dir, [CONFIG_NAME, ".git"]
        )
        git_dir = input_lookup.get(".git")
        if git_dir is not None and is_clean_git_repo(git_dir):
            logger.warning(
                "Oh I see you're using git for the dataset! Overwriting input files. This is fine! 🔥"
            )
        elif not args.force and not ask_for_confirmation(
            "The input will be modified in place. Are you sure you want to continue? 💾❌⚠️❓",
        ):
            exit(0)
    else:
        input_lookup = find_files_up_hierarchy(input_dir, [CONFIG_NAME])
        output_lookup = find_files_up_hierarchy(output_dir, [CONFIG_NAME])

    # Load config file
    for config_path in [
        args.config,
        output_lookup.get(CONFIG_NAME),
        input_lookup.get(CONFIG_NAME),
        find_files_up_hierarchy(Path("."), [CONFIG_NAME]).get(CONFIG_NAME),
    ]:
        if config_path is None:
            continue
        if config_path.exists():
            config_path = config_path.resolve()
            break
    else:
        logger.error(f"Could not find a config file in {input_dir}, {output_dir} or ./")
        exit(1)
    logger.info(f"🔧 Using config file: {config_path}")
    with open(
        config_path,
        CONFIG_OPEN_MODE,
        encoding="utf-8" if CONFIG_OPEN_MODE == "rt" else None,
    ) as f:
        config = tomllib.load(f)

    logger.info("🔧 Initializing tag normalizer...")
    start_time = time.time()
    tagset_normalizer = make_tagset_normalizer(config)
    logger.info(f"✅ Data loaded in {time.time() - start_time:.2f} seconds")

    logger.info("🚫 Creating blacklist...")
    blacklist = make_blacklist(
        tagset_normalizer,
        config,
        print_blacklist=args.print_blacklist,
    )
    logger.info(f"Blacklist size: {len(blacklist)} tags")

    start_time = time.time()
    stats = process_directory(
        input_dir,
        output_dir,
        tagset_normalizer,
        config,
        blacklist=blacklist,
    )

    logger.info(
        f"✅ Processing complete! Time taken: {time.time() - start_time:.2f} seconds"
    )
    logger.info(f"Files modified: {stats['processed_files']}")
    logger.info(f"Files skipped (no changes): {stats['skipped_files']}")
    counter = stats["counter"]
    logger.info(f"Unique tags: {len(counter)}")
    logger.info(f"Tag occurrences: {sum(counter.values())}")
    unknown_counter = [count for t, count in counter.items() if not isinstance(t, int)]
    logger.info(f"Unknown tags: {len(unknown_counter)}")
    logger.info(f"Unknown tags occurrences: {sum(unknown_counter)}")
    logger.info(f"Removed by blacklist: {stats['blacklist_instances']}")
    logger.info(f"Removed by implication: {stats['implied_instances']}")
    if args.print_topk or args.stats_categories:
        if not args.print_topk:
            args.print_topk = 100
        print_topk(
            counter,
            tagset_normalizer,
            config,
            n=args.print_topk,
            categories=args.stats_categories,
        )
    if args.print_implied_topk:
        print_topk(
            stats["implied_counter"],
            tagset_normalizer,
            config,
            n=args.print_implied_topk,
            implied=True,
        )

    logger.info("👋 Tag Normalizer finished. Have a great day!")


CONFIG_NAME = "normalize.toml"


def setup_logger(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def ask_for_confirmation(prompt, default=False):
    if default:
        prompt = f"{prompt} (Y/n): "
    else:
        prompt = f"{prompt} (y/N): "

    response = input(prompt).strip().lower()

    if response not in "yn":
        return default

    return response == "y"


def find_files_up_hierarchy(start_path: str, file_names: list[str]) -> dict[str, Path]:
    start_path = Path(start_path).resolve()
    found_files = {}

    current_path = start_path
    while True:
        for file_name in file_names:
            file_path = current_path / file_name
            if file_path.exists() and file_name not in found_files:
                found_files[file_name] = file_path

        parent = current_path.parent
        if len(found_files) == len(file_names) or parent == current_path:
            break

        current_path = parent

    return found_files


def is_clean_git_repo(directory):
    git_cmd = ["git", "-C", str(directory.parent.resolve())]
    try:
        # Check if it's a Git repository and if it's clean
        subprocess.run(
            [*git_cmd, "rev-parse", "--is-inside-work-tree"],
            check=True,
            capture_output=True,
        )
        status_output = subprocess.run(
            [*git_cmd, "status", "--porcelain"], check=True, capture_output=True
        ).stdout
        n_modified = len(status_output.splitlines())
        if n_modified == 0:
            return True
        else:
            logger.warning("🚫 Found a git repo, but it's dirty.")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


if __name__ == "__main__":
    main()

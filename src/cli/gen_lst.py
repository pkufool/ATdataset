#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Wei Kang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate an atdataset WebDataset list file.

The generated list is consumed by :class:`atdataset.ATDataset`/
``ATDataloader``.  Each output line contains four whitespace-separated columns::

    <absolute_tar_path> <absolute_manifest_path> <duration_hours> <num_samples>

For every tar shard matched by ``--audio-pattern``, this tool derives a shard key
from the tar basename, finds the matching transcript manifest in ``--txt-dir``
(``.json`` first, then ``.jsonl``), scans the manifest, and writes the total
sample duration in hours plus the number of valid JSON samples.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


def get_manifest_stats(manifest_path: str) -> Tuple[float, int]:
    """Return ``(total_duration_seconds, num_samples)`` for a JSONL manifest.

    Args:
      manifest_path:
        Path to a line-delimited JSON manifest.  Each non-empty line is expected
        to be a JSON object such as::

            {"audio_filepath": "utt.flac", "text": "...", "duration": 3.2}

    Notes:
      * Invalid JSON lines are skipped with a warning.
      * ``num_samples`` counts valid JSON objects, even if a line does not have a
        ``duration`` field.
      * Lines without ``duration`` contribute ``0`` seconds to the total.
    """
    total_duration = 0.0
    num_samples = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Warning: skipping invalid JSON line "
                    f"{manifest_path}:{line_no}: {line}",
                    file=sys.stderr,
                )
                continue
            num_samples += 1
            if "duration" in item:
                try:
                    total_duration += float(item["duration"])
                except (TypeError, ValueError):
                    print(
                        f"Warning: invalid duration in {manifest_path}:{line_no}: "
                        f"{item.get('duration')!r}; treated as 0",
                        file=sys.stderr,
                    )
    return total_duration, num_samples


# Backward-compatible helper name used by some external scripts.
def get_manifest_duration(manifest_path: str) -> float:
    """Return total manifest duration in seconds."""
    duration, _ = get_manifest_stats(manifest_path)
    return duration


def shard_key_from_tar(tar_path: str) -> str:
    """Derive shard key from a tar filename.

    Examples:
      ``train.000000.tar`` -> ``train.000000``
      ``train.000000.tar.gz`` -> ``train.000000``
    """
    name = Path(tar_path).name
    if name.endswith(".tar.gz"):
        return name[: -len(".tar.gz")]
    if name.endswith(".tar"):
        return name[: -len(".tar")]
    return Path(name).stem


def map_key_prefix(key: str, audio_prefix: str = "", txt_prefix: str = "") -> str:
    """Map an audio shard key to a transcript shard key."""
    if not audio_prefix and not txt_prefix:
        return key
    if not audio_prefix or not txt_prefix:
        raise ValueError("--audio-prefix and --txt-prefix must be specified together")
    if not key.startswith(audio_prefix):
        raise ValueError(
            f"Audio key {key!r} does not start with --audio-prefix {audio_prefix!r}"
        )
    return txt_prefix + key[len(audio_prefix) :]


def find_manifest(txt_dir: str, key: str) -> Optional[str]:
    """Find ``key.json`` or ``key.jsonl`` under ``txt_dir``."""
    for suffix in (".json", ".jsonl"):
        path = Path(txt_dir) / f"{key}{suffix}"
        if path.exists():
            return str(path)
    return None


def iter_tar_paths(audio_pattern: str, sort: bool = True) -> Iterable[str]:
    """Yield tar paths matched by a glob pattern."""
    paths = glob.glob(audio_pattern, recursive=True)
    if sort:
        paths = sorted(paths)
    return paths


def generate_lst(args) -> dict:
    """Generate a WebDataset list file from parsed CLI arguments."""
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tar_paths = list(iter_tar_paths(args.audio_pattern, sort=not args.no_sort))
    if not tar_paths:
        print(f"Warning: no tar files matched pattern: {args.audio_pattern}", file=sys.stderr)

    written = 0
    skipped = 0
    total_duration = 0.0
    total_samples = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for tar in tar_paths:
            key = shard_key_from_tar(tar)
            try:
                txt_key = map_key_prefix(key, args.audio_prefix, args.txt_prefix)
            except ValueError as e:
                if args.strict:
                    raise
                print(f"Warning: {e}. Skipping {tar}.", file=sys.stderr)
                skipped += 1
                continue

            txt_path = find_manifest(args.txt_dir, txt_key)
            if txt_path is None:
                msg = (
                    f"Transcript manifest for key {txt_key!r} does not exist in "
                    f"{args.txt_dir}; tried {txt_key}.json and {txt_key}.jsonl"
                )
                if args.strict:
                    raise FileNotFoundError(msg)
                print(f"Warning: {msg}. Skipping {tar}.", file=sys.stderr)
                skipped += 1
                continue

            duration_seconds, num_samples = get_manifest_stats(txt_path)
            total_duration += duration_seconds
            total_samples += num_samples
            written += 1
            out_f.write(
                f"{os.path.abspath(tar)} {os.path.abspath(txt_path)} "
                f"{duration_seconds / 3600.0:.3f} {num_samples}\n"
            )

    summary = {
        "output": str(output_path),
        "matched_tars": len(tar_paths),
        "written": written,
        "skipped": skipped,
        "total_duration_hours": total_duration / 3600.0,
        "total_samples": total_samples,
    }
    print(
        "Generated list: {output}\n"
        "  matched tar files : {matched_tars}\n"
        "  written entries   : {written}\n"
        "  skipped entries   : {skipped}\n"
        "  total duration h  : {total_duration_hours:.3f}\n"
        "  total samples     : {total_samples}".format(**summary)
    )
    return summary


def add_gen_lst_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register command-line arguments and examples for ``gen_lst``."""
    parser.description = "Generate an atdataset tar/manifest list file."
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = """
Output format
-------------
Each line in --output has four whitespace-separated columns:

  /abs/path/to/shard.tar /abs/path/to/shard.jsonl duration_hours num_samples

The first three columns are compatible with the original gen_lst.py output; the
fourth column is the number of valid JSON samples in the transcript manifest.

How matching works
------------------
1. --audio-pattern is expanded as a Python glob. Use quotes in the shell so the
   program, not the shell, receives patterns such as 'data/tars/audios/*.tar'.
2. The shard key is the tar basename without .tar or .tar.gz.
   Example: data/tars/audios/train.000000.tar -> train.000000
3. The transcript manifest is searched as:
     --txt-dir/<key>.json
     --txt-dir/<key>.jsonl
4. If tar and transcript prefixes differ, use --audio-prefix and --txt-prefix.
   Example: audio key audio.000000 and text key text.000000:
     --audio-prefix audio --txt-prefix text

Examples
--------
Generate a list for tars and manifests produced by `atdataset build`:

  atdataset gen_lst --audio-pattern 'data/tars/audios/*.tar' \\
    --txt-dir data/tars/manifests \\
    --output data/tars/data.lst

Generate a list where audio shard names start with `audio` but manifests start
with `text`:

  atdataset gen_lst --audio-pattern 'data/tars/audios/audio.*.tar' \\
    --txt-dir data/tars/manifests \\
    --output data/tars/data.lst \\
    --audio-prefix audio \\
    --txt-prefix text

Fail immediately on missing transcript manifests instead of warning/skipping:

  atdataset gen_lst --audio-pattern 'data/tars/audios/*.tar' \\
    --txt-dir data/tars/manifests \\
    --output data/tars/data.lst \\
    --strict
"""
    parser.add_argument(
        "--audio-pattern",
        required=True,
        help="Glob pattern for input tar shards, e.g. 'data/tars/audios/*.tar'.",
    )
    parser.add_argument(
        "--txt-dir",
        required=True,
        help="Directory containing transcript manifests named <shard-key>.json/jsonl.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output list file path.",
    )
    parser.add_argument(
        "--audio-prefix",
        default="",
        help="Optional prefix in audio tar shard keys to replace.",
    )
    parser.add_argument(
        "--txt-prefix",
        default="",
        help="Optional transcript manifest key prefix replacing --audio-prefix.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error on prefix mismatch or missing manifest instead of skipping.",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Keep glob order instead of sorting matched tar paths.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None):
    """Parse arguments for direct ``gen_lst.py`` execution."""
    parser = argparse.ArgumentParser(
        prog="atdataset gen_lst",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_gen_lst_args(parser)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for direct list generation."""
    args = parse_args(argv)
    generate_lst(args)


if __name__ == "__main__":
    main()

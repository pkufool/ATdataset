#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Wei Kang)
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
"""Build WebDataset tar shards and JSONL manifests for atdataset.

Input manifests are tab-separated text files with either three or five fields::

    id\taudio_path\ttext
    id\taudio_path\ttext\tstart\tduration

The three-field form stores the whole audio file.  The five-field form stores
only the requested segment.  Intermediate files are cached under
``$HOME/.atdataset`` so interrupted builds can be resumed without redoing work.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

# Torch's multithreaded behavior hurts when many worker processes encode audio.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# -----------------------------------------------------------------------------
# Text normalization copied from scirpts/write_tars.py to keep manifest text
# compatible with previously generated atdataset shards.
# -----------------------------------------------------------------------------


def str2bool(v):
    """Parse common string representations of booleans for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


_CJK_WITH_SPACE_RE = re.compile(
    r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF"
    r"\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])\s+"
    r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF"
    r"\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
)
_CJK_CHAR_RE = re.compile(
    r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF"
    r"\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
)


def remove_space_between_CJK_char(line: str) -> str:
    """Remove spaces between adjacent CJK characters."""
    line = line.replace("　", "")
    chars = _CJK_WITH_SPACE_RE.split(line.strip())
    return "".join([ch.strip() for ch in chars]).strip()


def tokenize_by_CJK_char(line: str) -> str:
    """Insert token boundaries around CJK characters."""
    chars = _CJK_CHAR_RE.split(line.strip())
    return " ".join([w.strip() for w in chars if w.strip()])


def norm_text(line: str) -> str:
    """Normalize transcript text before writing the JSONL manifest."""
    return remove_space_between_CJK_char(tokenize_by_CJK_char(line))


@dataclass
class BuildConfig:
    """Options that affect a resumable tar-building run."""

    input: str
    output_dir: str
    tmp_dir: str
    num_tars: Optional[int]
    samples_per_tar: Optional[int]
    duration_per_tar: Optional[float]
    balance_by: str
    format: str
    compression_level: float
    sample_rate: Optional[int]
    seed: int
    split_bits: int
    tar_prefix: str


@dataclass
class SplitPlan:
    """Description of cached split files and their aggregate statistics."""

    mode: str
    requested_num_tars: Optional[int]
    num_tars: int
    total_samples: int
    total_duration: Optional[float]
    samples_per_tar: Optional[int]
    duration_per_tar: Optional[float]
    split_files: List[str]


def atomic_write_json(path: Path, obj) -> None:
    """Atomically write a JSON file, creating its parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, path)


def atomic_replace_text(path: Path, lines: Iterable[str]) -> None:
    """Atomically replace a text file with the provided lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as f:
        for line in lines:
            f.write(line)
        tmp = f.name
    os.replace(tmp, path)


def parse_tsv_manifest_line(
    line: str, line_no: int, input_path: str, require_duration: bool = False
) -> Dict:
    """Parse one input TSV line into the internal sample dictionary."""
    raw = line.rstrip("\n")
    if not raw:
        raise ValueError("empty line")
    parts = raw.split("\t")
    if len(parts) == 3:
        item = {
            "id": parts[0],
            "audio_path": parts[1],
            "text": parts[2],
            "slice_audio": False,
        }
    elif len(parts) == 5:
        item = {
            "id": parts[0],
            "audio_path": parts[1],
            "text": parts[2],
            "start": float(parts[3]),
            "duration": float(parts[4]),
            "slice_audio": True,
        }
    else:
        raise ValueError(
            f"Invalid manifest line {line_no} in {input_path}: expected 3 or 5 "
            f"tab-separated fields, got {len(parts)}"
        )
    if not item["id"]:
        raise ValueError(f"Invalid manifest line {line_no} in {input_path}: empty id")
    if not item["audio_path"]:
        raise ValueError(
            f"Invalid manifest line {line_no} in {input_path}: empty audio_path"
        )
    if require_duration and "duration" not in item:
        item["duration"] = probe_audio_duration(item["audio_path"])
    return item


def probe_audio_duration(audio_path: str) -> float:
    """Return full audio duration in seconds without loading samples if possible."""
    try:
        info = sf.info(audio_path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        info = torchaudio.info(audio_path)
        return float(info.num_frames) / float(info.sample_rate)


def iter_input_manifest(
    input_path: str, require_duration: bool = False
) -> Iterator[Tuple[int, Dict]]:
    """Yield parsed samples from a plain-text or gzip-compressed TSV manifest."""
    opener = gzip.open if input_path.endswith(".gz") else open
    with opener(input_path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            yield line_no, parse_tsv_manifest_line(
                line, line_no, input_path, require_duration=require_duration
            )


def hash_config_for_context(config: BuildConfig) -> str:
    """Create a short hash for build options saved in the resume context."""
    # Output and tmp paths do not change the content of generated shards.
    data = asdict(config).copy()
    data.pop("output_dir", None)
    data.pop("tmp_dir", None)
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def split_plan_matches_request(
    data: Dict,
    *,
    mode: str,
    requested: Optional[int],
    split_bits: int,
) -> bool:
    """Check whether a cached split plan can be reused for this request."""
    if data.get("mode") != mode:
        return False
    if data.get("requested_num_tars") != requested:
        return False
    split_files = data.get("split_files", [])
    if int(data.get("num_tars", -1)) != len(split_files):
        return False
    if any(len(Path(p).name.rsplit("_", 1)[-1]) != split_bits for p in split_files):
        return False
    return bool(split_files) and all(Path(p).is_file() for p in split_files)


def prepare_shuffled_manifest(
    input_path: str, tmp_root: Path, seed: int, reuse: bool = True
) -> Tuple[Path, int]:
    """Parse, count, shuffle, and cache the input manifest for sample splitting."""
    shuffled = tmp_root / "shuffled_manifest.jsonl"
    stats_path = tmp_root / "shuffled_manifest.stats.json"
    stats = None
    if stats_path.is_file():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    if reuse and shuffled.is_file() and stats and "total_samples" in stats:
        print(f"Reusing shuffled manifest: {shuffled}")
        return shuffled, int(stats["total_samples"])

    print("Parsing and shuffling input manifest by sample count...")
    lines: List[str] = []
    for _, item in tqdm(iter_input_manifest(input_path), desc="read", dynamic_ncols=True):
        lines.append(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    random.Random(seed).shuffle(lines)
    atomic_replace_text(shuffled, lines)
    atomic_write_json(stats_path, {"total_samples": len(lines)})
    return shuffled, len(lines)


def split_by_samples(
    input_path: str,
    tmp_root: Path,
    seed: int,
    num_tars: Optional[int],
    samples_per_tar: Optional[int],
    split_bits: int,
) -> SplitPlan:
    """Create or reuse split files balanced by number of samples."""
    plan_path = tmp_root / "sample_split_plan.json"
    shuffled, total = prepare_shuffled_manifest(input_path, tmp_root, seed)
    if num_tars is not None:
        requested = num_tars
        real_num_tars = 1 if num_tars <= 1 else 1 << (num_tars - 1).bit_length()
    elif samples_per_tar is not None:
        requested = math.ceil(total / samples_per_tar) if samples_per_tar > 0 else 1
        real_num_tars = 1 if requested <= 1 else 1 << (requested - 1).bit_length()
    else:
        raise ValueError("Either num_tars or samples_per_tar is required")

    if plan_path.is_file():
        with open(plan_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if split_plan_matches_request(
            existing, mode="samples", requested=requested, split_bits=split_bits
        ):
            print(f"Reusing sample split plan: {plan_path}")
            fields = SplitPlan.__dataclass_fields__.keys()
            return SplitPlan(**{k: v for k, v in existing.items() if k in fields})

    real_samples_per_tar = max(1, math.ceil(total / real_num_tars))
    split_files = [
        str(tmp_root / f"split_{i:0{split_bits}d}") for i in range(real_num_tars)
    ]

    print(
        f"Writing {real_num_tars} sample-balanced splits "
        f"({real_samples_per_tar} samples/shard target)..."
    )
    handles = [open(p, "w", encoding="utf-8") for p in split_files]
    try:
        with open(shuffled, "r", encoding="utf-8") as f:
            for index, line in enumerate(tqdm(f, total=total, dynamic_ncols=True)):
                split_idx = min(index // real_samples_per_tar, real_num_tars - 1)
                handles[split_idx].write(line)
    finally:
        for h in handles:
            h.close()

    plan = SplitPlan(
        mode="samples",
        requested_num_tars=requested,
        num_tars=real_num_tars,
        total_samples=total,
        total_duration=None,
        samples_per_tar=real_samples_per_tar,
        duration_per_tar=None,
        split_files=split_files,
    )
    atomic_write_json(plan_path, asdict(plan))
    return plan


def prepare_duration_buckets(
    input_path: str, tmp_root: Path, reuse: bool = True
) -> Tuple[Path, int, float, List[str]]:
    """Compute durations and cache samples into one-second bucket files."""
    bucket_dir = tmp_root / "duration_buckets"
    stats_path = tmp_root / "duration_buckets.stats.json"
    stats = None
    if stats_path.is_file():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    if reuse and bucket_dir.is_dir() and stats:
        bucket_files = sorted(str(p) for p in bucket_dir.glob("duration_*s"))
        if bucket_files:
            print(f"Reusing duration buckets: {bucket_dir}")
            return (
                bucket_dir,
                int(stats["total_samples"]),
                float(stats["total_duration"]),
                bucket_files,
            )

    print("Parsing input manifest, computing durations, and writing 1s buckets...")
    if bucket_dir.exists():
        shutil.rmtree(bucket_dir)
    bucket_dir.mkdir(parents=True, exist_ok=True)

    handles: Dict[int, object] = {}
    total_samples = 0
    total_duration = 0.0
    try:
        for _, item in tqdm(
            iter_input_manifest(input_path, require_duration=True),
            desc="bucket",
            dynamic_ncols=True,
        ):
            duration = float(item["duration"])
            if duration < 0:
                raise ValueError(f"Negative duration for sample {item['id']}: {duration}")
            bucket = max(0, int(math.floor(duration)))
            if bucket not in handles:
                handles[bucket] = open(
                    bucket_dir / f"duration_{bucket}s", "a", encoding="utf-8"
                )
            handles[bucket].write(
                json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            total_samples += 1
            total_duration += duration
    finally:
        for h in handles.values():
            h.close()

    bucket_files = sorted(str(p) for p in bucket_dir.glob("duration_*s"))
    atomic_write_json(
        stats_path,
        {
            "total_samples": total_samples,
            "total_duration": total_duration,
            "bucket_files": bucket_files,
        },
    )
    return bucket_dir, total_samples, total_duration, bucket_files


def split_by_duration(
    input_path: str,
    tmp_root: Path,
    seed: int,
    num_tars: Optional[int],
    duration_per_tar_hours: Optional[float],
    split_bits: int,
) -> SplitPlan:
    """Create or reuse split files balanced by total audio duration."""
    plan_path = tmp_root / "duration_split_plan.json"
    _, total_samples, total_duration, bucket_files = prepare_duration_buckets(
        input_path, tmp_root
    )
    if num_tars is not None:
        requested = num_tars
        real_num_tars = 1 if num_tars <= 1 else 1 << (num_tars - 1).bit_length()
    elif duration_per_tar_hours is not None:
        target_seconds = duration_per_tar_hours * 3600.0
        requested = math.ceil(total_duration / target_seconds) if target_seconds > 0 else 1
        real_num_tars = 1 if requested <= 1 else 1 << (requested - 1).bit_length()
    else:
        raise ValueError("Either num_tars or duration_per_tar is required")

    if plan_path.is_file():
        with open(plan_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if split_plan_matches_request(
            existing, mode="duration", requested=requested, split_bits=split_bits
        ):
            print(f"Reusing duration split plan: {plan_path}")
            fields = SplitPlan.__dataclass_fields__.keys()
            return SplitPlan(**{k: v for k, v in existing.items() if k in fields})

    real_target_duration = total_duration / max(real_num_tars, 1)
    split_paths = [tmp_root / f"split_{i:0{split_bits}d}" for i in range(real_num_tars)]

    print(
        f"Writing {real_num_tars} duration-balanced splits "
        f"({real_target_duration / 3600.0:.3f} hours/shard target)..."
    )
    rng = random.Random(seed)
    buckets: Dict[str, List[str]] = {}
    for path in bucket_files:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        rng.shuffle(lines)
        if lines:
            buckets[path] = lines

    rng = random.Random(seed + 7919)
    split_lines: List[List[str]] = [[] for _ in range(real_num_tars)]
    split_durations = [0.0 for _ in range(real_num_tars)]

    nonempty_keys = [k for k, v in buckets.items() if v]
    pbar = tqdm(total=total_samples, desc="split", dynamic_ncols=True)
    while nonempty_keys:
        split_idx = min(range(real_num_tars), key=lambda i: split_durations[i])
        key = rng.choice(nonempty_keys)
        line = buckets[key].pop()
        if not buckets[key]:
            nonempty_keys.remove(key)
        item = json.loads(line)
        split_lines[split_idx].append(line)
        split_durations[split_idx] += float(item["duration"])
        pbar.update(1)
    pbar.close()

    split_files: List[str] = []
    for path, lines in zip(split_paths, split_lines):
        atomic_replace_text(path, lines)
        split_files.append(str(path))

    plan = SplitPlan(
        mode="duration",
        requested_num_tars=requested,
        num_tars=real_num_tars,
        total_samples=total_samples,
        total_duration=total_duration,
        samples_per_tar=None,
        duration_per_tar=real_target_duration,
        split_files=split_files,
    )
    payload = asdict(plan)
    payload["split_durations"] = split_durations
    atomic_write_json(plan_path, payload)
    return plan


def create_split_plan(args, tmp_root: Path) -> SplitPlan:
    """Validate split arguments and build the requested split plan."""
    specified = [
        args.num_tars is not None,
        args.samples_per_tar is not None,
        args.duration_per_tar is not None,
    ]
    if sum(specified) != 1:
        raise ValueError(
            "Specify exactly one of --num-tars, --samples-per-tar, "
            "or --duration-per-tar"
        )

    if args.num_tars is not None:
        if args.num_tars <= 0:
            raise ValueError("--num-tars must be positive")
        if args.balance_by == "samples":
            return split_by_samples(
                args.input,
                tmp_root,
                args.seed,
                num_tars=args.num_tars,
                samples_per_tar=None,
                split_bits=args.split_bits,
            )
        return split_by_duration(
            args.input,
            tmp_root,
            args.seed,
            num_tars=args.num_tars,
            duration_per_tar_hours=None,
            split_bits=args.split_bits,
        )

    if args.samples_per_tar is not None:
        if args.samples_per_tar <= 0:
            raise ValueError("--samples-per-tar must be positive")
        return split_by_samples(
            args.input,
            tmp_root,
            args.seed,
            num_tars=None,
            samples_per_tar=args.samples_per_tar,
            split_bits=args.split_bits,
        )

    if args.duration_per_tar is not None:
        if args.duration_per_tar <= 0:
            raise ValueError("--duration-per-tar must be positive")
        return split_by_duration(
            args.input,
            tmp_root,
            args.seed,
            num_tars=None,
            duration_per_tar_hours=args.duration_per_tar,
            split_bits=args.split_bits,
        )

    raise AssertionError("unreachable")


def load_audio_segment(item: Dict, sample_rate: Optional[int]) -> Tuple[torch.Tensor, int]:
    """Load a full audio file or segment and optionally resample it."""
    audio_path = item["audio_path"]
    if item.get("slice_audio", False):
        if "start" not in item or "duration" not in item:
            raise ValueError("start and duration are required for sliced audio")
        info = torchaudio.info(audio_path)
        sr = int(info.sample_rate)
        start_frame = int(float(item["start"]) * sr)
        num_frames = int(float(item["duration"]) * sr)
        try:
            segment, sr = torchaudio.load(
                audio_path,
                frame_offset=start_frame,
                num_frames=num_frames,
                channels_first=True,
            )
        except TypeError:
            # Older torchaudio versions do not support frame_offset/num_frames
            # for every backend. Fall back to loading and slicing in memory.
            segment, sr = torchaudio.load(audio_path, channels_first=True)
            segment = segment[:, start_frame : start_frame + num_frames]
    else:
        segment, sr = torchaudio.load(audio_path, channels_first=True)

    if sample_rate is not None and sr != sample_rate:
        segment = torchaudio.functional.resample(
            segment, orig_freq=sr, new_freq=sample_rate
        )
        sr = sample_rate

    if segment.ndim == 1:
        segment = segment.unsqueeze(0)
    segment = segment[:1, :]
    return segment, int(sr)


def encode_audio(segment: torch.Tensor, sample_rate: int, fmt: str, compression_level: float) -> bytes:
    """Encode a single-channel audio tensor as FLAC, WAV, or MP3 bytes."""
    segment_data = segment.transpose(0, 1).numpy()
    buf = io.BytesIO()
    if fmt == "MP3":
        sf.write(
            buf,
            segment_data,
            sample_rate,
            format="MP3",
            bitrate_mode="CONSTANT",
            compression_level=compression_level,
        )
    else:
        sf.write(buf, segment_data, sample_rate, format=fmt)
    data = buf.getvalue()
    buf.close()
    return data


def write_one_tar(task: Dict) -> Dict:
    """Write one split file to one tar shard and one JSONL manifest."""
    split_file = Path(task["split_file"])
    audio_tar_path = Path(task["audio_tar_path"])
    manifest_path = Path(task["manifest_path"])
    done_path = Path(task["done_path"])
    fmt = task["format"]
    compression_level = float(task["compression_level"])
    sample_rate = task["sample_rate"]

    if done_path.exists() and audio_tar_path.is_file() and manifest_path.is_file():
        return {
            "split_file": str(split_file),
            "audio_tar": str(audio_tar_path),
            "manifest": str(manifest_path),
            "samples": None,
            "duration": None,
            "skipped": True,
            "errors": 0,
        }

    audio_tar_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_tar = audio_tar_path.with_suffix(audio_tar_path.suffix + ".tmp")
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

    samples = 0
    total_duration = 0.0
    errors = 0
    seen_keys = set()
    with tarfile.open(tmp_tar, "w", encoding="utf-8") as audio_tar, open(
        tmp_manifest, "w", encoding="utf-8"
    ) as txt_jsonl, open(split_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            seg_id = item["id"]
            filename = f"{seg_id}.{fmt.lower()}"
            if filename in seen_keys:
                print(
                    f"Warning: duplicate key '{filename}' in {split_file}:{line_no}, "
                    f"skipping.",
                    flush=True,
                )
                continue
            seen_keys.add(filename)

            try:
                segment, sr = load_audio_segment(item, sample_rate=sample_rate)
                duration = float(segment.shape[1]) / float(sr)
                audio_bytes = encode_audio(segment, sr, fmt, compression_level)
            except Exception as e:
                errors += 1
                print(
                    f"Failed to process {item.get('audio_path')} "
                    f"({split_file}:{line_no}) with error: {e}",
                    flush=True,
                )
                continue

            info = tarfile.TarInfo(name=filename)
            info.size = len(audio_bytes)
            audio_tar.addfile(info, io.BytesIO(audio_bytes))

            txt_jsonl.write(
                json.dumps(
                    {
                        "audio_filepath": filename,
                        "text": norm_text(item["text"]),
                        "duration": duration,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            samples += 1
            total_duration += duration

    os.replace(tmp_tar, audio_tar_path)
    os.replace(tmp_manifest, manifest_path)
    done_path.touch()
    return {
        "split_file": str(split_file),
        "audio_tar": str(audio_tar_path),
        "manifest": str(manifest_path),
        "samples": samples,
        "duration": total_duration,
        "skipped": False,
        "errors": errors,
    }


def write_tars_parallel(args, plan: SplitPlan) -> List[Dict]:
    """Generate tar shards, optionally using multiple worker processes.

    When ``--split-start`` / ``--split-end`` are set, only the corresponding
    subset of shard indices is processed, enabling distributed builds across
    multiple machines.
    """
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audios"
    manifest_dir = output_dir / "manifests"
    split_start = getattr(args, "split_start", None) or 0
    split_end = getattr(args, "split_end", None) or len(plan.split_files)
    tasks = []
    for index, split_file in enumerate(plan.split_files):
        if index < split_start or index >= split_end:
            continue
        prefix = f"{args.tar_prefix}.{index:0{args.split_bits}d}"
        tasks.append(
            {
                "split_file": split_file,
                "audio_tar_path": str(audio_dir / f"{prefix}.tar"),
                "manifest_path": str(manifest_dir / f"{prefix}.jsonl"),
                "done_path": str(audio_dir / f".{prefix}.done"),
                "format": args.format,
                "compression_level": args.compression_level,
                "sample_rate": args.sample_rate,
            }
        )

    workers = max(1, int(args.workers))
    print(f"Generating {len(tasks)} tar shards with {workers} worker(s)...")
    results: List[Dict] = []
    if workers == 1:
        for task in tqdm(tasks, desc="tar", dynamic_ncols=True):
            results.append(write_one_tar(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_to_task = {ex.submit(write_one_tar, task): task for task in tasks}
            for fut in tqdm(
                as_completed(future_to_task),
                total=len(future_to_task),
                desc="tar",
                dynamic_ncols=True,
            ):
                results.append(fut.result())
    return sorted(results, key=lambda r: r["audio_tar"])


def write_dataset_list(output_dir: str, results: Sequence[Dict], name: str = "data.lst") -> str:
    """Write the final list file used by atdataset dataloaders."""
    lst_path = Path(output_dir) / name
    lines = []
    for r in results:
        duration = r.get("duration")
        samples = r.get("samples")
        if duration is None or samples is None:
            duration = 0.0
            samples = 0
            try:
                with open(r["manifest"], "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        duration += float(item.get("duration", 0.0))
                        samples += 1
            except FileNotFoundError:
                duration = 0.0
                samples = 0
        lines.append(
            f"{os.path.abspath(r['audio_tar'])} "
            f"{os.path.abspath(r['manifest'])} {duration / 3600.0:.3f} {samples}\n"
        )
    atomic_replace_text(lst_path, lines)
    return str(lst_path)


def add_build_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register command-line arguments for ``atdataset build``."""
    parser.add_argument("--input", required=True, help="Input TSV manifest file.")
    parser.add_argument(
        "--output-dir",
        default="data/tars",
        help="Output directory. Writes audios/*.tar and manifests/*.jsonl.",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help=(
            "Directory name under $HOME/.atdataset for intermediate files. "
            "Default: input manifest path with '/' replaced by '__'."
        ),
    )

    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument(
        "--num-tars",
        type=int,
        default=None,
        help=(
            "Requested tar shard count. The real count is the smallest power "
            "of two greater than or equal to this value."
        ),
    )
    split_group.add_argument(
        "--samples-per-tar",
        type=int,
        default=None,
        help=(
            "Requested samples per tar. The real tar count is recomputed as a "
            "power of two, then samples are evenly split."
        ),
    )
    split_group.add_argument(
        "--duration-per-tar",
        type=float,
        default=None,
        help=(
            "Requested hours per tar. The real tar count is recomputed as a "
            "power of two, then duration is evenly balanced."
        ),
    )
    parser.add_argument(
        "--balance-by",
        choices=["samples", "duration"],
        default="duration",
        help="When --num-tars is used, balance shards by sample count or duration.",
    )

    parser.add_argument(
        "--format",
        default="FLAC",
        choices=["FLAC", "MP3", "WAV"],
        help="Audio format to encode into tar shards.",
    )
    parser.add_argument(
        "--compression-level",
        type=float,
        default=0.65,
        help="MP3 compression level between 0 and 1. Only used for --format MP3.",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=None, help="Optional target sample rate."
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2)
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle/random seed.")
    parser.add_argument(
        "--split-bits", type=int, default=8, help="Zero padding width for shard ids."
    )
    parser.add_argument(
        "--tar-prefix",
        default=None,
        help="Output tar/jsonl prefix. Default: input basename without suffix.",
    )
    parser.add_argument(
        "--no-lst",
        type=str2bool,
        default=False,
        help="Do not write output-dir/data.lst.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        default=False,
        help=(
            "Only create the split plan and intermediate files without building "
            "tar shards. Useful for distributed builds: run --plan-only first, "
            "then run with --split-start/--split-end on each machine."
        ),
    )
    parser.add_argument(
        "--split-start",
        type=int,
        default=None,
        help="Start index of tar shards to build (inclusive).",
    )
    parser.add_argument(
        "--split-end",
        type=int,
        default=None,
        help="End index of tar shards to build (exclusive).",
    )
    return parser


def build(args) -> Dict:
    """Run the complete build workflow and return a summary dictionary."""
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input manifest does not exist: {args.input}")
    if args.tar_prefix is None:
        # Handles xxx.tsv and xxx.tsv.gz reasonably.
        name = input_path.name
        args.tar_prefix = Path(name[:-3]).stem if name.endswith(".gz") else input_path.stem

    args.input = str(input_path.resolve())

    tmp_name = args.tmp_dir if args.tmp_dir else args.input.replace("/", "__")
    tmp_root = Path.home() / ".atdataset" / tmp_name
    tmp_root.mkdir(parents=True, exist_ok=True)

    config = BuildConfig(
        input=args.input,
        output_dir=args.output_dir,
        tmp_dir=str(tmp_root),
        num_tars=args.num_tars,
        samples_per_tar=args.samples_per_tar,
        duration_per_tar=args.duration_per_tar,
        balance_by=args.balance_by,
        format=args.format,
        compression_level=args.compression_level,
        sample_rate=args.sample_rate,
        seed=args.seed,
        split_bits=args.split_bits,
        tar_prefix=args.tar_prefix,
    )
    context_path = tmp_root / "context.json"
    context = {}
    if context_path.is_file():
        with open(context_path, "r", encoding="utf-8") as f:
            context = json.load(f)
    config_hash = hash_config_for_context(config)
    if context and context.get("config_hash") != config_hash:
        print(
            f"Warning: existing context in {tmp_root} was created with a different "
            "configuration. Existing intermediate files will still be reused as "
            "requested; use a new --tmp-dir to force a clean rebuild."
        )
    context.update({"config": asdict(config), "config_hash": config_hash})
    atomic_write_json(context_path, context)

    print(f"Temporary directory: {tmp_root}")
    plan = create_split_plan(args, tmp_root)
    context["split_plan"] = asdict(plan)
    atomic_write_json(context_path, context)

    if args.plan_only:
        print(f"Plan created: {plan.num_tars} shards, {plan.total_samples} samples.")
        print(f"Split files saved under: {tmp_root}")
        return {
            "tmp_dir": str(tmp_root),
            "output_dir": args.output_dir,
            "num_tars": plan.num_tars,
            "total_input_samples": plan.total_samples,
            "plan_only": True,
        }

    results = write_tars_parallel(args, plan)
    lst_path = None if args.no_lst else write_dataset_list(args.output_dir, results)

    written_results = [r for r in results if not r.get("skipped")]
    skipped_results = [r for r in results if r.get("skipped")]
    total_samples = sum(r.get("samples") or 0 for r in written_results)
    total_errors = sum(int(r.get("errors") or 0) for r in results)
    total_duration = sum(float(r.get("duration") or 0.0) for r in written_results)
    summary = {
        "tmp_dir": str(tmp_root),
        "output_dir": args.output_dir,
        "num_tars": plan.num_tars,
        "total_input_samples": plan.total_samples,
        "newly_written_samples": total_samples,
        "newly_written_duration_hours": total_duration / 3600.0,
        "errors": total_errors,
        "dataset_list": lst_path,
        "newly_written_tars": len(written_results),
        "skipped_tars": len(skipped_results),
        "results": results,
    }
    atomic_write_json(tmp_root / "build_summary.json", summary)
    context["summary"] = summary
    atomic_write_json(context_path, context)

    print("Build complete.")
    print(f"  tar shards: {Path(args.output_dir) / 'audios'}")
    print(f"  manifests : {Path(args.output_dir) / 'manifests'}")
    if lst_path:
        print(f"  list file : {lst_path}")
    if total_errors:
        print(f"  errors    : {total_errors} samples failed and were skipped")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    """Parse top-level ``atdataset`` arguments or direct build arguments."""
    argv = sys.argv[1:] if argv is None else list(argv)

    # When invoked with a subcommand or no arguments (--help), show the
    # top-level help listing both subcommands.
    if not argv or argv[0] in ("build", "gen_lst", "-h", "--help"):
        from cli.gen_lst import add_gen_lst_args

        parser = argparse.ArgumentParser(
            prog="atdataset",
            description="ATdataset command line tools for building and managing WebDataset shards.",
        )
        subparsers = parser.add_subparsers(dest="command", required=True)
        add_build_args(subparsers.add_parser(
            "build",
            help="Build WebDataset tar shards from a TSV manifest.",
        ))
        add_gen_lst_args(subparsers.add_parser(
            "gen_lst",
            help="Generate a tar/manifest list file from existing shards.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        ))
        return parser.parse_args(argv)

    # Direct invocation compatibility: python build_tars.py --input ...
    parser = argparse.ArgumentParser(prog="atdataset build")
    add_build_args(parser)
    args = parser.parse_args(argv)
    args.command = "build"
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for the installed ``atdataset`` console script."""
    args = parse_args(argv)
    command = getattr(args, "command", "build")
    if command == "build":
        build(args)
    elif command == "gen_lst":
        from cli.gen_lst import generate_lst

        generate_lst(args)
    else:
        raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()

# ATdataset
Audio text dataset for pytorch training based on webdataset.

## Install

```bash
pip install atdataset
```

## Usage

see examples/example.py for more details.

## Command line tools

After installation, the package provides an `atdataset` command with two
subcommands:

- `atdataset build`: build WebDataset tar shards and per-shard JSONL manifests
  from a TSV input manifest.
- `atdataset gen_lst`: generate the list file consumed by `ATDataset` /
  `ATDataloader` from existing tar shards and JSONL manifests.

### Generate a tar list with `atdataset gen_lst`

`gen_lst` scans tar shards, finds the matching transcript manifest for each tar,
and writes one list entry per shard.

```bash
atdataset gen_lst \
  --audio-pattern 'data/tars/audios/*.tar' \
  --txt-dir data/tars/manifests \
  --output data/tars/data.lst
```

Output format:

```text
/abs/path/to/shard.tar /abs/path/to/shard.jsonl duration_hours num_samples
```

Column meanings:

1. absolute path to the audio tar shard;
2. absolute path to the matching transcript manifest (`.json` or `.jsonl`);
3. total duration of the shard in hours, computed from the manifest `duration`
   fields;
4. number of valid JSON samples in the manifest.

Matching rules:

1. `--audio-pattern` is expanded as a Python glob. Quote the pattern in the
   shell, for example `'data/tars/audios/*.tar'`.
2. The shard key is the tar basename without `.tar` or `.tar.gz`.
   For example, `data/tars/audios/train.000000.tar` maps to key
   `train.000000`.
3. The transcript manifest is searched under `--txt-dir` as `<key>.json` first,
   then `<key>.jsonl`.
4. If audio tar keys and manifest keys use different prefixes, replace the
   prefix with `--audio-prefix` and `--txt-prefix`:

```bash
atdataset gen_lst \
  --audio-pattern 'data/tars/audios/audio.*.tar' \
  --txt-dir data/tars/manifests \
  --output data/tars/data.lst \
  --audio-prefix audio \
  --txt-prefix text
```

By default, missing manifests or prefix mismatches are reported as warnings and
skipped. Use `--strict` to fail immediately:

```bash
atdataset gen_lst \
  --audio-pattern 'data/tars/audios/*.tar' \
  --txt-dir data/tars/manifests \
  --output data/tars/data.lst \
  --strict
```

Useful options:

- `--no-sort`: keep the raw glob order instead of sorting tar paths.
- `--audio-prefix` / `--txt-prefix`: map tar shard names to differently named
  transcript manifests.
- `--strict`: raise an error instead of warning/skipping when a match is missing.


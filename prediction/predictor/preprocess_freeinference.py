"""Materialize verified FreeInference graph, feature and label shards."""

import argparse
import gzip
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from .freeinference_graph import make_examples, reconstruct
from .inspect_freeinference import BLOCKS, ROOT, WEEKS

SPLITS = (
    ("train", "2026-08-23", "2026-08-30"),
    ("validation", "2026-08-30", "2026-09-02"),
    ("test", "2026-09-02", "2026-09-06"),
)


def midnight(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()) * 1000


def split_for(start, end, earliest=None):
    earliest = start if earliest is None else earliest
    for name, lower, upper in SPLITS:
        lower_ms, upper_ms = midnight(lower), midnight(upper)
        if lower_ms <= start < upper_ms:
            return name, lower_ms <= earliest and end < upper_ms
    raise ValueError("Root admission outside selected cohort")


def compact_json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def write_record(handle, record):
    handle.write(compact_json(record) + "\n")


def process_shard(raw_root, output, updates, job):
    week, filename, expected = job
    day = Path(filename).stem
    path = Path("release") / week / f"{filename}.zst"
    summary = {split: Counter() for split, _, _ in SPLITS}
    quality = Counter()
    root_ids = []
    example = None
    digest = hashlib.sha256()
    size = lines = 0
    with ExitStack() as stack:
        process = stack.enter_context(
            subprocess.Popen(["zstd", "-dc", str(raw_root / path)], stdout=subprocess.PIPE)
        )
        writers = {}
        for lines, line in enumerate(process.stdout, 1):
            size += len(line)
            digest.update(line)
            try:
                session = json.loads(BLOCKS.sub(b'"block_ids":[]', line))
                if session["week"] != week or session["day"] != day:
                    raise ValueError("Session week or day disagrees with shard")
                graph = reconstruct(session)
                origin = graph["nodes"]["root/r0"]["start_ms"]
                if not midnight(day) <= origin < midnight(day) + 86_400_000:
                    raise ValueError("Session day disagrees with its UTC admission")
                timed = [node for node in graph["nodes"].values() if node["start_ms"] is not None]
                first = min(session["start_ms"], min(node["start_ms"] for node in timed))
                last = max(session["end_ms"], max(node["end_ms"] for node in timed))
                split, retained = split_for(origin, last, first)
                root_ids.append(session["session_id"])
                counts = summary[split]
                counts["source_roots"] += 1
                quality.update(graph["quality"])
                if not retained:
                    counts["purged_boundary_roots"] += 1
                    continue

                if split not in writers:
                    directory = output / split
                    directory.mkdir(exist_ok=True)
                    writers[split] = {
                        table: stack.enter_context(gzip.open(
                            directory / f"{week}__{day}.{table}.jsonl.gz",
                            "wt", encoding="utf-8", compresslevel=1,
                        ))
                        for table in ("graphs", "snapshots", "targets")
                    }
                tables = writers[split]
                key = f'{week}:{session["session_id"]}'
                counts["graphs"] += 1
                counts.update(f'nodes_{node["kind"]}' for node in graph["nodes"].values())
                counts.update(f'edges_{edge["provenance"]}' for edge in graph["edges"])
                counts.update(f'edges_kind_{edge["kind"]}' for edge in graph["edges"])
                counts["tool_phase_intervals"] += len(graph["intervals"])
                counts["unknown_sibling_order_batches"] += sum(
                    len(node["tool_ids"]) > 1 for node in graph["nodes"].values()
                    if node["kind"] == "llm"
                )
                write_record(tables["graphs"], dict(
                    key=key, start_day=day, source_line=lines, source_shard=str(path),
                    root_start_ms=session["start_ms"], root_end_ms=session["end_ms"],
                    nodes=list(graph["nodes"].values()), edges=graph["edges"],
                    tool_phase_intervals=graph["intervals"],
                ))
                for snapshot, labels in make_examples(graph, updates):
                    if snapshot["cutoff_ms"] < origin or labels["observed_endpoint_ms"] > last:
                        raise ValueError("Snapshot or target outside validated session window")
                    if any(value is not None and value > snapshot["cutoff_ms"]
                           for value in snapshot["feature_available_ms"].values()):
                        raise ValueError("Feature availability exceeds cutoff")
                    if snapshot["ordinal"] == 0:
                        counts["admission_examples"] += 1
                    else:
                        counts["update_examples"] += 1
                    write_record(tables["snapshots"], dict(key=key, **snapshot))
                    write_record(tables["targets"], dict(key=key, **labels))

                    if split == "validation" and day == "2026-09-01" and lines == 141 and snapshot["ordinal"] == 0:
                        example = sanitize_example(graph, snapshot, labels, origin, path, lines)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{lines}: {error}") from error
        if process.wait():
            raise RuntimeError(f"zstd failed for {path}")
    if (size, lines, digest.hexdigest()) != (expected["bytes"], expected["lines"], expected["sha256"]):
        raise ValueError(f"Manifest mismatch for {path}")
    return path, summary, quality, root_ids, example


def sanitize_example(graph, snapshot, labels, origin, path, line):
    nodes = []
    for node in graph["nodes"].values():
        item = {key: node[key] for key in ("id", "kind")}
        for key in ("start_ms", "end_ms"):
            value = node[key]
            item[f"relative_{key}"] = None if value is None else value - origin
        if node["kind"] == "tool":
            item["category"] = node["category"]
        nodes.append(item)
    clean_labels = dict(labels)
    clean_labels["observed_endpoint_ms"] -= origin
    return dict(source_shard=str(path), source_line=line, timestamps_relative_to_admission=True,
                nodes=nodes, edges=graph["edges"],
                features=snapshot["features"], targets=clean_labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="preprocessed",
                        help="New output directory name under data/freeinference")
    parser.add_argument("--updates", action="store_true",
                        help="Also emit snapshots at qualifying root response completions")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or Path(args.output).name != args.output or args.output in {"", ".", ".."}:
        parser.error("Use a positive worker count and a simple output directory name")
    destination = ROOT / args.output
    if destination.exists() or destination.is_symlink():
        parser.error(f"Output already exists: {destination}. Use a new --output name.")

    jobs = []
    for week in WEEKS:
        manifest = json.loads((ROOT / "release" / week / "manifest.json").read_text())
        if manifest["week"] != week or manifest["format_version"] != 1:
            raise ValueError(f"Unsupported manifest for {week}")
        jobs.extend((week, filename, expected) for filename, expected in sorted(manifest["files"].items()))

    totals = {split: Counter() for split, _, _ in SPLITS}
    quality = Counter()
    seen = set()
    with tempfile.TemporaryDirectory(prefix=".preprocess-", dir=ROOT) as temp:
        staging = Path(temp) / "output"
        staging.mkdir()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for path, by_split, checks, ids, example in executor.map(
                lambda job: process_shard(ROOT, staging, args.updates, job), jobs,
            ):
                if len(ids) != len(set(ids)) or seen.intersection(ids):
                    raise ValueError(f"Repeated root session ID in {path}")
                seen.update(ids)
                for split, counts in by_split.items():
                    totals[split].update(counts)
                quality.update(checks)
                if example is not None:
                    (staging / "example.json").write_text(json.dumps(example, indent=2) + "\n")
                print(f"Verified {path}", flush=True)
        if quality["tool_latency_formula_mismatch"]:
            raise ValueError("Published tool latency metadata disagrees with exporter formula")
        report = dict(
            schema_version=1, verified_source_shards=len(jobs), include_updates=args.updates,
            source_revision="bb26cf72c9c20f2b05f765d88548156a74d3c253",
            splits=[dict(name=name, lower_utc=lower, upper_utc_exclusive=upper,
                         counts=totals[name]) for name, lower, upper in SPLITS],
            quality_all_source_roots=quality,
        )
        (staging / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(destination)
        print(f"Published {destination}", flush=True)


if __name__ == "__main__":
    main()

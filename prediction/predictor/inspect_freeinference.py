"""Verify the selected FreeInference release and summarize its trace structure."""

import hashlib
import json
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("data/freeinference")
WEEKS = ("2026-08-23_2026-08-29", "2026-08-30_2026-09-05")
# The v1 field is an array of numeric prefix IDs. Remove its payload after
# hashing to avoid allocating billions of integers irrelevant to this audit.
BLOCKS = re.compile(rb'"block_ids"\s*:\s*\[[0-9,\s]*\]')


def describe(session, origin):
    requests = []
    for request in session["requests"]:
        tools = []
        for tool in request["tool_calls"]:
            item = {key: tool.get(key) for key in
                    ("call_index", "category", "latency_ms", "outcome")}
            if tool.get("subagent"):
                item["child"] = describe(tool["subagent"], origin)
            tools.append(item)
        requests.append({
            "step": request["step"], "trigger": request["step_trigger"],
            "link": request["link_reason"], "tools": tools,
            "start": request["start_ms"] - origin,
            "end": request["end_ms"] - origin,
        })
    return requests


def inspect(job):
    week, name, expected = job
    path = ROOT / "release" / week / f"{name}.zst"
    counts, fields, categories, links, dates = (Counter() for _ in range(5))
    digest = hashlib.sha256()
    examples, identifiers = [], []
    with subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE) as process:
        for line_number, line in enumerate(process.stdout, 1):
            digest.update(line)
            counts["bytes"] += len(line)
            counts["lines"] += 1
            try:
                session = json.loads(BLOCKS.sub(b'"block_ids":[]', line))
            except ValueError:
                counts["invalid_json"] += 1
                continue
            day = session["day"]
            dates[day] += 1
            utc_day = datetime.fromtimestamp(session["start_ms"] / 1000, timezone.utc).date().isoformat()
            counts["day_vs_start_utc"] += day != utc_day
            if not "2026-08-23" <= day <= "2026-09-05":
                counts["outside_window"] += 1
                continue
            identifiers.append(session["session_id"])
            counts["root_sessions"] += 1
            stack = [(session, 0)]
            has_child = False
            while stack:
                current, depth = stack.pop()
                counts["nested_sessions"] += depth > 0
                counts["max_depth"] = max(counts["max_depth"], depth)
                fields.update("session." + key for key in current)
                requests = current["requests"]
                counts["empty_requests"] += not requests
                counts["negative_session_span"] += current["end_ms"] < current["start_ms"]
                previous = None
                for index, request in enumerate(requests):
                    fields.update("request." + key for key in request)
                    counts["llm_requests"] += 1
                    counts["nested_llm_requests"] += depth > 0
                    counts["invalid_step"] += request["step"] != index
                    start, end = request["start_ms"], request["end_ms"]
                    span = end - start
                    counts["negative_request_span"] += span < 0
                    ttft = request.get("ttft_ms")
                    counts["missing_ttft"] += ttft is None
                    counts["invalid_ttft"] += ttft is not None and not 0 <= ttft <= span
                    for key in ("prompt", "completion", "cached"):
                        token = (request.get("provider_tokens") or {}).get(key)
                        counts["missing_" + key] += token is None
                        counts["invalid_" + key] += token is not None and (type(token) is not int or token < 0)
                    if previous:
                        counts["out_of_order_request"] += start < previous["start_ms"]
                        counts["negative_request_gap"] += start < previous["end_ms"]
                        links[f'{request.get("step_trigger")} / {request.get("link_reason")}'] += 1
                    for tool in request["tool_calls"]:
                        fields.update("tool." + key for key in tool)
                        counts["tool_calls"] += 1
                        counts["nested_tool_calls"] += depth > 0
                        categories[tool.get("category") or "<missing>"] += 1
                        latency = tool.get("latency_ms")
                        counts["missing_tool_latency"] += latency is None
                        counts["negative_tool_latency"] += latency is not None and latency < 0
                        counts["tool_latency_with_siblings"] += latency is not None and len(request["tool_calls"]) > 1
                        if tool.get("subagent") is not None:
                            stack.append((tool["subagent"], depth + 1))
                            has_child = True
                    previous = request
            if has_child and len(session["requests"]) <= 6 and len(examples) < 3:
                example = describe(session, session["start_ms"])
                if len(json.dumps(example)) < 5000:
                    examples.append({"file": str(path.relative_to(ROOT)), "line": line_number, "requests": example})
        if process.wait():
            raise RuntimeError(f"zstd failed for {path}")
    counts["compressed_bytes"] = path.stat().st_size
    verified = (digest.hexdigest(), counts["bytes"], counts["lines"]) == (
        expected["sha256"], expected["bytes"], expected["lines"])
    return {
        "file": str(path.relative_to(ROOT)), "verified": verified,
        "counts": dict(counts), "fields": dict(fields), "categories": dict(categories),
        "links": dict(links), "dates": dict(dates), "examples": examples,
        "identifiers": identifiers,
    }


def main():
    jobs = []
    exclusions = {}
    for week in WEEKS:
        manifest = json.loads((ROOT / "release" / week / "manifest.json").read_text())
        exclusions[week] = manifest["excluded_sessions"]
        jobs.extend((week, name, expected) for name, expected in manifest["files"].items())
    totals = Counter()
    identifiers = Counter()
    shards = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for shard in pool.map(inspect, jobs):
            counts = shard["counts"].copy()
            totals["max_depth"] = max(totals["max_depth"], counts.pop("max_depth", 0))
            totals.update(counts)
            identifiers.update(shard.pop("identifiers"))
            shards.append(shard)
            print(shard["file"], shard["verified"], flush=True)
    totals["repeated_root_ids"] = sum(count - 1 for count in identifiers.values())
    output = {"totals": totals, "excluded_by_publisher": exclusions, "shards": shards}
    (ROOT / "inspection.json").write_text(json.dumps(output, indent=2) + "\n")
    if not all(shard["verified"] for shard in shards):
        raise SystemExit("One or more shards failed manifest verification")


if __name__ == "__main__":
    main()

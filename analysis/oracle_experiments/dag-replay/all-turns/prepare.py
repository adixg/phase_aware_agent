"""Build trace-grounded GPU jobs and mark uncertain cross-chain edges."""
import gzip
import io
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "source"))
import orjson
import zstandard
from analysis.replay.extract import BLOCK_TOKENS, _cut_blocks, _lcp
from analysis.replay.run_gate import PROFILES
from analysis.replay.sim import build_sessions

DAYS = ("2026-08-30", "2026-08-31", "2026-09-01")
PROFILE = PROFILES["l40s"]
WEEK = "2026-08-30_2026-09-05"
JOIN_MAX_MS = 2000
SEED = 0
SLOTS = 8
LOAD = 1.1


def compact(day):
    path = DATA / "oracle-sensitivity" / "compact" / f"{day}.json"
    return json.loads(path.read_bytes())["sessions"]


def cutoff(day):
    return datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000


def matched_graphs(day):
    path = DATA / "milestone-preprocessed" / "validation" / f"{WEEK}__{day}.graphs.jsonl.gz"
    with gzip.open(path, "rt") as source:
        return {g["source_line"]: g for g in map(json.loads, source)
                if any(n["kind"] == "subagent" for n in g["nodes"])}


def raw_matches(day, graphs):
    path = DATA / "release" / WEEK / "traces" / f"{day}.jsonl.zst"
    with path.open("rb") as source, io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(source)) as trace:
        for line_no, line in enumerate(trace, 1):
            if line_no in graphs:
                clean, blocks = _cut_blocks(line)
                yield line_no, orjson.loads(clean), iter(blocks)
            if line_no == max(graphs):
                break


def nested_requests(raw, arrays):
    records = {}

    def visit(chain, requests):
        previous = None
        previous_completion = 0
        for index, request in enumerate(requests):
            cur = next(arrays)
            prefix = 0 if previous is None else _lcp(previous, cur) * BLOCK_TOKENS
            completion = request["provider_tokens"]["completion"]
            prompt = len(cur) * BLOCK_TOKENS
            if previous is not None and (request["mutation"] or {}).get("transition") == "append":
                prefix += previous_completion
            node_id = f"{chain}/r{index}"
            cost = (PROFILE.overhead_ms + PROFILE.prefill_ms * (prompt - min(prompt, prefix))
                    + PROFILE.decode_ms * completion) / 1000
            records[node_id] = dict(start=request["start_ms"], end=request["end_ms"],
                                    service=cost, footprint=prompt + completion)
            previous, previous_completion = cur, completion
            for tool_index, tool in enumerate(request["tool_calls"]):
                if tool.get("subagent"):
                    visit(f"{node_id}/t{tool_index}/child", tool["subagent"]["requests"])

    visit("root", raw["requests"])
    assert next(arrays, None) is None
    return records


def inspect_branches(graph, requests):
    nodes = {n["id"]: n for n in graph["nodes"]}
    continuations = {(e["source"], e["target"]) for e in graph["edges"]
                     if e["kind"] == "continuation" and e["provenance"] == "observed"}
    branches = []
    for child in nodes.values():
        if child["kind"] != "subagent":
            continue
        tool = nodes[child["parent_tool"]]
        emitter = nodes[tool["emitter"]]
        jobs = sorted((n for n in nodes.values() if n["kind"] == "llm" and n["chain"] == child["id"]),
                      key=lambda n: n["step"])
        following = f"root/r{emitter['step'] + 1}"
        successor = nodes.get(following)
        assert child["root_turn"] == emitter["root_turn"]
        if successor is not None and successor["root_turn"] != child["root_turn"]:
            successor = None
        launch = jobs[0]["start_ms"] - emitter["end_ms"]
        residual = None if successor is None else successor["start_ms"] - jobs[-1]["end_ms"]
        gaps = [(b["start_ms"] - a["end_ms"]) / 1000 for a, b in zip(jobs, jobs[1:])]
        for job in [emitter, *jobs]:
            raw = requests[job["id"]]
            assert (raw["start"], raw["end"]) == (job["start_ms"], job["end_ms"])
        excluded = None
        if launch < 0:
            excluded = "child launches before emitter gateway completes"
        elif any(gap < 0 or (a["id"], b["id"]) not in continuations
                 for gap, a, b in zip(gaps, jobs, jobs[1:])):
            excluded = "child chain order or release gap is unsupported"
        # Only a single emitting tool, an observed continuation, and a short positive residual.
        join = (excluded is None and successor is not None and len(emitter["tool_ids"]) == 1
                and (emitter["id"], following) in continuations and 0 <= residual <= JOIN_MAX_MS)
        overlap = any(
            n["kind"] == "llm" and n["chain"] == "root" and n["root_turn"] == child["root_turn"]
            and max(j["start_ms"], n["start_ms"]) < min(j["end_ms"], n["end_ms"])
            for j in jobs for n in nodes.values()
        )
        branches.append(dict(turn=child["root_turn"], cross_chain_overlap=overlap,
                             chain=child["id"], emitter=emitter["id"], successor=following if successor else None,
                             launch_ms=launch, residual_ms=residual, emitted_tools=len(emitter["tool_ids"]),
                             join_candidate=join, excluded=excluded, spawn_provenance="publisher-inferred",
                             observed_join="unknown", jobs=[dict(id=n["id"], **requests[n["id"]]) for n in jobs],
                             gaps=gaps))
    eligible = [b for b in branches if b["excluded"] is None]
    for index, left in enumerate(eligible):
        for right in eligible[index+1:]:
            if left["turn"] == right["turn"] and any(
                max(a["start"], b["start"]) < min(a["end"], b["end"])
                for a in left["jobs"] for b in right["jobs"]
            ):
                left["cross_chain_overlap"] = right["cross_chain_overlap"] = True
    return branches


def prepare():
    output = {}
    for day in DAYS:
        graphs = matched_graphs(day)
        original = compact(day)
        kept = [s for s in original if all(r["start"] < cutoff("2026-09-02") and
                r["end"] < cutoff("2026-09-02") and r["prompt"] + r["completion"] <= 1_000_000
                for r in s["requests"])]
        by_sid = {s["sid"]: s for s in kept}
        turns = []
        for line, raw, arrays in raw_matches(day, graphs):
            graph = graphs[line]
            sid = raw["session_id"]
            assert graph["key"] == f"{WEEK}:{sid}"
            assert graph["root_start_ms"] == raw["requests"][0]["start_ms"]
            if sid not in by_sid:
                turns.append(dict(line=line, sid=sid, excluded="outside prior replay cohort", branches=[]))
                continue
            requests = nested_requests(raw, arrays)
            branches = inspect_branches(graph, requests)
            turns.append(dict(line=line, sid=sid, excluded=None, branches=branches))
        assert len(turns) == len(graphs)
        sessions = build_sessions(kept, PROFILE)
        output[day] = dict(candidates=turns, sessions=len(sessions),
                           root_jobs=sum(len(s.reqs) for s in sessions))
        print(day, len(sessions), "background sessions", len(turns), "matched root sessions", flush=True)
    previous = json.loads((ROOT.parent / "inputs.json").read_text())["setup"]
    budgets = previous["budgets"]
    assert (SEED, SLOTS, LOAD) == (previous["seed"], previous["slots"], previous["target_load"])
    branches = [b for day in DAYS for c in output[day]["candidates"] for b in c["branches"]]
    eligible = [b for b in branches if b["excluded"] is None]
    matched_turns = {(c["sid"], b["turn"]) for day in DAYS for c in output[day]["candidates"]
                     for b in c["branches"] if b["excluded"] is None}
    assert (len(branches), len(eligible), len(matched_turns)) == (67, 43, 32)
    assert sum(b["join_candidate"] for b in eligible) == 5
    output["setup"] = dict(seed=SEED, slots=SLOTS, target_load=LOAD, budgets=budgets,
                           join_residual_max_ms=JOIN_MAX_MS,
                           upstream="a3dbeae41b7dac6c06f3fe1863ad1300e348f405")
    (ROOT / "inputs.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    prepare()

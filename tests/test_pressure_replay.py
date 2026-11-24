"""Pressure replay: prompts carry the trace's shared prefixes, and the client replays a workload end to
end against a stub of vLLM's streaming completions API."""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("pressure_replay", Path(__file__).parents[1] / "scripts/pressure_replay.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)


def workload():
    reqs = [{"prompt": 40, "lcp": 0, "completion": 3, "gap_after_s": 0.01, "user_turn": True},
            {"prompt": 60, "lcp": 40, "completion": 2, "gap_after_s": 0.01, "user_turn": False},
            {"prompt": 50, "lcp": 30, "completion": 2, "gap_after_s": None, "user_turn": True}]
    return {"meta": {}, "sessions": [{"sid": "a", "arrival_s": 0.0, "requests": reqs},
                                     {"sid": "b", "arrival_s": 0.05, "requests": reqs[:2]}]}


def test_prompts_share_exactly_the_traced_prefix_and_repeat():
    s = workload()["sessions"][0]
    p = pr.prompts_for(s)
    assert [len(x) for x in p] == [40, 60, 50]
    assert p[1][:40] == p[0] and p[1][40:] != p[0][:20]
    assert p[2][:30] == p[1][:30] and p[2][30] != p[1][30]
    assert pr.prompts_for(s) == p                                   # deterministic
    assert pr.prompts_for(workload()["sessions"][1])[0] != p[0]     # sessions differ


@pytest.fixture
def stub():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": "stub"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(req)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for _ in range(req["max_tokens"]):
                self.wfile.write(b'data: {"choices": [{"text": "x"}]}\n\n')
            usage = {"prompt_tokens": len(req["prompt"]), "completion_tokens": req["max_tokens"],
                     "prompt_tokens_details": {"cached_tokens": 16}}
            self.wfile.write(f'data: {json.dumps({"choices": [], "usage": usage})}\n\ndata: [DONE]\n\n'.encode())

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", seen
    httpd.shutdown()


def test_replay_sends_every_request_with_forced_output_lengths(stub, tmp_path):
    url, seen = stub
    wl = tmp_path / "w.json"
    wl.write_text(json.dumps(workload()))
    args = type("A", (), {"base_url": url, "workload": str(wl), "speedup": 1.0, "out": str(tmp_path / "run"),
                          "limit_sessions": 0})()
    pr.replay(args)
    rows = [json.loads(l) for l in open(tmp_path / "run/requests.jsonl")]
    assert len(rows) == 5 and all(r["error"] is None for r in rows)
    assert all(r["ignore_eos"] and r["min_tokens"] == r["max_tokens"] for r in seen)
    assert sorted(r["completion_tokens"] for r in rows) == [2, 2, 2, 3, 3]
    a = sorted((r for r in rows if r["sid"] == "a"), key=lambda r: r["i"])
    assert all(x["end_s"] <= y["ready_s"] for x, y in zip(a, a[1:]))   # closed loop: one at a time per session
    assert json.load(open(tmp_path / "run/manifest.json"))["model"] == "stub"

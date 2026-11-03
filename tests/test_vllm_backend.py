"""VLLMBackend against a stub OpenAI-compatible server: request shape, and real
usage numbers (not word counts) landing in the trace, streamed or not."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent.graph import LangGraphReActAgent
from agent.llm_backend import REACT_SYSTEM_PROMPT, VLLMBackend, react_system_prompt
from agent.retriever import LocalRetriever
from agent.workloads import Document
from serving.instrumentation.tracer import Tracer

DOCS = [Document("paris", "Paris", "Paris is the capital of France.")]
TURNS = [
    "Thought: I need the capital.\nAction: retrieve[capital of France]",
    "Thought: Found it.\nAction: finish[Paris]",
]


@pytest.fixture
def server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._reply({"data": [{"id": "stub-model"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(req)
            n = len(requests)
            usage = {"prompt_tokens": 100 * n, "completion_tokens": 7 * n,
                     "prompt_tokens_details": {"cached_tokens": 50 * (n - 1)}}
            if not req.get("stream"):
                self._reply({"choices": [{"message": {"content": TURNS[n - 1]}}], "usage": usage})
                return
            # vLLM's shape: a role-only delta, content deltas, then a choices-less usage chunk
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            time.sleep(0.1)   # "queued + prefill" before the first token
            text = TURNS[n - 1]
            chunks = [{"choices": [{"delta": {"role": "assistant"}}]}]
            chunks += [{"choices": [{"delta": {"content": text[i:i + 10]}}]} for i in range(0, len(text), 10)]
            chunks.append({"choices": [], "usage": usage})
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.01)
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", requests
    httpd.shutdown()


def test_served_model(server):
    base_url, _ = server
    assert VLLMBackend.served_model(base_url) == "stub-model"


@pytest.mark.parametrize("stream", [True, False])
def test_trace_uses_server_usage(server, stream):
    base_url, requests = server
    agent = LangGraphReActAgent(VLLMBackend("stub-model", base_url=base_url, stream=stream),
                                LocalRetriever.from_documents(DOCS))
    tracer = Tracer("job")
    result = agent.run("job", "What is the capital of France?", tracer)

    assert result.success and result.answer == "Paris"
    llm_spans = [s for s in tracer.spans if s.phase == "llm_generate"]
    assert [s.meta["prompt_tokens"] for s in llm_spans] == [100, 200]
    assert [s.meta["output_tokens"] for s in llm_spans] == [7, 14]
    assert [s.meta["cached_tokens"] for s in llm_spans] == [0, 50]
    if stream:
        assert all(s.meta["ttft_s"] >= 0.1 and s.meta["decode_s"] > 0 for s in llm_spans)
        # span: time.time(), backend: perf_counter(); allow for the clocks disagreeing slightly
        assert all(s.meta["ttft_s"] + s.meta["decode_s"] <= s.duration_s + 0.01 for s in llm_spans)
    else:
        assert all("ttft_s" not in s.meta for s in llm_spans)

    first, second = requests
    assert first["model"] == "stub-model"
    assert first["messages"][0] == {"role": "system", "content": REACT_SYSTEM_PROMPT}
    assert first["stop"] == ["Observation:"]
    assert first.get("stream", False) is stream
    # every sampling parameter is explicit, so server/model defaults can't leak in
    assert {k: first[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty")} == \
        {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0}
    assert second["messages"][1]["content"].endswith("Observation: [1] Paris. Paris is the capital of France.")


def test_tool_list_and_chat_template_kwargs(server):
    base_url, requests = server
    prompt = react_system_prompt(("retrieve", "finish"))
    assert "calculator" not in prompt and "retrieve[query]" in prompt
    backend = VLLMBackend("stub-model", base_url=base_url, system_prompt=prompt,
                          chat_template_kwargs={"enable_thinking": False})
    backend.generate("Question: q", 0)
    assert requests[0]["messages"][0]["content"] == prompt
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}

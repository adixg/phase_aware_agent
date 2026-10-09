"""VLLMBackend against a stub OpenAI-compatible server: request shape, and real
usage numbers (not word counts) landing in the trace."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent.graph import LangGraphReActAgent
from agent.llm_backend import REACT_SYSTEM_PROMPT, VLLMBackend
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
            self._reply({
                "choices": [{"message": {"content": TURNS[n - 1]}}],
                "usage": {"prompt_tokens": 100 * n, "completion_tokens": 7 * n,
                          "prompt_tokens_details": {"cached_tokens": 50 * (n - 1)}},
            })

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", requests
    httpd.shutdown()


def test_served_model(server):
    base_url, _ = server
    assert VLLMBackend.served_model(base_url) == "stub-model"


def test_trace_uses_server_usage(server):
    base_url, requests = server
    agent = LangGraphReActAgent(VLLMBackend("stub-model", base_url=base_url), LocalRetriever.from_documents(DOCS))
    tracer = Tracer("job")
    result = agent.run("job", "What is the capital of France?", tracer)

    assert result.success and result.answer == "Paris"
    llm_spans = [s for s in tracer.spans if s.phase == "llm_generate"]
    assert [s.meta["prompt_tokens"] for s in llm_spans] == [100, 200]
    assert [s.meta["output_tokens"] for s in llm_spans] == [7, 14]
    assert [s.meta["cached_tokens"] for s in llm_spans] == [0, 50]

    first, second = requests
    assert first["model"] == "stub-model"
    assert first["messages"][0] == {"role": "system", "content": REACT_SYSTEM_PROMPT}
    assert first["stop"] == ["Observation:"]
    assert second["messages"][1]["content"].endswith("Observation: Paris. Paris is the capital of France.")

"""Behavior and trace-shape checks for the LangGraph ReAct agent."""

import pytest

from agent.graph import LangGraphReActAgent
from agent.llm_backend import DEFAULT_SCRIPT, LAST_STEP_NOTE, MockLLM
from agent.retriever import LocalRetriever
from agent.workloads import Document
from serving.instrumentation.tracer import Tracer

DOCS = [
    Document("france", "France", "Paris is the capital of France."),
    Document("paris", "Paris", "Paris has a population of about 2.1 million."),
]
QUESTION = "What is the population of the capital of France, doubled?"

UNKNOWN_TOOL = ["Thought: hm.\nAction: search[x]", "Thought: done.\nAction: finish[ok]"]
NO_ACTION = ["Thought: I have no idea."]
BAD_CALC = ["Thought: calc.\nAction: calculator[import os]", "Thought: done.\nAction: finish[ok]"]
NEVER_FINISHES = ["Thought: again.\nAction: retrieve[capital of France]"] * 3

LLM, RET, TOOL = "llm_generate", "retrieve", "tool_exec"


def _run(script, max_steps=8):
    llm = MockLLM(script, tokens_per_second=1e9, jitter_s=(0.0, 0.0))
    agent = LangGraphReActAgent(llm, LocalRetriever.from_documents(DOCS), max_steps=max_steps)
    tracer = Tracer("job")
    result = agent.run("job", QUESTION, tracer)
    return result, tracer.spans


def test_default_script_trace_shape():
    result, spans = _run(DEFAULT_SCRIPT)
    assert [s.phase for s in spans] == [LLM, RET, LLM, RET, LLM, TOOL, LLM]
    assert [s.step for s in spans] == list(range(7))
    assert result.success
    assert result.answer == "The population of Paris, doubled, is approximately 4.2 million."
    assert sorted(spans[0].meta) == ["output_text", "output_tokens", "prompt_tokens", "step"]
    assert sorted(spans[1].meta) == ["doc_ids", "query", "score", "search_s", "step", "top_doc"]
    assert sorted(spans[5].meta) == ["expr", "result", "step", "tool"]


def test_unknown_tool_records_error_and_continues():
    result, spans = _run(UNKNOWN_TOOL)
    assert [s.phase for s in spans] == [LLM, LLM]
    assert "Observation: error: unknown tool 'search'" in result.transcript
    assert result.success and result.answer == "ok"


def test_unparseable_action_stops_without_success():
    result, spans = _run(NO_ACTION)
    assert [s.phase for s in spans] == [LLM]
    assert not result.success and result.answer is None
    assert result.transcript[-1] == "Observation: could not parse an action; stopping."


def test_bad_calculator_input_is_reported_not_raised():
    result, spans = _run(BAD_CALC)
    assert [s.phase for s in spans] == [LLM, TOOL, LLM]
    assert "result" not in spans[1].meta
    assert any(line.startswith("Observation: error:") for line in result.transcript)
    assert result.success


def test_text_after_first_action_stays_out_of_transcript():
    rambling = ("Thought: look it up.\nAction: retrieve[capital of France]\n\n"
                "The capital is Lyon.\nAction: finish[Lyon]")
    result, spans = _run([rambling, "Thought: done.\nAction: finish[Paris]"])
    assert result.answer == "Paris"
    assert "Thought: look it up.\nAction: retrieve[capital of France]" in result.transcript
    assert not any("Lyon" in line for line in result.transcript)
    assert spans[0].meta["output_text"] == rambling


def test_max_steps_caps_the_loop():
    # NEVER_FINISHES repeats one query: only the first runs, the repeats are blocked
    result, spans = _run(NEVER_FINISHES, max_steps=3)
    assert [s.phase for s in spans] == [LLM, RET, LLM, LLM]
    assert not result.success and result.answer is None


def test_repeated_search_is_blocked_not_run():
    script = ["Thought: a.\nAction: retrieve[Capital of  France?]",
              "Thought: b.\nAction: retrieve[capital of france]",          # same query, normalized
              "Thought: c.\nAction: retrieve[population of Paris]",
              "Thought: d.\nAction: finish[Paris]"]
    result, spans = _run(script)
    assert [s.phase for s in spans] == [LLM, RET, LLM, LLM, RET, LLM]
    assert [s.meta.get("repeat_blocked") for s in spans if s.phase == LLM] == [None, True, None, None]
    assert 'Observation: You already searched for "capital of france". Use different words, or finish with your answer.' in result.transcript
    assert result.success and result.answer == "Paris"


def test_retrieve_returns_numbered_top_k_passages():
    llm = MockLLM(["Thought: look.\nAction: retrieve[Paris]", "Thought: done.\nAction: finish[ok]"],
                  tokens_per_second=1e9, jitter_s=(0.0, 0.0))
    agent = LangGraphReActAgent(llm, LocalRetriever.from_documents(DOCS), top_k=2)
    tracer = Tracer("job")
    result = agent.run("job", QUESTION, tracer)
    observation = result.transcript[2]
    assert observation.startswith("Observation: [1] ") and "\n[2] " in observation
    assert len(tracer.spans[1].meta["doc_ids"]) == 2


def test_last_step_gets_the_finish_note_in_its_prompt_only():
    prompts = []

    class Recording(MockLLM):
        def generate(self, prompt, step):
            prompts.append(prompt)
            return super().generate(prompt, step)

    llm = Recording(NEVER_FINISHES, tokens_per_second=1e9, jitter_s=(0.0, 0.0))
    agent = LangGraphReActAgent(llm, LocalRetriever.from_documents(DOCS), max_steps=3)
    tracer = Tracer("job")
    result = agent.run("job", QUESTION, tracer)
    assert [LAST_STEP_NOTE in p for p in prompts] == [False, False, True]
    assert not any(LAST_STEP_NOTE in line for line in result.transcript)
    assert [s.meta.get("last_step_note") for s in tracer.spans if s.phase == LLM] == [None, None, True]


def test_retrieve_delay_is_inside_the_span_and_recorded_apart():
    llm = MockLLM(DEFAULT_SCRIPT, tokens_per_second=1e9, jitter_s=(0.0, 0.0))
    agent = LangGraphReActAgent(llm, LocalRetriever.from_documents(DOCS), retrieve_delay=lambda step: 0.05 * (step + 1))
    tracer = Tracer("job")
    agent.run("job", QUESTION, tracer)
    rets = [s for s in tracer.spans if s.phase == RET]
    assert [s.meta["injected_delay_s"] for s in rets] == [0.05, 0.1]   # agent steps 0 and 1
    for s in rets:
        assert s.duration_s >= s.meta["injected_delay_s"]
        assert s.meta["search_s"] < s.duration_s

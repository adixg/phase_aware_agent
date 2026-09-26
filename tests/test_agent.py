"""Behavior and trace-shape checks for the LangGraph ReAct agent."""

from pathlib import Path

import pytest

from agent.graph import LangGraphReActAgent
from agent.llm_backend import DEFAULT_SCRIPT, MockLLM
from agent.retriever import LocalRetriever
from serving.instrumentation.tracer import Tracer

CORPUS = Path(__file__).resolve().parents[1] / "agent" / "retriever" / "corpus"
QUESTION = "What is the population of the capital of France, doubled?"

UNKNOWN_TOOL = ["Thought: hm.\nAction: search[x]", "Thought: done.\nAction: finish[ok]"]
NO_ACTION = ["Thought: I have no idea."]
BAD_CALC = ["Thought: calc.\nAction: calculator[import os]", "Thought: done.\nAction: finish[ok]"]
NEVER_FINISHES = ["Thought: again.\nAction: retrieve[capital of France]"] * 3

LLM, RET, TOOL = "llm_generate", "retrieve", "tool_exec"


def _run(script, max_steps=8):
    llm = MockLLM(script, tokens_per_second=1e9, jitter_s=(0.0, 0.0))
    agent = LangGraphReActAgent(llm, LocalRetriever.from_dir(CORPUS), max_steps=max_steps)
    tracer = Tracer("job")
    result = agent.run("job", QUESTION, tracer)
    return result, tracer.spans


def test_default_script_trace_shape():
    result, spans = _run(DEFAULT_SCRIPT)
    assert [s.phase for s in spans] == [LLM, RET, LLM, RET, LLM, TOOL, LLM]
    assert [s.step for s in spans] == list(range(7))
    assert result.success
    assert result.answer == "The population of Paris, doubled, is approximately 4.2 million."
    assert sorted(spans[0].meta) == ["output_tokens", "prompt_tokens", "step"]
    assert sorted(spans[1].meta) == ["query", "score", "step", "top_doc"]
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


def test_max_steps_caps_the_loop():
    result, spans = _run(NEVER_FINISHES, max_steps=3)
    assert [s.phase for s in spans] == [LLM, RET] * 3
    assert not result.success and result.answer is None

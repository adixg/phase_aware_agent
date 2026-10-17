"""LangGraph implementation of the ReAct agent.

Same contract as `agent.agent.ReActAgent`: `run(job_id, question, tracer)`
returns an `AgentResult`, and each loop iteration is one LLM_GENERATE span
followed by at most one RETRIEVE / TOOL_EXEC span. Spans are opened inside
the graph nodes, so the trace schema is unchanged by the migration.
"""

from __future__ import annotations

import re
import time
from typing import Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent.common import _ACTION_RE, AgentResult, _safe_eval
from agent.llm_backend import LAST_STEP_NOTE, Generation, LLMBackend
from agent.retriever import LocalRetriever
from serving.instrumentation.phases import Phase
from serving.instrumentation.tracer import Tracer


class AgentState(TypedDict):
    transcript: list[str]
    step: int
    tool: Optional[str]
    arg: Optional[str]
    answer: Optional[str]
    success: bool
    searched: list[str]      # normalized retrieve queries already run in this job
    repeat: bool             # the latest Action re-ran one of them


def _norm_query(q: str) -> str:
    """Case/punctuation/spacing-insensitive form, so "Paris  population?" repeats "paris population"."""
    return " ".join(re.sub(r"[^\w\s]", " ", q.lower()).split())


class LangGraphReActAgent:
    """Thought -> Action -> Observation loop expressed as a LangGraph state machine.

    generate --(retrieve|calculator|unknown)--> tool node --> generate ...
             --(finish)--> finish --> END
             --(no parseable action)--> parse_error --> END
             --(retrieve repeating an earlier query)--> repeat_search --> generate ...
    The loop also ends once `max_steps` generations have run.
    """

    def __init__(self, llm: LLMBackend, retriever: LocalRetriever, *, max_steps: int = 8, top_k: int = 3,
                 retrieve_delay: Callable[[int], float] | None = None):
        self.llm = llm
        self.retriever = retriever
        self.max_steps = max_steps
        self.top_k = top_k   # passages per retrieve; one bad top hit derails multi-hop questions
        # step -> seconds of emulated backend latency added inside each RETRIEVE span (agent.delay)
        self.retrieve_delay = retrieve_delay

    def run(self, job_id: str, question: str, tracer: Tracer) -> AgentResult:
        graph = self._build(tracer)
        initial: AgentState = {
            "transcript": [f"Question: {question}"],
            "step": 0,
            "tool": None,
            "arg": None,
            "answer": None,
            "success": False,
            "searched": [],
            "repeat": False,
        }
        # ~2 node visits per step, plus slack for the terminal node.
        final = graph.invoke(initial, {"recursion_limit": 2 * self.max_steps + 5})
        return AgentResult(
            job_id=job_id,
            question=question,
            answer=final["answer"],
            transcript=final["transcript"],
            success=final["success"],
        )

    def _build(self, tracer: Tracer):
        g = StateGraph(AgentState)

        def generate(state: AgentState) -> dict:
            step = state["step"]
            prompt = "\n".join(state["transcript"])
            last_step = step == self.max_steps - 1
            if last_step:  # prompt-only: the note never enters the transcript
                prompt += "\n" + LAST_STEP_NOTE
            with tracer.phase(Phase.LLM_GENERATE, step=step) as meta:
                if last_step:
                    meta["last_step_note"] = True
                out = self.llm.generate(prompt, step)
                if isinstance(out, Generation):
                    text = out.text
                    meta["prompt_tokens"] = out.prompt_tokens
                    meta["output_tokens"] = out.output_tokens
                    if out.cached_tokens is not None:
                        meta["cached_tokens"] = out.cached_tokens
                else:  # plain-text backends (MockLLM): word-count estimates
                    text = out
                    meta["prompt_tokens"] = len(prompt.split())
                    meta["output_tokens"] = len(text.split())
                # The model's raw turn (Thought + Action): shows why a step ran
                # long, and is input for next-phase prediction later.
                meta["output_text"] = text
                match = _ACTION_RE.search(text)
                tool = match.group(1).lower() if match else None
                arg = match.group(2).strip() if match else None
                # Small models re-run the same search despite the prompt; the repeat
                # returns the same passages and only pads the job with no-op
                # retrieve/generate cycles. It is not executed (see repeat_search).
                repeat = tool == "retrieve" and _norm_query(arg) in state["searched"]
                if repeat:
                    meta["repeat_blocked"] = True
            # Models often keep going after the Action and invent its result (or a
            # second Action); carrying that forward makes later steps treat the
            # invention as fact. Only the turn up to the first Action enters the
            # transcript; meta.output_text keeps the raw output.
            turn = text[: match.end()] if match else text
            return {"transcript": state["transcript"] + [turn], "tool": tool, "arg": arg, "repeat": repeat}

        def parse_error(state: AgentState) -> dict:
            return {"transcript": state["transcript"] + ["Observation: could not parse an action; stopping."]}

        def finish(state: AgentState) -> dict:
            return {"answer": state["arg"], "success": True}

        def retrieve(state: AgentState) -> dict:
            step, arg = state["step"], state["arg"]
            with tracer.phase(Phase.RETRIEVE, step=step, query=arg) as meta:
                t0 = time.perf_counter()
                hits = self.retriever.retrieve(arg, top_k=self.top_k)
                meta["search_s"] = time.perf_counter() - t0
                if self.retrieve_delay is not None:
                    delay = self.retrieve_delay(step)
                    time.sleep(delay)
                    meta["injected_delay_s"] = delay
                meta["top_doc"] = hits[0].doc_id if hits else None
                meta["score"] = hits[0].score if hits else 0.0
                meta["doc_ids"] = [h.doc_id for h in hits]
            observation = "\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, 1)) if hits else "No results found."
            return {**_observe(state, observation), "searched": state["searched"] + [_norm_query(arg)]}

        def repeat_search(state: AgentState) -> dict:
            # No RETRIEVE span: the generate span that asked carries meta.repeat_blocked.
            return _observe(state, f'You already searched for "{state["arg"]}". '
                                   "Use different words, or finish with your answer.")

        def calculator(state: AgentState) -> dict:
            step, arg = state["step"], state["arg"]
            with tracer.phase(Phase.TOOL_EXEC, step=step, tool="calculator", expr=arg) as meta:
                try:
                    result = _safe_eval(arg)
                    meta["result"] = result
                    observation = str(result)
                except Exception as e:  # scripted/malformed tool input
                    observation = f"error: {e}"
            return _observe(state, observation)

        def unknown_tool(state: AgentState) -> dict:
            return _observe(state, f"error: unknown tool '{state['tool']}'")

        def _observe(state: AgentState, observation: str) -> dict:
            return {
                "transcript": state["transcript"] + [f"Observation: {observation}"],
                "step": state["step"] + 1,
            }

        def route_action(state: AgentState) -> str:
            tool = state["tool"]
            if tool is None:
                return "parse_error"
            if state["repeat"]:
                return "repeat_search"
            if tool in ("finish", "retrieve", "calculator"):
                return tool
            return "unknown_tool"

        def route_after_tool(state: AgentState) -> str:
            return END if state["step"] >= self.max_steps else "generate"

        g.add_node("generate", generate)
        g.add_node("parse_error", parse_error)
        g.add_node("finish", finish)
        g.add_node("retrieve", retrieve)
        g.add_node("calculator", calculator)
        g.add_node("unknown_tool", unknown_tool)
        g.add_node("repeat_search", repeat_search)

        g.set_entry_point("generate")
        g.add_conditional_edges(
            "generate",
            route_action,
            ["parse_error", "finish", "retrieve", "calculator", "unknown_tool", "repeat_search"],
        )
        for tool_node in ("retrieve", "calculator", "unknown_tool", "repeat_search"):
            g.add_conditional_edges(tool_node, route_after_tool, ["generate", END])
        g.add_edge("parse_error", END)
        g.add_edge("finish", END)
        return g.compile()

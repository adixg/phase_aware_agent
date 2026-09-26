"""LangGraph implementation of the ReAct agent.

Same contract as `agent.agent.ReActAgent`: `run(job_id, question, tracer)`
returns an `AgentResult`, and each loop iteration is one LLM_GENERATE span
followed by at most one RETRIEVE / TOOL_EXEC span. Spans are opened inside
the graph nodes, so the trace schema is unchanged by the migration.
"""

from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent.common import _ACTION_RE, AgentResult, _safe_eval
from agent.llm_backend import LLMBackend
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


class LangGraphReActAgent:
    """Thought -> Action -> Observation loop expressed as a LangGraph state machine.

    generate --(retrieve|calculator|unknown)--> tool node --> generate ...
             --(finish)--> finish --> END
             --(no parseable action)--> parse_error --> END
    The loop also ends once `max_steps` generations have run.
    """

    def __init__(self, llm: LLMBackend, retriever: LocalRetriever, *, max_steps: int = 8):
        self.llm = llm
        self.retriever = retriever
        self.max_steps = max_steps

    def run(self, job_id: str, question: str, tracer: Tracer) -> AgentResult:
        graph = self._build(tracer)
        initial: AgentState = {
            "transcript": [f"Question: {question}"],
            "step": 0,
            "tool": None,
            "arg": None,
            "answer": None,
            "success": False,
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
            with tracer.phase(Phase.LLM_GENERATE, step=step) as meta:
                text = self.llm.generate(prompt, step)
                meta["prompt_tokens"] = len(prompt.split())
                meta["output_tokens"] = len(text.split())
            match = _ACTION_RE.search(text)
            tool = match.group(1).lower() if match else None
            arg = match.group(2).strip() if match else None
            return {"transcript": state["transcript"] + [text], "tool": tool, "arg": arg}

        def parse_error(state: AgentState) -> dict:
            return {"transcript": state["transcript"] + ["Observation: could not parse an action; stopping."]}

        def finish(state: AgentState) -> dict:
            return {"answer": state["arg"], "success": True}

        def retrieve(state: AgentState) -> dict:
            step, arg = state["step"], state["arg"]
            with tracer.phase(Phase.RETRIEVE, step=step, query=arg) as meta:
                hits = self.retriever.retrieve(arg, top_k=1)
                meta["top_doc"] = hits[0].doc_id if hits else None
                meta["score"] = hits[0].score if hits else 0.0
            observation = hits[0].text if hits else "No results found."
            return _observe(state, observation)

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

        g.set_entry_point("generate")
        g.add_conditional_edges(
            "generate",
            route_action,
            ["parse_error", "finish", "retrieve", "calculator", "unknown_tool"],
        )
        for tool_node in ("retrieve", "calculator", "unknown_tool"):
            g.add_conditional_edges(tool_node, route_after_tool, ["generate", END])
        g.add_edge("parse_error", END)
        g.add_edge("finish", END)
        return g.compile()

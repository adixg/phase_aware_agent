from enum import Enum


class Phase(str, Enum):
    """The phases a ReAct-style RAG agent step can be in.

    Kept small on purpose: this is the vocabulary the scheduler ultimately
    conditions on (current phase, predicted next phase), so every phase here
    must correspond to a distinct resource profile (GPU decode vs CPU
    retrieval vs external tool I/O vs idle wait).
    """

    LLM_GENERATE = "llm_generate"   # prefill+decode: the ReAct "Thought"/"Action" call
    RETRIEVE = "retrieve"           # retriever lookup (+ any --retrieve-delay; see meta.search_s / delay_s)
    TOOL_EXEC = "tool_exec"         # non-retrieval tool call (e.g. calculator)
    WAIT = "wait"                   # queued/blocked on a shared resource
    WAIT_GPU = "wait_gpu"           # queued for an LLM slot (--llm-slots), before LLM_GENERATE
    WAIT_CPU = "wait_cpu"           # queued for a retrieval worker (--retrieve-workers), before RETRIEVE

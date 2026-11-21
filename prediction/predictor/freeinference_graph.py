"""Normalize FreeInference v1 graphs and extract causal prediction examples."""

from collections import Counter
from graphlib import TopologicalSorter


def reconstruct(session):
    nodes, edges, intervals = {}, [], []
    quality = Counter()

    def edge(source, target, kind, evidence, provenance="observed"):
        edges.append(dict(source=source, target=target, kind=kind,
                          evidence=evidence, provenance=provenance))

    def visit(chain, path, root_turn=None, parent_tool=None):
        requests = chain["requests"]
        if not requests:
            raise ValueError("Empty session chain")
        quality["missing_harness"] += not chain.get("harness")
        if parent_tool:
            nodes[path] = dict(
                id=path, kind="subagent", chain=path, root_turn=root_turn,
                harness=chain.get("harness"), start_ms=chain["start_ms"],
                end_ms=chain["end_ms"], observed_at_ms=None,
                parent_tool=parent_tool, completion_dependency="unknown",
            )
        previous = None
        for index, request in enumerate(requests):
            if request["step"] != index:
                raise ValueError("Unexpected request step")
            start, end = request["start_ms"], request["end_ms"]
            if not chain["start_ms"] <= start <= end <= chain["end_ms"]:
                raise ValueError("Request timestamp outside chain bounds or reversed")
            if previous and start < previous["start_ms"]:
                raise ValueError("Request starts out of order")
            rid = f"{path}/r{index}"
            turn = root_turn if parent_tool else request["turn_index"]
            ttft = request.get("ttft_ms")
            quality["missing_ttft"] += ttft is None
            quality["invalid_ttft"] += ttft is not None and not 0 <= ttft <= end - start
            tokens = request.get("provider_tokens") or {}
            for field in ("prompt", "completion", "cached"):
                quality[f"missing_provider_{field}"] += tokens.get(field) is None
                value = tokens.get(field)
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError("Invalid provider token count")
            definitions = request.get("tool_definition_names")
            node = nodes[rid] = dict(
                id=rid, kind="llm", chain=path, root_turn=turn,
                step=index, turn_index=request["turn_index"],
                trigger=request.get("step_trigger"), link_reason=request.get("link_reason"),
                start_ms=start, end_ms=end, gateway_duration_ms=end - start,
                harness=chain.get("harness"), model=request.get("model"),
                provider=request.get("provider"), stream=request.get("stream"),
                provider_tokens=tokens, role_tokens=request.get("role_tokens"),
                n_messages=request.get("n_messages"), tool_definition_names=definitions,
                finish_reason=request.get("finish_reason"), ttft_ms=ttft,
                tool_ids=[], sibling_tool_execution_order=(
                    "unknown" if len(request["tool_calls"]) > 1 else None
                ),
            )
            quality["missing_model"] += node["model"] is None
            quality["missing_role_tokens"] += node["role_tokens"] is None
            quality["missing_tool_definitions"] += definitions is None
            if parent_tool and index == 0:
                edge(path, rid, "child_entry", "nested_request_membership")
            if previous:
                gap = start - previous["end_ms"]
                quality["overlapping_consecutive_requests"] += gap < 0
                tool_ids = previous["tool_ids"]
                reason = node["link_reason"]
                if reason in {"tool_call_id", "assistant_message"}:
                    edge(previous["id"], rid, "continuation", reason)
                else:
                    quality["unknown_continuations"] += 1
                if tool_ids and node["trigger"] != "user" and node["trigger"] is not None:
                    intervals.append(dict(
                        emitter=previous["id"], next_request=rid,
                        start_ms=previous["end_ms"], end_ms=start,
                        signed_gap_ms=gap, elapsed_ms=gap if gap >= 0 else None,
                        kind="tool_phase_gateway_gap", tool_ids=tool_ids,
                        sibling_execution_order="unknown" if len(tool_ids) > 1 else None,
                        result_dependencies=(
                            "single_candidate" if reason == "tool_call_id" and len(tool_ids) == 1
                            else "unknown_subset"
                        ),
                    ))
                    if reason == "tool_call_id" and len(tool_ids) == 1:
                        edge(tool_ids[0], rid, "tool_feedback", "single_call_candidate", "inferred")
                    for tid in tool_ids:
                        latency = nodes[tid]["released_latency_ms"]
                        quality["tool_latency_formula_mismatch"] += latency != gap
                else:
                    quality["tool_latency_formula_mismatch"] += sum(
                        nodes[tid]["released_latency_ms"] is not None for tid in tool_ids
                    )
            for tool in request["tool_calls"]:
                tid = f'{rid}/t{tool["call_index"]}'
                if tid in nodes:
                    raise ValueError("Repeated tool call index")
                node["tool_ids"].append(tid)
                latency = tool.get("latency_ms")
                category = tool.get("category")
                child = tool.get("subagent")
                quality["missing_tool_category"] += category is None
                quality["missing_tool_latency"] += latency is None
                quality["negative_tool_latency"] += latency is not None and latency < 0
                quality["missing_tool_outcome"] += tool.get("outcome") is None
                quality["missing_tool_result_tokens"] += tool.get("result_tokens") is None
                quality["subagent_category_without_match"] += category == "Subagent" and child is None
                nodes[tid] = dict(
                    id=tid, kind="tool", chain=path, root_turn=turn,
                    emitter=rid, call_index=tool["call_index"], name=tool.get("name"),
                    category=category, outcome=tool.get("outcome"),
                    result_tokens=tool.get("result_tokens"), parse_status=tool.get("parse_status"),
                    released_latency_ms=latency, start_ms=None, end_ms=None,
                    duration_ms=None, emission_known_by_ms=end,
                )
                edge(rid, tid, "emits", "response_tool_list")
                if child is not None:
                    child_path = f"{tid}/child"
                    edge(tid, child_path, "spawn", "publisher_child_match", "inferred")
                    visit(child, child_path, turn, tid)
            if index == len(requests) - 1:
                quality["tool_latency_formula_mismatch"] += sum(
                    nodes[tid]["released_latency_ms"] is not None for tid in node["tool_ids"]
                )
            previous = node

    visit(session, "root")
    graph = dict(nodes=nodes, edges=edges, intervals=intervals, quality=quality)
    validate_graph(graph)
    return graph


def validate_graph(graph):
    nodes = graph["nodes"]
    sorter = TopologicalSorter({node_id: () for node_id in nodes})
    seen = set()
    signatures = {
        "emits": ("llm", "tool", "observed"),
        "continuation": ("llm", "llm", "observed"),
        "tool_feedback": ("tool", "llm", "inferred"),
        "spawn": ("tool", "subagent", "inferred"),
        "child_entry": ("subagent", "llm", "observed"),
    }
    for edge in graph["edges"]:
        source, target = edge["source"], edge["target"]
        key = (source, target, edge["kind"])
        if source not in nodes or target not in nodes or key in seen:
            raise ValueError("Missing endpoint or duplicate graph edge")
        signature = (nodes[source]["kind"], nodes[target]["kind"], edge["provenance"])
        if signature != signatures[edge["kind"]]:
            raise ValueError("Invalid edge types or provenance")
        left, right = nodes[source], nodes[target]
        if edge["kind"] == "emits" and (right["emitter"] != source or target not in left["tool_ids"]):
            raise ValueError("Tool does not belong to the emitting response")
        if edge["kind"] == "spawn" and right["parent_tool"] != source:
            raise ValueError("Subagent does not belong to the spawning tool")
        if edge["kind"] == "child_entry" and (right["chain"] != source or right["step"] != 0):
            raise ValueError("Invalid child session entry")
        if edge["kind"] == "continuation" and (
            left["chain"] != right["chain"] or left["step"] + 1 != right["step"]
            or edge["evidence"] != right["link_reason"]
        ):
            raise ValueError("Continuation disagrees with published request chain")
        if edge["kind"] == "tool_feedback":
            emitter = nodes[left["emitter"]]
            if (emitter["tool_ids"] != [source] or emitter["chain"] != right["chain"]
                    or emitter["step"] + 1 != right["step"]
                    or right["link_reason"] != "tool_call_id"):
                raise ValueError("Inferred feedback lacks a single-call candidate")
        seen.add(key)
        sorter.add(target, source)
    tuple(sorter.static_order())
    for interval in graph["intervals"]:
        emitter = nodes[interval["emitter"]]
        successor = nodes[interval["next_request"]]
        if (emitter["chain"] != successor["chain"] or
                emitter["step"] + 1 != successor["step"] or
                emitter["tool_ids"] != interval["tool_ids"] or
                emitter["end_ms"] != interval["start_ms"] or
                successor["start_ms"] != interval["end_ms"]):
            raise ValueError("Inconsistent tool-phase interval")


ROLE_KEYS = ("system", "user", "assistant", "tool", "tool_definitions")


def make_examples(graph, updates=False):
    """Yield paired snapshots and labels for the root's initial user turn."""
    nodes = graph["nodes"]
    first = nodes["root/r0"]
    origin = first["start_ms"]
    in_scope = {key: node for key, node in nodes.items() if node["root_turn"] == first["root_turn"]}
    llms = [node for node in in_scope.values() if node["kind"] == "llm"]
    root_llms = [node for node in llms if node["chain"] == "root"]
    tools = [node for node in in_scope.values() if node["kind"] == "tool"]
    children = [node for node in in_scope.values() if node["kind"] == "subagent"]
    endpoint = max(node["end_ms"] for node in llms)
    # A later user request can mark the end of the root turn, but it does not
    # establish that all outstanding tool or subagent work has finished.
    later_user_turn = min(
        (node["start_ms"] for node in nodes.values() if node["kind"] == "llm"
         and node["chain"] == "root" and node["trigger"] == "user"
         and node["root_turn"] != first["root_turn"]), default=None,
    )
    role = first["role_tokens"]
    prompt_estimate = (
        sum(role[key] for key in ROLE_KEYS)
        if role is not None and all(type(role.get(key)) is int and role[key] >= 0 for key in ROLE_KEYS)
        else None
    )
    cutoffs = [(origin, "admission")]
    if updates:
        cutoffs.extend(
            (time, "root_response_completed")
            for time in sorted({node["end_ms"] for node in root_llms})
            if origin < time and (later_user_turn is None or time <= later_user_turn)
        )
    for ordinal, (cutoff, event) in enumerate(cutoffs):
        admission = event == "admission"
        # The only guaranteed event at admission is the initial request input.
        completed = [] if admission else [node for node in root_llms if node["end_ms"] <= cutoff]
        admitted = [first] if admission else [node for node in root_llms if node["start_ms"] < cutoff]
        emitted = [nodes[tid] for node in completed for tid in node["tool_ids"]]
        intervals = [] if admission else [
            row for row in graph["intervals"] if row["emitter"] in in_scope
            and nodes[row["emitter"]]["chain"] == "root"
            and row["elapsed_ms"] is not None and row["end_ms"] < cutoff
            and row["next_request"] in in_scope
        ]
        # Update cutoffs include responses completed at the timestamp and
        # exclude new request admissions at the same timestamp.
        if any(node["end_ms"] > cutoff for node in completed) or any(
            node["start_ms"] >= cutoff for node in admitted if not admission
        ) or any(node["emission_known_by_ms"] > cutoff for node in emitted) or any(
            row["end_ms"] >= cutoff for row in intervals
        ):
            raise ValueError("Future event in feature history")
        features = dict(
            model=first["model"], harness=first["harness"], stream=first["stream"],
            initial_prompt_estimate=prompt_estimate, initial_role_tokens=role,
            initial_message_count=first["n_messages"],
            available_tool_names=first["tool_definition_names"],
            elapsed_ms=cutoff - origin, admitted_root_llms=len(admitted),
            completed_root_llms=len(completed), emitted_root_tool_count=len(emitted),
            emitted_root_tool_categories=dict(Counter(
                node["category"] for node in emitted if node["category"] is not None
            )), missing_emitted_root_tool_category=sum(node["category"] is None for node in emitted),
            completed_root_token_history=[node["provider_tokens"] for node in completed],
            closed_root_tool_phase_elapsed_ms=[row["elapsed_ms"] for row in intervals],
            observed_subagent_history=None,
        )
        availability = dict.fromkeys(features, origin)
        for key in ("elapsed_ms", "admitted_root_llms", "completed_root_llms", "emitted_root_tool_count",
                    "emitted_root_tool_categories", "missing_emitted_root_tool_category",
                    "completed_root_token_history", "closed_root_tool_phase_elapsed_ms"):
            availability[key] = cutoff
        availability["observed_subagent_history"] = None
        snapshot = dict(
            ordinal=ordinal, horizon="initial_user_turn", cutoff_ms=cutoff, event=event,
            features=features, feature_available_ms=availability,
            known_root_admission_ids=[node["id"] for node in admitted],
        )
        # Remaining LLMs are unfinished requests, including in-flight work.
        future_llms = llms if admission else [node for node in llms if node["end_ms"] > cutoff]
        future_tools = tools if admission else [node for node in tools if node["emission_known_by_ms"] > cutoff]
        # The child gateway start is a label-time proxy, not a runtime spawn notification.
        future_children = [node for node in children if node["start_ms"] >= cutoff]
        future = {node["id"] for node in future_llms + future_tools + future_children}
        supported = [index for index, edge in enumerate(graph["edges"])
                     if edge["source"] in in_scope and edge["target"] in in_scope
                     and (edge["source"] in future or edge["target"] in future)]
        categories = Counter(node["category"] for node in future_tools if node["category"] is not None)
        labels = dict(
            ordinal=ordinal, remaining_llm_count=len(future_llms),
            remaining_nested_llm_count=sum(node["chain"] != "root" for node in future_llms),
            future_tool_count=len(future_tools), future_tool_category_counts=dict(categories),
            unknown_future_tool_category_count=len(future_tools) - categories.total(),
            future_subagent_tool_emitted=bool(categories.get("Subagent")) if
            len(future_tools) == categories.total() or categories.get("Subagent") else None,
            future_matched_subagent_count=len(future_children),
            future_matched_subagent_occurs=bool(future_children),
            actual_subagent_branch_occurs=None,
            remaining_observed_duration_ms=endpoint - cutoff, observed_endpoint_ms=endpoint,
            true_workflow_completion_ms=None,
            later_root_turn_observed=later_user_turn is not None,
            terminal_tool_execution_unknown=any(
                nodes[tid]["end_ms"] is None
                for node in llms if node["end_ms"] == endpoint for tid in node["tool_ids"]
            ),
            future_node_ids=sorted(future), supported_edge_indices=supported,
        )
        yield snapshot, labels

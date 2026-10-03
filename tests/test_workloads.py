"""Every workload loader must satisfy the same contract; fixtures mirror the real file formats."""

import json
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from agent.common import _safe_eval
from agent.workloads import Task, WORKLOADS, get_workload, pool_documents


def _write_fixtures(root):
    hp = root / "hotpotqa"
    hp.mkdir()
    (hp / "hotpot_dev_distractor_v1.json").write_text(json.dumps([
        {"_id": f"h{i}", "question": f"q{i}?", "answer": f"Ans {i}", "type": "bridge", "level": "hard",
         "supporting_facts": [["T1", 0]], "context": [["T1", ["s1.", "s2."]], ["T2", ["s3."]]]}
        for i in range(5)
    ]))
    mu = root / "musique"
    mu.mkdir()
    (mu / "musique_ans_v1.0_dev.jsonl").write_text("\n".join(json.dumps(
        {"id": f"2hop__{i}", "question": f"q{i}?", "answer": f"Ans {i}", "answer_aliases": [f"A{i}"],
         "answerable": i != 4, "question_decomposition": [{}, {}],
         "paragraphs": [{"idx": 0, "title": "T", "paragraph_text": "text", "is_supporting": True}]}
    ) for i in range(6)))
    ar = root / "agentic_rag_tracer"
    ar.mkdir()
    for name in ("2hop_inference", "3hop_comparison"):
        n = int(name[0])
        rows = []
        for i in range(3):
            row = {f"hop_{h}": {"question": f"h{h}?", "answer": "a", "doc": f'"Title {h}"\nbody {h}',
                                "optional_answers": [f"Final {i}", f"Opt {i}"]} for h in range(1, n + 1)}
            rows.append({**row, "final_question": f"final {name} {i}?", "final_answer": f"Final {i}"})
        (ar / f"{name}.jsonl").write_text("\n".join(map(json.dumps, rows)))


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    _write_fixtures(root)
    return root


@pytest.fixture(params=sorted(WORKLOADS))
def workload(request, data_dir):
    return get_workload(request.param, data_dir)


def test_tasks_satisfy_contract(workload):
    tasks = list(workload.tasks(limit=3, seed=1))
    assert len(tasks) == 3
    assert all(isinstance(t, Task) and t.workload == workload.name for t in tasks)
    assert len({t.task_id for t in tasks}) == 3
    assert all(t.task_id.startswith(workload.name + "/") and t.question and t.gold_answers for t in tasks)
    docs = pool_documents(tasks)
    assert len({d.doc_id for d in docs}) == len(docs)


def test_sampling_is_deterministic(workload):
    ids = lambda **kw: [t.task_id for t in workload.tasks(**kw)]
    assert ids(limit=3, seed=7) == ids(limit=3, seed=7)


def test_gold_is_correct_and_wrong_is_not(workload):
    task = next(workload.tasks(limit=1))
    assert workload.is_correct(task, task.gold_answers[0])
    assert not workload.is_correct(task, "definitely wrong 12345")
    assert not workload.is_correct(task, None)


def test_normalization_ignores_case_articles_punctuation(data_dir):
    w = get_workload("hotpotqa", data_dir)
    task = next(w.tasks(limit=1))
    assert w.is_correct(task, "  the ANS 0!! ") or w.is_correct(task, task.gold_answers[0].upper())


def test_musique_skips_unanswerable_and_keeps_aliases(data_dir):
    w = get_workload("musique", data_dir)
    tasks = list(w.tasks())
    assert len(tasks) == 5 and all(not t.task_id.endswith("__4") for t in tasks)
    assert tasks[0].gold_answers == ("Ans 0", "A0") and tasks[0].meta["num_hops"] == 2


def test_agentic_rag_tracer_hops_and_docs(data_dir):
    tasks = list(get_workload("agentic_rag_tracer", data_dir).tasks())
    assert len(tasks) == 6
    three = next(t for t in tasks if t.meta["num_hops"] == 3)
    assert three.meta["category"] == "comparison" and len(three.documents) == 3
    assert three.documents[0].title == "Title 1" and three.documents[0].text == "body 1"
    assert tasks[0].gold_answers == ("Final 0", "Opt 0") and tasks[0].meta["final_answer"] == "Final 0"


def test_calculator_gold_matches_safe_eval():
    w = get_workload("calculator", None, num_tasks=20)
    tasks = list(w.tasks(seed=3))
    assert len(tasks) == 20 and not any(t.documents for t in tasks)
    for t in tasks:
        assert float(t.gold_answers[0]) == _safe_eval(t.meta["expression"])
    assert w.is_correct(tasks[0], tasks[0].gold_answers[0] + ".0")


def test_unknown_workload():
    with pytest.raises(ValueError):
        get_workload("nope")


def test_normalization_keeps_numbers_distinct_and_strips_accents():
    from agent.workloads.base import normalize_answer as n
    assert n("15,615") == "15615" and n("15,615") != n("15.615") and n("10.3%") != n("103")
    assert n("Café") == n("cafe") and n("U.S.") == "us"


def test_agentic_rag_tracer_drops_unrelated_or_empty_aliases(tmp_path):
    ar = tmp_path / "agentic_rag_tracer"
    ar.mkdir()
    hop = lambda opts: {"question": "q", "answer": "a", "doc": '"T"\nbody', "optional_answers": opts}
    rows = [
        {"hop_1": hop([]), "hop_2": hop(["Tarmo Mitt", "#", "An"]), "final_question": "q1?", "final_answer": "Tarmo Mitt"},
        {"hop_1": hop([]), "hop_2": hop(["The birth of X"]), "final_question": "q2?", "final_answer": "X"},
    ]
    (ar / "2hop_inference.jsonl").write_text("\n".join(map(json.dumps, rows)))
    a, b = get_workload("agentic_rag_tracer", tmp_path).tasks()
    assert a.gold_answers == ("Tarmo Mitt",) and b.gold_answers == ("X",)


def test_pool_documents_dedupes_by_text():
    from agent.workloads import Document
    d = lambda i: Document(doc_id=f"t{i}#hop_1", title="T", text="same")
    tasks = [Task(task_id=f"w/{i}", workload="w", question="q", gold_answers=("a",), documents=(d(i),)) for i in range(3)]
    assert [x.doc_id for x in pool_documents(tasks)] == ["t0#hop_1"]


def test_split_doc_handles_real_title_formats():
    from agent.workloads.agentic_rag_tracer import _split_doc
    assert _split_doc('"Santry"\nbody') == ("Santry", "body")
    assert _split_doc("Dumfries\nbody") == ("Dumfries", "body")          # ~5% of real passages have no quotes
    assert _split_doc('"Joaquín ""Jack"" García"\nbody')[0] == 'Joaquín "Jack" García'


# --- real-format fixtures: untouched records copied from the six AgenticRAGTracer files ---

REAL_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def real_tasks():
    return {t.task_id.split("/", 1)[1]: t for t in get_workload("agentic_rag_tracer", REAL_DIR).tasks()}


def test_real_records_load_with_expected_shape(real_tasks):
    assert len(real_tasks) == 10
    assert Counter(t.meta["num_hops"] for t in real_tasks.values()) == {2: 6, 3: 2, 4: 2}
    assert Counter(t.meta["category"] for t in real_tasks.values()) == {"comparison": 5, "inference": 5}
    for t in real_tasks.values():
        assert len(t.documents) == t.meta["num_hops"] == len(t.meta["hop_questions"])
        assert all(d.title and d.text for d in t.documents)
        assert t.meta["final_answer"] == t.gold_answers[0]


def test_real_titles_unquoted_and_csv_escaped(real_tasks):
    assert [d.title for d in real_tasks["2hop_comparison/2"].documents] == ["Aik Thi Misaal", "Denma"]  # no quotes in source
    # the source text is Unicode-decomposed (NFD), so compare after composing
    title = real_tasks["2hop_inference/2"].documents[0].title
    assert unicodedata.normalize("NFC", title) == 'Joaquín "Jack" García'


def test_real_aliases_come_from_last_hop_and_are_filtered(real_tasks):
    t = real_tasks["2hop_comparison/0"]
    assert "RLL(1,3) encoding" in t.gold_answers and len(t.gold_answers) == 9
    # alias list describes something else ("The end of Aik Thi Misaal"): only the final answer survives
    assert real_tasks["2hop_comparison/2"].gold_answers == ("Aik Thi Misaal",)
    # "#" alias normalizes to nothing and is dropped; "No." keeps its real aliases
    yes_no = real_tasks["2hop_inference/1"]
    assert yes_no.gold_answers == ('No.', 'Number', 'No', 'N°')


def test_real_gold_matches_and_wrong_answers_do_not(real_tasks):
    w = get_workload("agentic_rag_tracer", REAL_DIR)
    for t in real_tasks.values():
        assert all(w.is_correct(t, g) for g in t.gold_answers)
        assert not w.is_correct(t, "")
        assert not w.is_correct(t, "definitely wrong 12345")
        assert not any(w.is_correct(t, o.gold_answers[0]) for o in real_tasks.values() if o is not t)
    # comparison: the other candidate named in the question is wrong
    t = real_tasks["2hop_comparison/1"]
    assert w.is_correct(t, "the carbon12 building.") and not w.is_correct(t, "William Minott House")
    assert not w.is_correct(t, "William Minott House and Carbon12 building")


def test_real_sample_is_deterministic_and_representative():
    w = get_workload("agentic_rag_tracer", REAL_DIR)
    ids = lambda **kw: [t.task_id for t in w.tasks(**kw)]
    assert ids(limit=3, seed=0) == ids(limit=3, seed=0) == [
        "agentic_rag_tracer/2hop_comparison/0", "agentic_rag_tracer/3hop_comparison/0", "agentic_rag_tracer/4hop_inference/0"]
    assert ids(limit=3, seed=1) != ids(limit=3, seed=0)
    assert ids(limit=3, seed=0) == sorted(ids(limit=3, seed=0), key=ids().index)   # file order preserved
    assert ids(limit=100) == ids()                                                # limit above size -> everything
    sample = list(w.tasks(limit=3, seed=0))
    assert sorted(t.meta["num_hops"] for t in sample) == [2, 3, 4]


def test_real_pool_has_unique_ids_and_dedupes_text(real_tasks):
    tasks = list(real_tasks.values())
    docs = [d for t in tasks for d in t.documents]
    assert len({d.doc_id for d in docs}) == len(docs)
    pooled = pool_documents(tasks)
    assert len(pooled) == len({(d.title, d.text) for d in docs}) <= len(docs)


def test_reduced_slice_is_stratified_deterministic_and_checked():
    from agent.workloads.agentic_rag_tracer import SLICE_PLAN
    assert sum(SLICE_PLAN.values()) == 100
    w = get_workload("agentic_rag_tracer", REAL_DIR)
    plan = {(2, "comparison"): 2, (3, "inference"): 1, (4, "comparison"): 1}
    a, b = w.reduced_slice(plan, seed=5), w.reduced_slice(plan, seed=5)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert Counter((t.meta["num_hops"], t.meta["category"]) for t in a) == plan
    file_order = [t.task_id for t in w.tasks()]
    assert [t.task_id for t in a] == sorted((t.task_id for t in a), key=file_order.index)
    with pytest.raises(ValueError):
        w.reduced_slice({(2, "comparison"): 50})

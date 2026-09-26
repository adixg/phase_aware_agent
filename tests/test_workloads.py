"""Every workload loader must satisfy the same contract; fixtures mirror the real file formats."""

import json

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
                                "optional_answers": [f"Opt {i}"]} for h in range(1, n + 1)}
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
    assert "Opt 0" in tasks[0].gold_answers


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

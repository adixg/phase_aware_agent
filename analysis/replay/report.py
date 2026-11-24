"""Tables and the stop-rule verdict from run_gate.py output.

    python analysis/replay/report.py data/freeinference/gate/class_l40s.jsonl
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict

ORACLE_ADMISSION = ("llf_oracle", "srpt_oracle")
CHEAP_ADMISSION = ("fcfs", "returning", "las", "edf", "llf_cheap", "srpt_cheap")
SIGNALS = ("calls", "gaps", "tokens")
REALISTIC_KV = (1e6, 2e6, 4e6)   # roughly one to two large GPUs at ~56 KB per token
SMALL, LARGE = 3.0, 5.0          # points of attainment
JCT_BAR = 10.0                   # percent turn-latency reduction vs attained service that justifies a predictor
ATLAS_STANDIN = ("las", "lru")   # the sim has no parallel requests, so ATLAS's critical path is just attained service
DEPTHS = (1, 2, 4, 8)


def load(path: str, field: str = "attainment"):
    cells = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if r.get(field) is not None:
            cells[(r["load"], r["tight"], r["kv"])][(r["policy"], r["eviction"], r["day"])] = r[field]
    return cells


def per_day(cell: dict, pair: tuple[str, str]) -> dict[str, float]:
    return {d: v for (p, e, d), v in cell.items() if (p, e) == pair}


def mean(cell: dict, pair: tuple[str, str]) -> float | None:
    v = per_day(cell, pair)
    return st.mean(v.values()) if v else None


def best(cell: dict, pairs) -> tuple[tuple[str, str] | None, float]:
    scored = [(mean(cell, p), p) for p in pairs if mean(cell, p) is not None]
    return (None, float("nan")) if not scored else max(scored, key=lambda x: x[0])[::-1]


def gap(cell: dict, oracle_pairs, cheap_pairs):
    """Best oracle minus best cheap, in points, with the across-day standard error of that difference."""
    po, o = best(cell, oracle_pairs)
    pc, c = best(cell, cheap_pairs)
    if po is None or pc is None:
        return None
    a, b = per_day(cell, po), per_day(cell, pc)
    diffs = [100 * (a[d] - b[d]) for d in a if d in b]
    se = st.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else float("nan")
    return 100 * (o - c), se, po, pc, 100 * c


def reduction(cell: dict, pair: tuple[str, str], base: tuple[str, str] = ATLAS_STANDIN) -> float | None:
    """Percent drop of a latency metric versus the baseline policy (positive = better), per-day mean."""
    a, b = per_day(cell, pair), per_day(cell, base)
    days = [d for d in a if d in b and b[d] > 0]
    return st.mean(100 * (b[d] - a[d]) / b[d] for d in days) if days else None


def jct_section(path: str) -> None:
    """Turn latency (the JCT of a turn) against attained service, in the cells where memory is realistic."""
    mean_l, p95_l = load(path, "mean_latency"), load(path, "p95_latency")
    best_of = lambda cell, pairs: max((r for r in (reduction(cell, p) for p in pairs) if r is not None), default=None)
    oracle, cheap = [("psrpt_oracle", "lru")], [("psrpt_cheap", "lru")]
    print("\n\nTurn latency (JCT) versus attained service (LAS; stands in for ATLAS, which differs only with parallel requests)")
    print("percent reduction, positive = faster; shortest-remaining-time-first with history vs with the true future, per cell")
    print("(no deadline logic in these policies: the deadline-aware ones trade mean latency for attainment, so they cannot test JCT)")
    print("%6s %6s %6s | %10s %10s | %10s %10s" % ("load", "tight", "KV", "mean cheap", "mean orcl", "p95 cheap", "p95 orcl"))
    rows = []
    for k in sorted(mean_l):
        if k[2] != 0 and k[2] not in REALISTIC_KV:
            continue
        m, p = mean_l[k], p95_l.get(k, {})
        vals = (best_of(m, cheap), best_of(m, oracle), best_of(p, cheap), best_of(p, oracle))
        if None in vals:
            continue
        rows.append((k, vals))
        print("%6s %6s %6s | %+9.1f%% %+9.1f%% | %+9.1f%% %+9.1f%%" % (
            k[0], k[1], "inf" if k[2] == 0 else "%dM" % (k[2] / 1e6), *vals))
    for name, i in (("mean", 1), ("p95", 3)):
        g = [v[i] for _, v in rows]
        if g:
            print(f"  oracle {name} latency: max {max(g):+.1f}%, median {st.median(g):+.1f}%; "
                  f"{sum(x >= JCT_BAR for x in g)}/{len(g)} cells at the {JCT_BAR:.0f}% bar -> "
                  f"{'build' if max(g) >= JCT_BAR else 'drop the predictor'}")


def ladder_rows(cells: dict) -> list[dict]:
    """Per realistic-KV cell, attainment at each rung of the information ladder and what each step adds:
    LAS (attained service, the ATLAS stand-in) -> EDF (adds the deadlines) -> oracle (adds the future)."""
    rows = []
    for (ld, tight, kv), c in sorted(cells.items()):
        if kv not in REALISTIC_KV:
            continue
        las, edf = mean(c, ATLAS_STANDIN), mean(c, ("edf", "lru"))
        po, o = best(c, [(a, "lru") for a in ORACLE_ADMISSION])
        if None in (las, edf, po):
            continue
        a, b = per_day(c, po), per_day(c, ("edf", "lru"))
        d = [100 * (a[x] - b[x]) for x in a if x in b]
        rows.append({"load": ld, "tight": tight, "kv": kv, "las": 100 * las, "edf": 100 * edf, "oracle": 100 * o,
                     "deadlines": 100 * (edf - las), "future": 100 * (o - edf), "total": 100 * (o - las),
                     "se": st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")})
    return rows


def ladder_section(cells: dict) -> None:
    """What knowing the deadlines adds over attained service, and what knowing the future adds over that."""
    rows = ladder_rows(cells)
    print("\n\nInformation ladder: attained service (LAS) -> + deadlines (EDF) -> + the future (oracle)")
    print("attainment %, mean over test days; the two steps in points. Only the last step is what a predictor can supply.")
    print("%6s %6s %6s | %6s %6s %6s | %10s %12s | %10s" % ("load", "tight", "KV", "LAS", "EDF", "oracle", "deadlines", "future", "oracle-LAS"))
    for r in rows:
        print("%6s %6s %6s | %5.1f%% %5.1f%% %5.1f%% | %+9.1f %+8.1f±%.1f | %+10.1f" % (
            r["load"], r["tight"], "%dM" % (r["kv"] / 1e6), r["las"], r["edf"], r["oracle"],
            r["deadlines"], r["future"], r["se"], r["total"]))
    if not rows:
        return
    for key, name in (("deadlines", "deadlines (EDF - LAS)"), ("future", "the future (oracle - EDF)"), ("total", "oracle - LAS")):
        g = [r[key] for r in rows]
        print(f"  {name:27s} max {max(g):+.1f}, median {st.median(g):+.1f}; {sum(x >= SMALL for x in g)}/{len(g)} cells at "
              f"{SMALL:.0f}+ points, {sum(x >= LARGE for x in g)}/{len(g)} at {LARGE:.0f}+")
    top = max(r["future"] for r in rows)
    decision = "drop the predictor" if top < SMALL else ("build" if top >= LARGE else "noise-test the oracle")
    print(f"  verdict on the future step, the one a predictor supplies: {decision}")
    by_kv = {kv: [r for r in rows if r["kv"] == kv] for kv in sorted({r["kv"] for r in rows})}
    print("  mean by KV, deadlines / future: " + ", ".join(
        "%dM %+.1f / %+.1f" % (kv / 1e6, st.mean(r["deadlines"] for r in v), st.mean(r["future"] for r in v)) for kv, v in by_kv.items()))


def depth_section(path: str) -> None:
    """How many steps of lookahead are worth it: k = 0 is history only, k = all is the oracle."""
    att, mean_l = load(path), load(path, "mean_latency")
    if not any(pol.endswith(tuple(f"_la{k}" for k in DEPTHS)) for c in att.values() for pol, _, _ in c):
        return      # swept without --lookahead
    print("\n\nLookahead depth (eviction lru, averaged over realistic KV sizes, all loads and tightnesses)")
    print("attainment points over the history-only policy; mean-latency percent drop vs LAS")
    print("(the k=0 baseline here is LLF/SRPT with historical averages, which is far worse than EDF: read the curve for how")
    print(" deep a window must be to match an oracle, and the information ladder above for what the future adds over EDF)")
    print("%6s | %s" % ("family", "   ".join("k=%-3s att / jct" % x for x in ("0",) + DEPTHS + ("all",))))
    for fam in ("llf", "srpt", "psrpt"):
        names = [f"{fam}_cheap"] + [f"{fam}_la{k}" for k in DEPTHS] + [f"{fam}_oracle"]
        cols = []
        for n in names:
            a, j = [], []
            for key in att:
                if key[2] not in REALISTIC_KV:
                    continue
                base, v = mean(att[key], (f"{fam}_cheap", "lru")), mean(att[key], (n, "lru"))
                if v is not None and base is not None:
                    a.append(100 * (v - base))
                r = reduction(mean_l.get(key, {}), (n, "lru"))
                if r is not None:
                    j.append(r)
            cols.append("%+5.1f / %+5.1f%%" % (st.mean(a) if a else float("nan"), st.mean(j) if j else float("nan")))
        print("%6s | %s" % (fam, "   ".join(cols)))


FORECAST_VARIANTS = (("predfirst", "forecast, 1st turns"), ("predall", "forecast, all turns"), ("calls", "true call count"))


def decompose_section(path: str) -> None:
    """Which part of a session's future carries the oracle's gain, when a deadline covers the whole session.
    Every variant is the same LLF/SRPT policy; only what it is told about the remaining calls changes."""
    att, mean_l = load(path), load(path, "mean_latency")
    if not any(pol.endswith("_ucount") for c in att.values() for pol, _, _ in c):
        return      # swept without --decompose
    steps = [("history", "cheap"), ("turn", "ucur"), ("+ turns", "ucount"), ("all calls", "calls"), ("oracle", "oracle")]
    best_of = lambda c, suffix: best(c, [(f"{f}_{suffix}", "lru") for f in ("llf", "srpt")])[1]
    cells = {k: c for k, c in att.items() if k[2] in REALISTIC_KV}
    print("\n\nWhere a session oracle's gain comes from (deadlines over whole sessions; eviction lru, realistic KV sizes)")
    print("Each column is the better of LLF and SRPT told more of the future: the calls left in the user's current turn")
    print("(`turn`), plus the true number of turns still to come at the average size (`+ turns`), the true call count of the")
    print("whole session (`all calls`), and everything (`oracle`). `history` knows none of it. Attainment %, mean over cells.")
    print("%6s %6s | %7s %8s %8s %9s %9s %8s | share of the history-to-oracle gap closed by: turn / + turns / all calls" % (
        "KV", "cells", "EDF", *[n for n, _ in steps]))
    for kv in sorted({k[2] for k in cells}):
        ks = [k for k in cells if k[2] == kv]
        edf = [mean(cells[k], ("edf", "lru")) for k in ks]
        cols = [[best_of(cells[k], suf) for k in ks] for _, suf in steps]
        if any(x is None for x in edf) or any(x is None for col in cols for x in col):
            continue
        m = [100 * st.mean(col) for col in cols]
        gap_total = m[4] - m[0]
        share = ["%4.0f%%" % (100 * (m[i] - m[0]) / gap_total) if gap_total > 0.05 else "   n/a" for i in (1, 2, 3)]
        print("%5dM %6d | %6.1f%% %7.1f%% %7.1f%% %8.1f%% %8.1f%% %7.1f%% | %s / %s / %s" % (
            kv / 1e6, len(ks), 100 * st.mean(edf), *m, *share))
    print("\nMean latency, percent faster than attained service (pure shortest-remaining-time first; unlimited memory and realistic KV sizes)")
    print("%6s | %8s %8s %9s %9s %8s" % ("KV", "history", "turn", "+ turns", "all calls", "oracle"))
    names = [("psrpt_cheap", "lru"), ("psrpt_ucur", "lru"), ("psrpt_ucount", "lru"), ("psrpt_calls", "lru"), ("psrpt_oracle", "lru")]
    for kv in sorted({k[2] for k in mean_l if k[2] == 0 or k[2] in REALISTIC_KV}):
        ks = [k for k in mean_l if k[2] == kv]
        vals = [[reduction(mean_l[k], n) for k in ks] for n in names]
        if any(x is None for col in vals for x in col):
            continue
        print("%6s | %+7.1f%% %+7.1f%% %+8.1f%% %+8.1f%% %+7.1f%%" % (
            "inf" if kv == 0 else "%dM" % (kv / 1e6), *[st.mean(col) for col in vals]))


def forecast_section(path: str) -> None:
    """What the fitted admission forecaster buys over history, against the ceiling of knowing the call count."""
    att, mean_l = load(path), load(path, "mean_latency")
    if not any(pol.endswith("_predall") for c in att.values() for pol, _, _ in c):
        return      # swept without --forecast
    cells = [k for k in att if k[2] in REALISTIC_KV]
    print("\n\nAdmission forecaster (Andrew's EmpiricalPredictor, fitted on the training week; eviction lru, realistic KV sizes,")
    print("all loads and tightnesses). Same policy, only the remaining-call estimate changes; the rest stays historical.")
    print("attainment points over the same policy with history, mean over cells (and cells where it is better / worse by over 0.5)")
    print("%6s | %s | %s" % ("family", "   ".join("%-24s" % name for _, name in FORECAST_VARIANTS), "forecast closes"))
    for fam in ("llf", "srpt"):
        deltas = {}
        for suffix, _ in FORECAST_VARIANTS:
            deltas[suffix] = [100 * (mean(att[k], (f"{fam}_{suffix}", "lru")) - mean(att[k], (f"{fam}_cheap", "lru")))
                              for k in cells if mean(att[k], (f"{fam}_{suffix}", "lru")) is not None]
        cols = ["%+5.1f (%2d up/%2d down)" % (st.mean(v), sum(x > 0.5 for x in v), sum(x < -0.5 for x in v))
                for v in deltas.values()]
        closed = 100 * st.mean(deltas["predall"]) / st.mean(deltas["calls"]) if st.mean(deltas["calls"]) else float("nan")
        print("%6s | %s | %5.1f%% of the gap to true calls" % (fam, "   ".join("%-24s" % c for c in cols), closed))
    print("turn latency, shortest-remaining-time-first with no deadline logic: percent drop versus LAS, median over cells")
    print("%6s | %s" % ("", "   ".join("%-14s" % name for name in ("history", "forecast 1st", "forecast all", "true calls", "full oracle"))))
    cols = []
    for suffix in ("cheap", "predfirst", "predall", "calls", "oracle"):
        r = [x for k in cells if (x := reduction(mean_l.get(k, {}), (f"psrpt_{suffix}", "lru"))) is not None]
        cols.append("%+6.1f%%" % st.median(r) if r else "   n/a")
    print("%6s | %s" % ("psrpt", "   ".join("%-14s" % c for c in cols)))


def online_section(path: str) -> None:
    """The online forecaster in the simulator, against the committed simulator gate (analysis/dag/online_gates.py):
    the share of the gap between history and the truth that it closes, and no worse p95 latency than history."""
    from analysis.dag.online_gates import SIM_MIN
    att, p95 = load(path), load(path, "p95_latency")
    if not any(pol.endswith("_online") for c in att.values() for pol, _, _ in c):
        return      # swept without --online
    cells = [k for k in att if k[2] in REALISTIC_KV]
    print("\n\nOnline forecaster in the simulator (exploratory: it failed its offline gate). Realistic KV sizes, all loads and")
    print("tightnesses. Attainment points over history; share = online gain / the gain from the truth.")
    print("%-26s %8s %8s %8s %9s %12s" % ("", "online", "truth", "share", "up/down", "p95 vs hist"))

    def row(name, hist, online, truth):
        d = [(100 * (mean(att[k], online) - mean(att[k], hist)), 100 * (mean(att[k], truth) - mean(att[k], hist)))
             for k in cells if None not in (mean(att[k], online), mean(att[k], hist), mean(att[k], truth))]
        if not d:
            return
        gain, ceiling = st.mean(x for x, _ in d), st.mean(y for _, y in d)
        lat = [r for k in cells if (r := reduction(p95.get(k, {}), online, hist)) is not None]
        share = gain / ceiling if ceiling else float("nan")
        print("%-26s %+8.2f %+8.2f %7.0f%% %4d/%-4d %+11.1f%%  %s" % (
            name, gain, ceiling, 100 * share, sum(x > 0.5 for x, _ in d), sum(x < -0.5 for x, _ in d),
            st.mean(lat) if lat else float("nan"),
            "PASS" if share >= SIM_MIN and (not lat or st.mean(lat) >= 0) else "fail"))

    for fam in ("llf", "srpt"):
        row(f"admission {fam}", (f"{fam}_cheap", "lru"), (f"{fam}_online", "lru"), (f"{fam}_calls", "lru"))
    for adm in ("fcfs", "edf", "llf_cheap", "srpt_cheap"):
        row(f"eviction under {adm}", (adm, "cheap"), (adm, "online"), (adm, "oracle"))


def label(pair) -> str:
    return "/".join(pair).replace("_cheap", "~").replace("_oracle", "*")


def main(path: str) -> None:
    cells = load(path)
    print(f"# {path}\n")
    if not any(("fcfs", "lru") in c.keys() or any(k[:2] == ("fcfs", "lru") for k in c) for c in cells.values()):
        print("(a partial run, selected with --only: only the sections that need no full grid)")
        ladder_section(cells)
        forecast_section(path)
        online_section(path)
        decompose_section(path)
        jct_section(path)
        return
    verdicts = defaultdict(list)
    adm_cheap = [(a, "lru") for a in CHEAP_ADMISSION]
    adm_oracle = [(a, "lru") for a in ORACLE_ADMISSION]
    evict_cheap = [(a, e) for a in ("fcfs", "edf", "llf_cheap", "srpt_cheap") for e in ("lru", "cheap")]
    joint_cheap = adm_cheap + evict_cheap
    joint_oracle = [(a, e) for a in ORACLE_ADMISSION for e in ("lru", "oracle")] + \
        [(a, "oracle") for a in ("fcfs", "edf", "llf_cheap", "srpt_cheap")]

    for (load_t, tight) in sorted({(k[0], k[1]) for k in cells}):
        print(f"\nload {load_t}, tightness {tight}")
        print("%8s %9s %9s %10s | %8s | %8s | %8s | %s" % ("KV", "FCFS/LRU", "EDF/LRU", "bestCheap",
                                                          "adm gap", "evict", "joint", "best cheap / oracle"))
        for kv in sorted({k[2] for k in cells if k[0] == load_t and k[1] == tight}):
            c = cells[(load_t, tight, kv)]
            kvs = "inf" if kv == 0 else "%dM" % (kv / 1e6)
            fcfs, edf = mean(c, ("fcfs", "lru")), mean(c, ("edf", "lru"))
            ga = gap(c, adm_oracle, adm_cheap)
            line = "%8s %8.1f%% %8.1f%%" % (kvs, 100 * fcfs, 100 * edf)
            if kv == 0:                  # unlimited memory: nothing to evict
                print(line, "%9.1f%% | %+5.1f±%.1f |        - |        - | %s / %s" % (
                    ga[4], ga[0], ga[1], label(ga[3]), label(ga[2])))
                verdicts["admission"].append((load_t, tight, kv, ga[0]))
                continue
            ge = max((gap(c, [(a, "oracle")], [(a, "lru"), (a, "cheap")])
                      for a in ("fcfs", "edf", "llf_cheap", "srpt_cheap")), key=lambda g: g[0])
            gj = gap(c, joint_oracle, joint_cheap)
            print(line, "%9.1f%% | %+5.1f±%.1f | %+5.1f±%.1f | %+5.1f±%.1f | %s / %s" % (
                gj[4], ga[0], ga[1], ge[0], ge[1], gj[0], gj[1], label(gj[3]), label(gj[2])))
            if kv in REALISTIC_KV:
                verdicts["admission"].append((load_t, tight, kv, ga[0]))
                verdicts["eviction"].append((load_t, tight, kv, ge[0]))
                verdicts["joint"].append((load_t, tight, kv, gj[0]))

    print("\n\nWhich signal is worth having? Attainment points over the same policy family with no true signals")
    print("(eviction lru, averaged over the realistic KV sizes and all loads and tightnesses)")
    for fam in ("llf", "srpt"):
        rows = {s: [] for s in SIGNALS + ("all",)}
        for k, c in cells.items():
            if k[2] not in REALISTIC_KV:
                continue
            base = mean(c, (f"{fam}_cheap", "lru"))
            for s in SIGNALS:
                v = mean(c, (f"{fam}_{s}", "lru"))
                if v is not None:
                    rows[s].append(100 * (v - base))
            rows["all"].append(100 * (mean(c, (f"{fam}_oracle", "lru")) - base))
        print(f"  {fam}: " + "  ".join(f"{s} {st.mean(v):+.1f}" for s, v in rows.items() if v))

    ladder_section(cells)
    depth_section(path)
    forecast_section(path)
    online_section(path)
    decompose_section(path)
    jct_section(path)

    print("\n\nVerdict on attainment (realistic KV: 1, 2, 4M tokens; unlimited memory also counted for admission)")
    for ch, v in verdicts.items():
        gaps = [x[3] for x in v]
        share_large = sum(g >= LARGE for g in gaps) / len(gaps)
        n_small = sum(g < SMALL for g in gaps)
        peak = max(v, key=lambda x: x[3])
        decision = "drop the predictor" if max(gaps) < SMALL else \
            ("build" if max(gaps) >= LARGE else "noise-test the oracle")
        print(f"  {ch}: gap max {max(gaps):+.1f} at load {peak[0]} tight {peak[1]} KV {peak[2] / 1e6:.0f}M, "
              f"median {st.median(gaps):+.1f}; {n_small}/{len(gaps)} cells under {SMALL:.0f} points, "
              f"{100 * share_large:.0f}% at {LARGE:.0f}+ -> {decision}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)

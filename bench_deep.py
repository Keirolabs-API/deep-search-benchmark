"""SimpleQA benchmark of Keiro's deep search endpoint (POST /api/v2/deep).

Per question: is the gold answer in the token-budgeted evidence (what an agent
reads), and anywhere in the response (results with page text). Exact-match
misses go to an LLM judge that counts variant forms (date order, abbreviations,
numerals vs words, transliterations, paraphrase) and rejects derived answers,
partial answers and different values.

Sample: random.seed(42), random.sample(rows, OFFSET + N)[OFFSET:]. The
defaults reproduce the published 200-question run.

Env: KEIRO_API_KEY, OPENROUTER_API_KEY (judge), N (200), OFFSET (300),
     CONC (3), BUDGET (1000), BASE (https://api.keirolabs.cloud/api/v2),
     JUDGE_MODEL, OUT (results.json)
"""
import asyncio
import csv
import json
import os
import random
import re
import time
import urllib.request

import httpx

BASE = os.getenv("BASE", "https://api.keirolabs.cloud/api/v2")
KEY = os.environ["KEIRO_API_KEY"]
N, OFFSET = int(os.getenv("N", "200")), int(os.getenv("OFFSET", "300"))
CONC, BUDGET = int(os.getenv("CONC", "3")), int(os.getenv("BUDGET", "1000"))
CSV = os.getenv("SIMPLEQA_CSV", "data/simple_qa_test_set.csv")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "google/gemini-3.1-flash-lite")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OUT = os.getenv("OUT", "results.json")


def norm(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def has(text: str, gold: str) -> bool:
    return bool(text) and norm(gold) in norm(text)


def windows(texts: list[str], gold: str, n: int = 6) -> list[str]:
    """Windows around the gold answer's rarest words: what the judge reads."""
    words = [w for w in norm(gold).split() if len(w) >= 3 or w.isdigit()] or norm(gold).split()
    blob = [t for t in texts if t]
    if not words or not blob:
        return []
    rare = sorted(words, key=lambda w: sum(t.lower().count(w) for t in blob) or 1e9)[:2]
    out = []
    for t in blob:
        low = t.lower()
        for w in rare:
            i = low.find(w)
            if i >= 0:
                out.append(t[max(0, i - 400):i + 400])
                break
        if len(out) >= n:
            break
    return out


async def judge(c: httpx.AsyncClient, q: str, gold: str, texts: list[str]) -> bool:
    ex = "\n---\n".join(texts)[:12000]
    if not ex or not OPENROUTER_KEY:
        return False
    prompt = (f"Question: {q}\nGold answer: {gold}\n\nExcerpts:\n{ex}\n\n"
              "Does any excerpt explicitly state the gold answer? Count variant forms: date order, "
              "abbreviations, numerals vs words, transliterations, paraphrases, every part of a compound "
              "answer stated. Do NOT count answers you would have to compute or infer, partial answers, "
              "or a different value. Reply with exactly YES or NO.")
    for _ in range(2):
        try:
            r = await c.post("https://openrouter.ai/api/v1/chat/completions", timeout=60,
                             headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                             json={"model": JUDGE_MODEL, "temperature": 0, "max_tokens": 800, **({"reasoning": {"effort": "minimal"}} if "minimax" in JUDGE_MODEL else {}),
                                   "messages": [{"role": "user", "content": prompt}]})
            words = re.findall(r"\b(YES|NO)\b", (r.json()["choices"][0]["message"]["content"] or "").upper())
            return bool(words) and words[-1] == "YES"  # last verdict: the model may restate the rules first
        except Exception:
            await asyncio.sleep(1)
    return False


async def one(c: httpx.AsyncClient, row: dict) -> dict:
    q, gold = row["problem"].strip(), row["answer"].strip()
    t = time.time()
    d, err = None, ""
    for attempt in range(3):  # retry 429/5xx and network errors, never a 200
        try:
            r = await c.post(f"{BASE}/deep", json={"query": q, "budget": BUDGET, "content": True}, timeout=120)
            if r.status_code == 200:
                d = r.json()
                break
            err = f"HTTP {r.status_code}: {r.text[:150]}"
        except Exception as e:
            err = str(e)[:200]
        await asyncio.sleep(5 * (attempt + 1))
    if d is None:
        return {"q": q, "gold": gold, "error": err}
    res, ev = d["results"], d["evidence"]
    for x in res:
        x["snippet"], x["content"] = x.get("snippet") or "", x.get("content") or ""
    rank = next((i for i, x in enumerate(res) if has(x["snippet"], gold) or has(x["content"], gold)), None)
    hit, ev_hit = rank is not None, any(has(e["text"], gold) for e in ev)
    ev_texts = [e["text"] for e in ev]
    j_ev = ev_hit or await judge(c, q, gold, windows(ev_texts, gold) or ev_texts[:3])
    j_hit = hit or j_ev or await judge(c, q, gold, windows([x["snippet"] for x in res] + [x["content"] for x in res], gold))
    return {
        "q": q, "gold": gold,
        "hit": hit, "evidence_hit": ev_hit, "judged_hit": j_hit, "judged_evidence_hit": j_ev,
        "rank": rank, "status": d["answer_status"], "confidence": d["confidence"],
        "evidence_tokens": d["meta"]["evidence_tokens"], "latency_ms": int((time.time() - t) * 1000),
        "early_exit": d["meta"]["early_exit"], "fetched": d["meta"]["fetched"],
        "agreeing_domains": d.get("agreeing_domains", []), "checked": d["meta"].get("checked"),
        "evidence": [{"text": e["text"], "url": e["url"], "likelihood": e.get("likelihood"),
                      "corroborated": e.get("corroborated")} for e in ev],
        "urls": [x["url"] for x in res],
    }


async def main():
    os.makedirs(os.path.dirname(CSV) or ".", exist_ok=True)
    if not os.path.exists(CSV):
        urllib.request.urlretrieve("https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv", CSV)
    rows = list(csv.DictReader(open(CSV, newline="", encoding="utf-8")))
    random.seed(42)
    sample = random.sample(rows, N + OFFSET)[OFFSET:]
    sem, recs = asyncio.Semaphore(CONC), []
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {KEY}"}) as c:
        async def run(i, row):
            async with sem:
                rec = await one(c, row)
            recs.append(rec)
            print(f"[{len(recs)}/{N}] " + ("ERR " + rec["error"] if "error" in rec else
                  f"hit={rec['hit']!s:5} ev={rec['evidence_hit']!s:5} J={rec['judged_hit']!s:5} Jev={rec['judged_evidence_hit']!s:5} {rec['status']:9} "
                  f"{rec['evidence_tokens']:4}tok {rec['latency_ms']:6}ms | {rec['q'][:50]} | {rec['gold'][:25]}"), flush=True)
        await asyncio.gather(*[run(i, r) for i, r in enumerate(sample)])
    ok = [r for r in recs if "error" not in r]
    lat = sorted(r["latency_ms"] for r in ok)
    n = len(recs)
    summary = {
        "n": n, "errors": n - len(ok), "offset": OFFSET, "budget": BUDGET, "judge": JUDGE_MODEL,
        "hit": f"{sum(r['hit'] for r in ok)}/{n}",
        "evidence_hit": f"{sum(r['evidence_hit'] for r in ok)}/{n}",
        "judged_hit": f"{sum(r['judged_hit'] for r in ok)}/{n}",
        "judged_evidence_hit": f"{sum(r['judged_evidence_hit'] for r in ok)}/{n}",
        "top1": f"{sum(r['rank'] == 0 for r in ok)}/{n}",
        "avg_evidence_tokens": round(sum(r["evidence_tokens"] for r in ok) / max(1, len(ok))),
        "p50_ms": lat[len(lat) // 2] if lat else None, "p90_ms": lat[int(len(lat) * 0.9)] if lat else None,
        "early_exit": sum(r["early_exit"] for r in ok),
        "status": {s: sum(r["status"] == s for r in ok) for s in ("found", "weak", "not_found")},
        # calibration: is "found" actually right?
        "found_precision": f"{sum(r['evidence_hit'] for r in ok if r['status'] == 'found')}/"
                           f"{sum(r['status'] == 'found' for r in ok)}",
    }
    json.dump({"summary": summary, "records": recs}, open(OUT, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    asyncio.run(main())

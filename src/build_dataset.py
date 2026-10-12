"""Build the (LLM_name, LLM_input, LLM_output) dataset from the public M4 benchmark.

Source: M4 (Wang et al., EACL 2024), https://github.com/mbzuai-nlp/M4  (data/*.jsonl)
We use the 5 English domains x 5 generators that have full coverage.

Cleaning decisions (documented in the report):
  * BLOOMz files store `machine_text` = prompt + completion (the prompt is echoed).
    We use `machine_abstract` / `machine_answer` (completion only) as the response and
    recover the filled prompt by stripping the completion off `machine_text`.
  * PeerRead files hold 4 (prompt, review) pairs per paper; we explode them.
  * Leading/trailing whitespace is stripped from every response: it is an artifact of
    the API client (e.g. Cohere responses start with "\n\n"), not of the model's writing.
  * Topic-controlled sampling: for every domain we prefer topics (same human source
    document) that were answered by all 5 models, so topic words cannot reveal the model.
  * Train/val/test split is grouped by topic: a topic never appears in two splits.

Usage:  python src/build_dataset.py --m4_dir path/to/M4/data --per_cell 600
"""
import argparse, json, os, re, random
from collections import defaultdict
import pandas as pd

DOMAINS = ["wikipedia", "wikihow", "reddit", "arxiv", "peerread"]
MODELS = {  # label -> (family, file-name suffix per domain)
    "ChatGPT": ("OpenAI GPT", {"wikipedia": "chatgpt", "wikihow": "chatGPT", "reddit": "chatGPT", "arxiv": "chatGPT", "peerread": "chatgpt"}),
    "GPT-3 davinci": ("OpenAI GPT", {d: "davinci" for d in DOMAINS}),
    "Cohere": ("Cohere Command", {d: "cohere" for d in DOMAINS}),
    "Dolly-v2": ("Databricks Dolly (Pythia)", {"wikipedia": "dolly", "wikihow": "dolly2", "reddit": "dolly", "arxiv": "dolly", "peerread": "dolly"}),
    "BLOOMz": ("BigScience BLOOM", {d: "bloomz" for d in DOMAINS}),
}


def norm_key(s, n=80):
    s = re.sub(r"[^a-z0-9]", "", str(s).lower())
    return s[:n]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def extract(domain, model, rows):
    """Yield dicts with topic, prompt, response for one (domain, model) file."""
    out = []
    for r in rows:
        if domain == "peerread":
            if model == "BLOOMz":
                topic = norm_key(r["title"])
                prompts = [p.replace("{title}", r["title"]).replace("{abstract}", r["abstract"]) for p in r["prompts"]]
                resps = r["bloom_reviews"]
            else:
                topic = None  # filled from machine/human text below
                prompts, resps = r["prompt"], r["machine_text"]
                # chatgpt/davinci/cohere/dolly: titles appear in human data; use source id mapping
                topic = str(r.get("source_ID", r.get("source_id")))
            for i, (p, m) in enumerate(zip(prompts, resps)):
                out.append({"topic": topic, "prompt": p, "response": m, "variant": i})
            continue
        if model == "BLOOMz":
            ans_key = "machine_answer" if domain == "reddit" else "machine_abstract"
            ans = r.get(ans_key)
            if not isinstance(ans, str):
                continue
            mt = r["machine_text"]
            if mt.endswith(ans) and len(mt) > len(ans):
                prompt = mt[: len(mt) - len(ans)]
            else:  # fall back to filling the template
                prompt = r["prompt"]
                for k in ("title", "headline"):
                    if k in r:
                        prompt = prompt.replace("{" + k + "}", str(r[k]))
            human = r.get("text") if domain != "arxiv" else r.get("abstract")
            out.append({"topic": norm_key(human), "prompt": prompt, "response": ans})
        else:
            human = r.get("human_text")
            out.append({"topic": norm_key(human), "prompt": r["prompt"], "response": r["machine_text"]})
    return out


def peerread_topic_map(m4_dir):
    """PeerRead: map each generator's source id -> normalized paper title (via BLOOMz file,
    which stores id + title), so topics are comparable across models."""
    b = load_jsonl(os.path.join(m4_dir, "peerread_bloomz.jsonl"))
    return {str(r["id"]): norm_key(r["title"]) for r in b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m4_dir", required=True)
    ap.add_argument("--per_cell", type=int, default=600, help="examples per (model, domain)")
    ap.add_argument("--out", default="data/dataset.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pr_map = peerread_topic_map(args.m4_dir)

    pools = {}  # (domain, model) -> list of records
    for d in DOMAINS:
        for m, (fam, suf) in MODELS.items():
            rows = load_jsonl(os.path.join(args.m4_dir, f"{d}_{suf[d]}.jsonl"))
            recs = extract(d, m, rows)
            if d == "peerread" and m != "BLOOMz":
                for x in recs:
                    x["topic"] = pr_map.get(x["topic"], "id" + x["topic"])
            clean = []
            for x in recs:
                if not isinstance(x["response"], str) or not isinstance(x["prompt"], str):
                    continue
                resp = x["response"].strip()
                if len(resp.split()) < 10:  # drop empty / degenerate generations
                    continue
                x["response"] = resp
                x["prompt"] = x["prompt"].strip()
                clean.append(x)
            pools[(d, m)] = clean
            print(f"{d:10s} {m:14s} raw={len(rows):5d} usable={len(clean):5d}")

    records = []
    for d in DOMAINS:
        topic_sets = [set(x["topic"] for x in pools[(d, m)]) for m in MODELS]
        shared = set.intersection(*topic_sets)
        print(f"[{d}] topics shared by all 5 models: {len(shared)}")
        shared = sorted(shared)
        rng.shuffle(shared)
        for m in MODELS:
            pool = pools[(d, m)]
            rng.shuffle(pool)
            by_topic = defaultdict(list)
            for x in pool:
                by_topic[x["topic"]].append(x)
            chosen, used = [], set()
            # 1) one example per shared topic (topic-controlled)
            for t in shared:
                if len(chosen) >= args.per_cell:
                    break
                chosen.append(by_topic[t][0]); used.add(id(by_topic[t][0]))
            # 2) top up with remaining examples if the shared pool is too small
            for x in pool:
                if len(chosen) >= args.per_cell:
                    break
                if id(x) not in used:
                    chosen.append(x); used.add(id(x))
            for x in chosen:
                records.append({"model": m, "family": MODELS[m][0], "domain": d, "topic": d + ":" + x["topic"],
                                "prompt": x["prompt"], "response": x["response"]})

    df = pd.DataFrame(records)
    # grouped split by topic: 70/15/15
    topics = sorted(df["topic"].unique())
    rng.shuffle(topics)
    n = len(topics)
    split_of = {t: ("train" if i < 0.70 * n else "val" if i < 0.85 * n else "test") for i, t in enumerate(topics)}
    df["split"] = df["topic"].map(split_of)
    df.insert(0, "id", range(len(df)))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby(["domain", "model"]).size().unstack())
    print(df["split"].value_counts())
    print("saved", args.out, len(df))


if __name__ == "__main__":
    main()

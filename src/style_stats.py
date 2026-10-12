"""RQ4: hand-crafted stylometric features per response.
 1) per-model averages (table + figures)
 2) a classifier that sees ONLY these ~20 numbers (how far does pure style get us?)
 3) feature importance (random forest, permutation importance on test)
Also writes the prompt-template table used for the RQ2 discussion.
Usage: python src/style_stats.py
"""
import json, os, re, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
LABELS = ["ChatGPT", "GPT-3 davinci", "Cohere", "Dolly-v2", "BLOOMz"]

WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")


def feats(t):
    words = WORD.findall(t)
    nw = max(len(words), 1)
    lines = [l for l in t.split("\n")]
    nonempty = [l for l in lines if l.strip()]
    sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    first100 = [w.lower() for w in words[:100]]
    return {
        "n_words": len(words),
        "n_chars": len(t),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "ttr_first100": len(set(first100)) / max(len(first100), 1),  # length-controlled lexical diversity
        "ttr_all": len(set(w.lower() for w in words)) / nw,
        "avg_sent_len": nw / max(len(sents), 1),
        "newlines_per_100w": 100 * t.count("\n") / nw,
        "blank_lines_per_100w": 100 * t.count("\n\n") / nw,
        "list_lines_frac": sum(bool(re.match(r"\s*([-*•]|\d+[.)])\s", l)) for l in nonempty) / max(len(nonempty), 1),
        "markdown_per_100w": 100 * len(re.findall(r"[#*_`|]", t)) / nw,
        "commas_per_100w": 100 * t.count(",") / nw,
        "periods_per_100w": 100 * t.count(".") / nw,
        "colons_per_100w": 100 * (t.count(":") + t.count(";")) / nw,
        "parens_per_100w": 100 * t.count("(") / nw,
        "quotes_per_100w": 100 * (t.count('"') + t.count("“")) / nw,
        "qmark_excl_per_100w": 100 * (t.count("?") + t.count("!")) / nw,
        "digits_per_100w": 100 * len(re.findall(r"\d", t)) / nw,
        "upper_ratio": sum(c.isupper() for c in t) / max(sum(c.isalpha() for c in t), 1),
        "first_person_per_100w": 100 * sum(w.lower() in ("i", "me", "my", "we", "our") for w in words) / nw,
        "ends_with_punct": float(t.rstrip()[-1:] in ".!?\"')") if t.strip() else 0.0,
    }


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "dataset.csv"), keep_default_na=False)
    F = pd.DataFrame([feats(r) for r in df["response"]])
    F["model"], F["domain"], F["split"] = df["model"], df["domain"], df["split"]
    F.to_csv(os.path.join(ROOT, "results", "style_features.csv"), index=False)
    cols = [c for c in F.columns if c not in ("model", "domain", "split")]
    means = F.groupby("model")[cols].mean().loc[LABELS]
    means.round(3).to_csv(os.path.join(ROOT, "results", "style_means_by_model.csv"))
    print(means.round(2).T)

    tr, te = F[F.split != "test"], F[F.split == "test"]
    out = {}
    for name, clf in [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
                      ("random_forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0))]:
        clf.fit(tr[cols], tr["model"])
        p = clf.predict(te[cols])
        out[name] = {"accuracy": float(accuracy_score(te["model"], p)), "macro_f1": float(f1_score(te["model"], p, average="macro"))}
        print(name, out[name])
        if name == "random_forest":
            pi = permutation_importance(clf, te[cols], te["model"], n_repeats=5, random_state=0, n_jobs=-1)
            out["permutation_importance"] = dict(sorted(zip(cols, pi.importances_mean.round(4).tolist()), key=lambda kv: -kv[1]))
    json.dump(out, open(os.path.join(ROOT, "results", "rq4_style_classifier.json"), "w"), indent=1)
    print(json.dumps(out["permutation_importance"], indent=1))

    # prompt templates per (domain, model): first 60 chars of the prompt, and how many distinct ones
    rows = []
    for (d, m), g in df.groupby(["domain", "model"]):
        heads = g["prompt"].str.replace(r"\s+", " ", regex=True).str[:60]
        rows.append({"domain": d, "model": m, "distinct_prompt_openings": heads.nunique(),
                     "most_common_opening": heads.value_counts().index[0]})
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "prompt_templates.csv"), index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()

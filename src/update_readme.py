"""Fill the results table in README.md from results/*.json (run after make_figures.py)."""
import json, os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load(n):
    p = os.path.join(ROOT, "results", n + ".json")
    return json.load(open(p))["test"] if os.path.exists(p) else None


def cell(r):
    return f"{r['accuracy']:.3f} / {r['macro_f1']:.3f}" if r else "-"


rows = ["## Headline results (test accuracy / macro-F1, chance = 0.20)", "",
        "| Classifier | Prompt only | Response only (RQ1) | Prompt + response |", "|---|---|---|---|"]
for disp, key in [("CNN", "cnn"), ("BiLSTM", "lstm"), ("TF-IDF + LR", "tfidf")]:
    rows.append(f"| {disp} | " + " | ".join(cell(load(f"rq2_{key}_{m}")) for m in ["input", "output", "both"]) + " |")
rows += ["", "RQ3 (CNN, accuracy on a never-seen domain): " + ", ".join(
    f"{d} {load(f'rq3_cnn_out_{d}')['accuracy']:.2f}" for d in ["wikipedia", "wikihow", "reddit", "arxiv", "peerread"] if load(f"rq3_cnn_out_{d}")),
    "", "RQ4 (CNN macro-F1 after ablation): " + ", ".join(
    f"{a} {load(f'rq4_cnn_{a}')['macro_f1']:.2f}" for a in ["no_format", "lower", "no_punct", "style_only", "first50"] if load(f"rq4_cnn_{a}")),
    "", "Full numbers, figures and analysis are in the report."]
p = os.path.join(ROOT, "README.md")
s = open(p).read()
start = s.find("## Headline results")
if start >= 0:  # replace an existing table
    end = s.find("## Reproduce")
    s = s[:start] + "RESULTS_TABLE\n\n" + s[end:]
s = s.replace("RESULTS_TABLE", "\n".join(rows))
open(p, "w").write(s)
print("\n".join(rows))

"""Build every figure and summary table for the report from results/*.json.
Usage: python src/make_figures.py
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
LABELS = ["ChatGPT", "GPT-3 davinci", "Cohere", "Dolly-v2", "BLOOMz"]
SHORT = ["ChatGPT", "davinci", "Cohere", "Dolly", "BLOOMz"]
DOMAINS = ["wikipedia", "wikihow", "reddit", "arxiv", "peerread"]
# categorical slots (fixed order) from the validated reference palette
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
METHOD_COLOR = {"CNN": C[0], "BiLSTM": C[1], "TF-IDF+LR": C[2]}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False})


def load(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def conf_plot(ax, cm, title):
    cm = np.array(cm, dtype=float)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(SHORT, rotation=35, ha="right"); ax.set_yticklabels(SHORT)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title); ax.grid(False)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{norm[i, j]*100:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else INK)


def bars(ax, groups, series, values, ylabel, ylim=(0, 1), fmt="{:.2f}"):
    """grouped bars: values[s][g]; 2px gap between adjacent bars."""
    n = len(series); w = 0.8 / n; x = np.arange(len(groups))
    for k, s in enumerate(series):
        v = values[s]
        b = ax.bar(x + (k - (n - 1) / 2) * w, v, w * 0.92, label=s, color=METHOD_COLOR.get(s, C[k % 5]), linewidth=0)
        for xi, vi in zip(x + (k - (n - 1) / 2) * w, v):
            if vi is not None and not np.isnan(vi):
                ax.text(xi, vi + 0.01, fmt.format(vi), ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel(ylabel); ax.set_ylim(*ylim); ax.grid(axis="x", visible=False)
    ax.set_yticks(np.arange(0, 1.01, 0.2))


def main():
    rows = []
    # ---------- RQ1 + RQ2 ----------
    methods = {"CNN": "cnn", "BiLSTM": "lstm", "TF-IDF+LR": "tfidf"}
    modes = ["input", "output", "both"]
    vals = {}
    for disp, key in methods.items():
        vals[disp] = []
        for m in modes:
            r = load(f"rq2_{key}_{m}")
            vals[disp].append(r["test"]["macro_f1"] if r else np.nan)
            if r:
                rows.append({"experiment": "RQ1/RQ2", "method": disp, "input": m, "accuracy": r["test"]["accuracy"],
                             "macro_f1": r["test"]["macro_f1"], "epochs": r.get("epochs_run"), "train_time_s": r.get("train_time_s")})
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    bars(ax, ["Input only\n(prompt)", "Output only\n(response)", "Input + Output"], list(methods), vals, "Test macro-F1")
    ax.set_ylim(0, 1.18)
    ax.axhline(0.2, color=INK2, lw=1, ls="--", label="chance (0.20)")
    ax.legend(ncol=4, loc="upper left", fontsize=7.5); ax.set_title("RQ2: which input identifies the LLM?")
    save(fig, "rq2_inputs")

    r_c, r_l = load("rq2_cnn_output"), load("rq2_lstm_output")
    if r_c and r_l:
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
        conf_plot(axs[0], r_c["test"]["confusion"], f"CNN (acc {r_c['test']['accuracy']:.3f})")
        conf_plot(axs[1], r_l["test"]["confusion"], f"BiLSTM (acc {r_l['test']['accuracy']:.3f})")
        fig.suptitle("RQ1: confusion matrices, response only (row %)", fontsize=10, fontweight="bold", color=INK)
        fig.tight_layout(); save(fig, "rq1_confusion")
        # training curves
        fig, ax = plt.subplots(figsize=(5.0, 2.6))
        for r, disp in [(r_c, "CNN"), (r_l, "BiLSTM")]:
            h = r["history"]
            ax.plot([e["epoch"] for e in h], [e["val_macro_f1"] for e in h], marker="o", ms=4, lw=2, color=METHOD_COLOR[disp], label=disp)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Validation macro-F1"); ax.legend(); ax.set_title("Training curves (response only)")
        save(fig, "training_curves")
    r_ci, r_li = load("rq2_cnn_input"), load("rq2_lstm_input")
    if r_ci:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        conf_plot(ax, r_ci["test"]["confusion"], f"CNN, prompt only (acc {r_ci['test']['accuracy']:.3f})")
        save(fig, "rq2_input_confusion")

    # per-domain accuracy table for RQ2
    pd_rows = []
    for disp, key in methods.items():
        for m in modes:
            r = load(f"rq2_{key}_{m}")
            if r and "per_domain_acc" in r["test"]:
                pd_rows.append({"method": disp, "input": m, **r["test"]["per_domain_acc"]})
    pd.DataFrame(pd_rows).to_csv(os.path.join(RES, "summary_per_domain.csv"), index=False)

    # seeds (variance)
    seed_rows = []
    for key, disp in [("cnn", "CNN"), ("lstm", "BiLSTM")]:
        accs = []
        for name in [f"rq2_{key}_output", f"rq1_{key}_output_seed1", f"rq1_{key}_output_seed2"]:
            r = load(name)
            if r: accs.append((r["test"]["accuracy"], r["test"]["macro_f1"]))
        if accs:
            a = np.array(accs)
            seed_rows.append({"method": disp, "n_seeds": len(a), "acc_mean": a[:, 0].mean(), "acc_std": a[:, 0].std(),
                              "f1_mean": a[:, 1].mean(), "f1_std": a[:, 1].std()})
    pd.DataFrame(seed_rows).to_csv(os.path.join(RES, "summary_seeds.csv"), index=False)

    # ---------- RQ3 ----------
    rq3 = {}
    for disp, key in methods.items():
        ind = load(f"rq2_{key}_output")
        rq3[disp] = {"in": [], "out": []}
        for d in DOMAINS:
            r = load(f"rq3_{key}_out_{d}")
            rq3[disp]["in"].append(ind["test"]["per_domain_acc"][d] if ind else np.nan)
            rq3[disp]["out"].append(r["test"]["accuracy"] if r else np.nan)
            if r:
                rows.append({"experiment": "RQ3", "method": disp, "input": f"held-out {d}", "accuracy": r["test"]["accuracy"],
                             "macro_f1": r["test"]["macro_f1"], "epochs": r.get("epochs_run"), "train_time_s": r.get("train_time_s")})
    if any(not np.all(np.isnan(rq3[k]["out"])) for k in rq3):
        fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
        for ax, disp in zip(axs, ["CNN", "TF-IDF+LR"]):
            vals3 = {"Seen domain (in-domain test)": rq3[disp]["in"], "Unseen domain (held out)": rq3[disp]["out"]}
            n = 2; w = 0.38; x = np.arange(5)
            for k, (s, v) in enumerate(vals3.items()):
                ax.bar(x + (k - 0.5) * w, v, w * 0.92, color=[C[0], C[1]][k], label=s, linewidth=0)
                for xi, vi in zip(x + (k - 0.5) * w, v):
                    if not np.isnan(vi): ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=6.5, color=INK2)
            ax.set_xticks(x); ax.set_xticklabels(DOMAINS, rotation=20); ax.set_ylim(0, 1.08); ax.set_title(disp); ax.grid(axis="x", visible=False)
        axs[0].set_ylabel("Test accuracy")
        h, l = axs[0].get_legend_handles_labels()
        fig.suptitle("RQ3: accuracy on a domain seen vs. never seen in training", fontsize=10, fontweight="bold", color=INK)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
        save(fig, "rq3_domains")
        # per-model recall on held-out domains (CNN)
        rec = {}
        for key in ["cnn", "tfidf"]:
            for d in DOMAINS:
                r = load(f"rq3_{key}_out_{d}")
                if r:
                    rec[(key, d)] = {LABELS[i] if False else m: r["test"]["per_class"][m]["recall"] for i, m in enumerate(LABELS)}
        if rec:
            t = pd.DataFrame(rec).T
            t.to_csv(os.path.join(RES, "rq3_per_model_recall.csv"))

    # ---------- RQ4 ablations ----------
    abls = ["none", "no_format", "lower", "no_punct", "style_only", "first50"]
    abl_names = ["Original", "No\nformatting", "Lower-case", "No punct.\n+ no format", "Style only\n(content\nmasked)", "First 50\nwords"]
    vals4 = {}
    for disp, key in methods.items():
        vals4[disp] = []
        for a in abls:
            r = load(f"rq2_{key}_output") if a == "none" else load(f"rq4_{key}_{a}")
            vals4[disp].append(r["test"]["macro_f1"] if r else np.nan)
            if r and a != "none":
                rows.append({"experiment": "RQ4", "method": disp, "input": f"ablation {a}", "accuracy": r["test"]["accuracy"],
                             "macro_f1": r["test"]["macro_f1"], "epochs": r.get("epochs_run"), "train_time_s": r.get("train_time_s")})
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    bars(ax, abl_names, list(methods), vals4, "Test macro-F1", (0, 1.2))
    ax.legend(ncol=3, loc="upper center", fontsize=8); ax.set_title("RQ4: which signals does the classifier need? (response only)")
    save(fig, "rq4_ablations")

    # style features small multiples
    sf = os.path.join(RES, "style_features.csv")
    if os.path.exists(sf):
        F = pd.read_csv(sf)
        feats = [("n_words", "Words per response"), ("ttr_first100", "Lexical diversity\n(TTR, first 100 words)"),
                 ("newlines_per_100w", "Line breaks\nper 100 words"), ("avg_sent_len", "Words per sentence"),
                 ("commas_per_100w", "Commas per 100 words"), ("ends_with_punct", "Share of responses ending\nwith . ! ? or quote")]
        fig, axs = plt.subplots(2, 3, figsize=(7.6, 4.4))
        for ax, (f, t) in zip(axs.flat, feats):
            mean = np.array([F.loc[F.model == m, f].mean() for m in LABELS])
            ci = np.array([1.96 * F.loc[F.model == m, f].std() / np.sqrt((F.model == m).sum()) for m in LABELS])
            ax.bar(range(5), mean, 0.7, color=C, linewidth=0)
            ax.errorbar(range(5), mean, yerr=ci, fmt="none", ecolor=INK, lw=1, capsize=2)
            ax.set_xticks(range(5)); ax.set_xticklabels(SHORT, rotation=35, ha="right", fontsize=7)
            ax.set_title(t, fontsize=8.5); ax.grid(axis="x", visible=False)
        fig.suptitle("RQ4: stylometric profile per model (bar = mean, whisker = 95% CI)", fontsize=10, fontweight="bold", color=INK)
        fig.tight_layout(); save(fig, "rq4_style")

    # ---------- extra analyses: family-level accuracy, accuracy vs response length ----------
    df = pd.read_csv(os.path.join(ROOT, "data", "dataset.csv"), keep_default_na=False)
    te = df[df.split == "test"].reset_index(drop=True)
    fam = np.array([0, 0, 1, 2, 3])  # ChatGPT & davinci -> OpenAI family
    extra = {}
    lens = te["response"].str.split().str.len().values
    edges = [0, 100, 200, 300, 400, 10**6]; names = ["<100", "100-199", "200-299", "300-399", "400+"]
    fig, ax = plt.subplots(figsize=(5.2, 2.7)); plotted = False
    for key, disp in [("cnn", "CNN"), ("lstm", "BiLSTM")]:
        pth = os.path.join(RES, f"rq2_{key}_output_preds.npy")
        if not os.path.exists(pth):
            continue
        y, p = np.load(pth)
        # predictions were saved in test_df order, which is te order
        extra[disp] = {"model_acc": float((y == p).mean()), "family_acc": float((fam[y] == fam[p]).mean()),
                       "openai_internal_confusions": int(((y == 0) & (p == 1)).sum() + ((y == 1) & (p == 0)).sum()),
                       "openai_total_errors_on_openai_rows": int(((y <= 1) & (y != p)).sum())}
        accs, ns = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (lens >= lo) & (lens < hi); accs.append(float((y[m] == p[m]).mean())); ns.append(int(m.sum()))
        extra[disp]["acc_by_length"] = dict(zip(names, accs)); extra[disp]["n_by_length"] = dict(zip(names, ns))
        ax.plot(names, accs, marker="o", ms=5, lw=2, color=METHOD_COLOR[disp], label=disp); plotted = True
    if plotted:
        ax.set_xlabel("Response length (words)"); ax.set_ylabel("Test accuracy"); ax.set_ylim(0, 1); ax.legend()
        ax.set_title("Accuracy vs. response length (response only)"); save(fig, "acc_by_length")
    else:
        plt.close(fig)
    # RQ2: best accuracy ANY prompt-only classifier could reach on this test set (Bayes bound):
    # group test examples by their exact prompt string and predict the majority model of each group.
    def bound(g):
        k = g["prompt"].str.replace(r"\s+", " ", regex=True).str.strip()
        return float(g.assign(k=k).groupby("k")["model"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(g))
    extra["prompt_only_bound"] = {"overall": bound(te), **{d: bound(te[te.domain == d]) for d in DOMAINS}}
    ci, ti = load("rq2_cnn_input"), load("rq2_tfidf_input")
    if ci:
        fig, ax = plt.subplots(figsize=(6.2, 2.7))
        series = {"Best possible from prompt alone": [extra["prompt_only_bound"][d] for d in DOMAINS],
                  "CNN, prompt only": [ci["test"]["per_domain_acc"][d] for d in DOMAINS]}
        x = np.arange(5); w = 0.38
        for k, (nm, v) in enumerate(series.items()):
            ax.bar(x + (k - 0.5) * w, v, w * 0.92, color=[INK2, C[0]][k] if k == 0 else C[0], label=nm, linewidth=0, alpha=0.45 if k == 0 else 1)
            for xi, vi in zip(x + (k - 0.5) * w, v):
                ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=6.5, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels(DOMAINS); ax.set_ylim(0, 1.22); ax.set_ylabel("Test accuracy"); ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", fontsize=7.5); ax.set_title("RQ2: prompt-only accuracy is capped by how many templates differ")
        save(fig, "rq2_prompt_bound")
    json.dump(extra, open(os.path.join(RES, "extra_analysis.json"), "w"), indent=1)
    print(json.dumps(extra, indent=1))

    pd.DataFrame(rows).to_csv(os.path.join(RES, "summary_all.csv"), index=False)
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()

"""Run all experiments.  Usage:
    python src/experiments.py --rq rq1_rq2   # CNN/LSTM/TF-IDF x {output, input, both}
    python src/experiments.py --rq rq1_seeds # CNN output-only with 3 seeds (variance)
    python src/experiments.py --rq rq3 --models cnn   # leave-one-domain-out (CNN + TF-IDF; see report)
    python src/experiments.py --rq rq4 --models cnn                      # ablations (CNN + TF-IDF)
    python src/experiments.py --rq rq4_one --models lstm --abl style_only  # one BiLSTM ablation
    python src/experiments.py --rq rq3_noformat --models cnn --domains wikipedia,wikihow  # RQ3 follow-up
"""
import argparse, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from train import run_nn, run_tfidf

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")


def splits(df):
    return df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq", required=True)
    ap.add_argument("--models", default="cnn,lstm")
    ap.add_argument("--abl", default="none")
    ap.add_argument("--domains", default="wikipedia,wikihow,reddit,arxiv,peerread")
    args = ap.parse_args()
    df = pd.read_csv(DATA, keep_default_na=False)
    tr, va, te = splits(df)
    models = args.models.split(",")
    do_tfidf = "cnn" in models or "tfidf" in models  # run the cheap baseline only once (in the CNN process)

    if args.rq == "rq1_rq2":
        for mode in ["output", "input", "both"]:
            if do_tfidf: run_tfidf(f"rq2_tfidf_{mode}", tr, va, te, mode)
        for m in models:
            for mode in ["output", "input", "both"]:
                run_nn(f"rq2_{m}_{mode}", m, tr, va, te, mode)

    elif args.rq == "rq1_seeds":  # extra seeds for the CNN only (BiLSTM is too slow on our CPU budget)
        for m in [m for m in models if m == "cnn"]:
            for s in [1, 2]:
                run_nn(f"rq1_{m}_output_seed{s}", m, tr, va, te, "output", seed=s)

    elif args.rq == "rq4_one":  # a single ablation, e.g. --models lstm --abl style_only (used to run jobs in parallel)
        run_nn(f"rq4_{models[0]}_{args.abl}", models[0], tr, va, te, "output", args.abl)

    elif args.rq == "rq3":
        # leave-one-domain-out: train/val on 4 domains, test on ALL examples of the unseen domain
        for held in ["wikipedia", "wikihow", "reddit", "arxiv", "peerread"]:
            trd, vad = tr[tr.domain != held], va[va.domain != held]
            ted = df[df.domain == held]
            if do_tfidf: run_tfidf(f"rq3_tfidf_out_{held}", trd, vad, ted, "output")
            for m in models:  # BiLSTM capped at 6 epochs here to fit the CPU budget
                run_nn(f"rq3_{m}_out_{held}", m, trd, vad, ted, "output", max_epochs=6 if m == "lstm" else None)

    elif args.rq == "rq3_noformat":
        # follow-up to RQ3: same leave-one-domain-out folds, but with layout removed from every response
        # (tests whether the cross-domain failure is caused by the line-break shortcut)
        for held in args.domains.split(","):
            trd, vad = tr[tr.domain != held], va[va.domain != held]
            ted = df[df.domain == held]
            if do_tfidf: run_tfidf(f"rq3nf_tfidf_out_{held}", trd, vad, ted, "output", "no_format")
            for m in models:
                if m != "tfidf":
                    run_nn(f"rq3nf_{m}_out_{held}", m, trd, vad, ted, "output", "no_format")

    elif args.rq == "rq4":
        for abl in ["no_format", "no_punct", "lower", "style_only", "first50"]:
            if do_tfidf: run_tfidf(f"rq4_tfidf_{abl}", tr, va, te, "output", abl)
            for m in models:
                if m == "lstm" and abl not in ("no_format", "style_only"):
                    continue  # BiLSTM: only the two most informative ablations (CPU budget)
                run_nn(f"rq4_{m}_{abl}", m, tr, va, te, "output", abl)
    else:
        raise ValueError(args.rq)


if __name__ == "__main__":
    main()

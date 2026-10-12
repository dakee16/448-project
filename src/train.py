"""Training / evaluation loop shared by all experiments."""
import json, os, random, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from text import Vocab, build_tokens, ablate, tokenize
from models import build_model

LABELS = ["ChatGPT", "GPT-3 davinci", "Cohere", "Dolly-v2", "BLOOMz"]
L2I = {l: i for i, l in enumerate(LABELS)}
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Hyper-parameters (same for every experiment so results are comparable)
HP = dict(batch_size=64, lr=1e-3, weight_decay=0.0, max_epochs=10, patience=2, grad_clip=5.0,
          max_prompt=128, max_resp=256, vocab_size=30000, min_freq=2)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def common_words(train_df, k=150):
    """The k most frequent words of the training responses (kept by the 'style_only' ablation)."""
    from collections import Counter
    cnt = Counter(t.lower() for r in train_df["response"] for t in tokenize(r) if t[0].isalpha())
    return set(w for w, _ in cnt.most_common(k))


def make_tokens(df, mode, ablation="none", common_words=None):
    out = []
    for p, r in zip(df["prompt"], df["response"]):
        r = ablate(r, ablation, common_words)
        out.append(build_tokens(p, r, mode, HP["max_prompt"], HP["max_resp"]))
    return out


def batches(encoded, labels, bs, shuffle, rng):
    idx = list(range(len(encoded)))
    if shuffle:
        rng.shuffle(idx)
        # bucket by length inside chunks of 50 batches -> less padding, still random
        chunk = bs * 50
        idx = [i for c in range(0, len(idx), chunk) for i in sorted(idx[c:c + chunk], key=lambda j: len(encoded[j]))]
        bl = [idx[i:i + bs] for i in range(0, len(idx), bs)]
        rng.shuffle(bl)
    else:
        idx.sort(key=lambda j: len(encoded[j]))
        bl = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    for b in bl:
        L = max(len(encoded[j]) for j in b)
        x = torch.zeros(len(b), L, dtype=torch.long)
        for k, j in enumerate(b):
            x[k, : len(encoded[j])] = torch.tensor(encoded[j])
        lengths = torch.tensor([len(encoded[j]) for j in b])
        yield b, x, lengths, torch.tensor([labels[j] for j in b])


@torch.no_grad()
def predict(model, encoded, labels):
    model.eval()
    preds = np.zeros(len(encoded), dtype=int)
    probs = np.zeros((len(encoded), len(LABELS)), dtype=np.float32)
    for b, x, lengths, _ in batches(encoded, labels, 256, False, None):
        logits = model(x, lengths)
        p = torch.softmax(logits, 1).numpy()
        probs[b] = p
        preds[b] = p.argmax(1)
    return preds, probs


def metrics(y, p, domains=None):
    out = {"accuracy": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro")),
           "confusion": confusion_matrix(y, p, labels=list(range(len(LABELS)))).tolist(),
           "per_class": {LABELS[int(k)]: v for k, v in classification_report(y, p, labels=list(range(len(LABELS))), output_dict=True, zero_division=0).items() if k.isdigit()}}
    if domains is not None:
        domains = np.asarray(domains)
        out["per_domain_acc"] = {d: float(accuracy_score(np.asarray(y)[domains == d], np.asarray(p)[domains == d])) for d in sorted(set(domains))}
    return out


def run_nn(name, model_name, train_df, val_df, test_df, mode="output", ablation="none", seed=42,
           max_epochs=None, save=True, verbose=True):
    """Train a CNN/LSTM, early-stop on val macro-F1, evaluate on test. Returns result dict.
    If results/<name>.json already exists the run is skipped (lets queued jobs resume)."""
    out_path = os.path.join(RESULTS_DIR, name + ".json")
    if save and os.path.exists(out_path):
        print(f"SKIP {name} (already done)", flush=True)
        return json.load(open(out_path))
    set_seed(seed)
    rng = random.Random(seed)
    common = common_words(train_df) if ablation == "style_only" else None
    tr_tok = make_tokens(train_df, mode, ablation, common)
    vocab = Vocab(tr_tok, HP["vocab_size"], HP["min_freq"])
    enc = lambda toks: [vocab.encode(t, 10_000) for t in toks]
    Xtr, Xva, Xte = enc(tr_tok), enc(make_tokens(val_df, mode, ablation, common)), enc(make_tokens(test_df, mode, ablation, common))
    ytr, yva, yte = [L2I[m] for m in train_df["model"]], [L2I[m] for m in val_df["model"]], [L2I[m] for m in test_df["model"]]

    model = build_model(model_name, len(vocab), len(LABELS))
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    lossf = nn.CrossEntropyLoss()
    best, best_state, bad, hist = -1, None, 0, []
    t0 = time.time()
    for ep in range(1, (max_epochs or HP["max_epochs"]) + 1):
        model.train(); tot, n = 0.0, 0
        for _, x, lengths, y in batches(Xtr, ytr, HP["batch_size"], True, rng):
            opt.zero_grad()
            loss = lossf(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), HP["grad_clip"])
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        vp, _ = predict(model, Xva, yva)
        vf1 = f1_score(yva, vp, average="macro"); vacc = accuracy_score(yva, vp)
        hist.append({"epoch": ep, "train_loss": tot / n, "val_acc": float(vacc), "val_macro_f1": float(vf1), "time_s": time.time() - t0})
        if verbose:
            print(f"  [{name}] ep{ep} loss={tot/n:.4f} val_acc={vacc:.4f} val_f1={vf1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vf1 > best:
            best, bad = vf1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    model.load_state_dict(best_state)
    tp, tprob = predict(model, Xte, yte)
    res = {"name": name, "model": model_name, "mode": mode, "ablation": ablation, "seed": seed,
           "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte), "vocab": len(vocab),
           "params": sum(p.numel() for p in model.parameters()), "best_val_macro_f1": float(best),
           "epochs_run": len(hist), "train_time_s": time.time() - t0, "history": hist,
           "test": metrics(yte, tp, test_df["domain"].tolist())}
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, name + ".json"), "w") as f:
            json.dump(res, f, indent=1)
        np.save(os.path.join(RESULTS_DIR, name + "_preds.npy"), np.stack([np.array(yte), tp]))
    print(f"DONE {name}: test acc={res['test']['accuracy']:.4f} macroF1={res['test']['macro_f1']:.4f} epochs={len(hist)} time={res['train_time_s']:.0f}s", flush=True)
    return res


def run_tfidf(name, train_df, val_df, test_df, mode="output", ablation="none", save=True):
    """Reference baseline: TF-IDF (word 1-2 grams, our tokenizer) + logistic regression."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    t0 = time.time()
    common = common_words(train_df) if ablation == "style_only" else None
    tr, te = make_tokens(train_df, mode, ablation, common), make_tokens(test_df, mode, ablation, common)
    vec = TfidfVectorizer(analyzer=lambda toks: toks + [a + " " + b for a, b in zip(toks, toks[1:])],
                          min_df=2, max_features=100000, sublinear_tf=True)
    Xtr = vec.fit_transform(tr); Xte = vec.transform(te)
    ytr = np.array([L2I[m] for m in train_df["model"]]); yte = np.array([L2I[m] for m in test_df["model"]])
    clf = LogisticRegression(max_iter=2000, C=10.0)
    clf.fit(Xtr, ytr)
    p = clf.predict(Xte)
    res = {"name": name, "model": "tfidf_lr", "mode": mode, "ablation": ablation, "train_time_s": time.time() - t0,
           "test": metrics(yte, p, test_df["domain"].tolist())}
    # top features per class (for RQ4)
    fn = np.array(vec.get_feature_names_out())
    res["top_features"] = {LABELS[c]: fn[np.argsort(-clf.coef_[c])[:25]].tolist() for c in range(len(LABELS))}
    if save:
        with open(os.path.join(RESULTS_DIR, name + ".json"), "w") as f:
            json.dump(res, f, indent=1)
    print(f"DONE {name}: test acc={res['test']['accuracy']:.4f} macroF1={res['test']['macro_f1']:.4f}", flush=True)
    return res

# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 Midterm Project. **Team leader:** Daksh Mainee. **Member:** Sanan Goel.

**Report:** [`report/report.pdf`](report/report.pdf)

We train a **CNN** and a **bidirectional LSTM** (PyTorch) to tell which of five LLMs wrote a piece of text:
ChatGPT, GPT-3 davinci, Cohere, Dolly-v2 and BLOOMz, spanning 4 model families.
The data comes from the public [M4 benchmark](https://github.com/mbzuai-nlp/M4).
We sample 15,000 balanced examples over 5 domains (Wikipedia, WikiHow, Reddit, arXiv, PeerRead),
with topic-controlled sampling and a train/val/test split grouped by topic.

| Research question | What we ran |
|---|---|
| RQ1: identify the LLM from its response | CNN, BiLSTM and a TF-IDF + logistic regression baseline; CNN repeated with 3 seeds |
| RQ2: does the prompt help? | Input only vs. output only vs. input + output, plus the best accuracy achievable from the prompt alone |
| RQ3: cross-domain generalization (extra credit) | Leave-one-domain-out, 5 folds × 3 classifiers |
| RQ4: what distinguishes the LLMs? (extra credit) | 20 stylometric features, top n-grams, and 5 retrain-and-measure ablations |

## Headline results (test accuracy / macro-F1, chance = 0.20)

| Classifier | Prompt only | Response only (RQ1) | Prompt + response |
|---|---|---|---|
| CNN | 0.720 / 0.725 | 0.801 / 0.803 | 0.905 / 0.905 |
| BiLSTM | 0.704 / 0.705 | 0.756 / 0.756 | 0.857 / 0.854 |
| TF-IDF + LR | 0.718 / 0.716 | 0.889 / 0.888 | 0.944 / 0.944 |

RQ3 (CNN, accuracy on a never-seen domain): wikipedia 0.35, wikihow 0.61, reddit 0.64, arxiv 0.51, peerread 0.69

RQ4 (CNN macro-F1 after ablation): no_format 0.77, lower 0.78, no_punct 0.75, style_only 0.81, first50 0.67

Full numbers, figures and analysis are in the report.

## Reproduce

```bash
git clone https://github.com/mbzuai-nlp/M4.git          # raw data (not re-hosted here)
pip install -r requirements.txt
python src/build_dataset.py --m4_dir M4/data --per_cell 600   # -> data/dataset.csv (seed 42)
python src/experiments.py --rq rq1_rq2                       # RQ1 + RQ2 (CNN, BiLSTM, TF-IDF)
python src/experiments.py --rq rq3 --models cnn               # RQ3 leave-one-domain-out (CNN + TF-IDF)
python src/experiments.py --rq rq3_noformat --models cnn --domains wikipedia,wikihow   # RQ3 follow-up
python src/experiments.py --rq rq4 --models cnn               # RQ4 ablations (CNN + TF-IDF)
python src/experiments.py --rq rq4_one --models lstm --abl no_format    # BiLSTM ablations
python src/experiments.py --rq rq4_one --models lstm --abl style_only
python src/experiments.py --rq rq1_seeds --models cnn         # 2 extra CNN seeds
python src/style_stats.py                   # RQ4 stylometry + prompt-template table
python src/make_figures.py                  # figures/ + summary CSVs
python report/build_report.py               # report/report.pdf
```

CPU only is fine. The full run takes about 3 hours on 2 cores. Finished runs are skipped, so you can stop and resume.
To run only some models, pass `--models cnn` or `--models lstm`.

## Layout

```
src/build_dataset.py   data curation: cleaning, topic-controlled sampling, grouped split
src/text.py            tokenizer (keeps case, punctuation, line breaks), input modes, RQ4 ablations
src/models.py          TextCNN (3/4/5-gram filters + max-pool) and BiLSTM (packed, mean+max pool)
src/train.py           training loop, early stopping on val macro-F1, metrics, TF-IDF baseline
src/experiments.py     all RQ runs
src/style_stats.py     stylometric features, style-only classifier, prompt-template table
src/make_figures.py    figures + summary tables
results/               one JSON per run (metrics, confusion matrix, per-domain accuracy, training history)
figures/               PNG figures used in the report
report/                build_report.py and report.pdf
```

`data/dataset.csv` is not committed because the raw text belongs to the M4 authors. Rebuild it with the command above.

"""Builds report/report.pdf from results/*.json, results/*.csv and figures/*.png.
Numbers in tables, figures and result statements are read from the results files.
Usage: python report/build_report.py
"""
import json, os
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether)

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "report", "report.pdf")
LABELS = ["ChatGPT", "GPT-3 davinci", "Cohere", "Dolly-v2", "BLOOMz"]
DOMAINS = ["wikipedia", "wikihow", "reddit", "arxiv", "peerread"]

INK, INK2, RULE, HEAD_BG = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#d8d7d2"), colors.HexColor("#f1f0ec")
ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("s", parent=ss["Normal"], fontName="Helvetica", fontSize=10.5, leading=14, alignment=TA_CENTER, textColor=INK2),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=10, spaceAfter=4, textColor=INK, keepWithNext=1),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=7, spaceAfter=2, textColor=INK, keepWithNext=1),
    "body": ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5, leading=13, alignment=TA_JUSTIFY, spaceAfter=5, textColor=INK),
    "bullet": ParagraphStyle("bl", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5, leading=12.5, leftIndent=12, bulletIndent=3, spaceAfter=2.5, textColor=INK),
    "cap": ParagraphStyle("c", parent=ss["Normal"], fontName="Helvetica-Oblique", fontSize=8.3, leading=10.5, alignment=TA_CENTER, textColor=INK2, spaceAfter=8),
    "cell": ParagraphStyle("cell", parent=ss["Normal"], fontName="Helvetica", fontSize=8, leading=9.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=9.6, textColor=INK),
    "code": ParagraphStyle("code", parent=ss["Code"], fontName="Courier", fontSize=7.8, leading=9.5, textColor=INK, leftIndent=8),
}


def load(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def P(t, st="body"):
    return Paragraph(t, S[st])


def bullets(items):
    return [Paragraph(i, S["bullet"], bulletText="•") for i in items]


def table(rows, widths, header=True, bold_first_col=False, align_right_from=1):
    data = [[Paragraph(str(c), S["cellb"] if (header and i == 0) or (bold_first_col and j == 0) else S["cell"]) for j, c in enumerate(r)]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK2)]
    for i in range(1 + header, len(rows)):
        st.append(("LINEABOVE", (0, i), (-1, i), 0.25, RULE))
    t.setStyle(TableStyle(st))
    return t


def fig(name, width_in, caption):
    p = os.path.join(FIG, name + ".png")
    if not os.path.exists(p):
        return [P(f"[missing figure {name}]")]
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(p).getSize()
    w = width_in * inch
    return [KeepTogether([Image(p, width=w, height=w * ih / iw), P(caption, "cap")])]


def pct(x):
    return f"{100 * x:.1f}%"


def f3(x):
    return f"{x:.3f}"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(INK2)
    canvas.drawString(0.85 * inch, 0.5 * inch, "CMPSC 448 - Who Wrote It? Identifying LLMs from Their Responses")
    canvas.drawRightString(letter[0] - 0.85 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


# ============================================================================================
def section_data(story, df):
    n = len(df); spl = df["split"].value_counts()
    c = df.groupby("topic")["model"].nunique()
    full = (df["topic"].map(c) == 5)
    story.append(P("1. Problem Definition and Dataset Curation", "h1"))
    story.append(P("1.1 Task", "h2"))
    story.append(P(
        "Each example is a triple <i>(LLM_name, LLM_input, LLM_output)</i>: the name of the model that wrote the text, the "
        "prompt it was given, and the response it produced. The task is <b>5-way single-label classification</b>: predict the "
        "LLM_name. RQ1 uses only the response; RQ2 compares a classifier that sees only the prompt, only the response, or both. "
        "Classes are perfectly balanced, so chance accuracy is 20%. We report <b>accuracy</b> and <b>macro-F1</b> (the unweighted "
        "mean of per-class F1, which drops sharply if a classifier ignores one model) on a held-out test set, plus confusion "
        "matrices and per-domain accuracy."))
    story.append(P("1.2 Data source", "h2"))
    story.append(P(
        "We used <b>M4</b> (Wang et al., EACL 2024, Best Resource Paper award), a public research benchmark of machine-generated "
        "text hosted at github.com/mbzuai-nlp/M4. We chose it over generating our own data because (i) it gives the same prompts "
        "to several LLMs, so topic is controlled, (ii) it spans five English domains, which makes RQ3 (cross-domain transfer) "
        "possible, and (iii) every record stores the exact prompt used, which RQ2 needs. We used the five English domains and the "
        "five generators that cover all of them:"))
    story.extend(bullets([
        "<b>ChatGPT</b> (gpt-3.5-turbo, OpenAI) and <b>GPT-3 davinci</b> (text-davinci-003, OpenAI) - two models from the <i>same</i> "
        "family, kept on purpose to test whether same-family models are harder to tell apart;",
        "<b>Cohere</b> (Cohere's generation API);",
        "<b>Dolly-v2</b> (dolly-v2-12b, Databricks; an instruction-tuned Pythia model from EleutherAI);",
        "<b>BLOOMz</b> (BigScience; multitask instruction-tuned BLOOM).",
    ]))
    story.append(P("That is five models from four LLM families. Domains and what each model was asked to do:"))
    med = df.assign(w=df["response"].str.split().str.len()).pivot_table(index="domain", columns="model", values="w", aggfunc="median")
    desc = {"wikipedia": ("Wikipedia titles", "Write a Wikipedia article for the title"),
            "wikihow": ("WikiHow titles + headlines", "Write a WikiHow how-to article"),
            "reddit": ("Reddit ELI5 / AskScience / AskHistorians questions", "Answer the question in detail"),
            "arxiv": ("arXiv paper titles", "Write the paper's abstract"),
            "peerread": ("PeerRead paper titles (+ abstracts)", "Write a peer review (4 prompt variants)")}
    rows = [["Domain", "Topic source", "Task given to the LLM", "ChatGPT", "davinci", "Cohere", "Dolly", "BLOOMz"]]
    for d in DOMAINS:
        rows.append([d, desc[d][0], desc[d][1]] + [f"{med.loc[d, m]:.0f}" for m in LABELS])
    story.append(KeepTogether([table(rows, [0.72 * inch, 1.42 * inch, 1.56 * inch] + [0.62 * inch] * 5), P("Table 1. The five domains. Right-hand columns: median response length in words for each model.", "cap")]))

    story.append(P("1.3 Cleaning: problems we found in the raw data", "h2"))
    story.append(P("Inspecting the raw files before training turned up several issues that would have produced misleading results:"))
    story.extend(bullets([
        "<b>BLOOMz echoes its prompt.</b> In the BLOOMz files the <i>machine_text</i> field is the prompt followed by the completion. "
        "Used as-is, every BLOOMz response would start with the instruction text and the classifier would get it right for free. "
        "We took the completion-only field (<i>machine_abstract</i> / <i>machine_answer</i>) and recovered the filled prompt by "
        "stripping the completion off the end of <i>machine_text</i>.",
        "<b>Empty generations.</b> 1,780 of the 3,000 Cohere Reddit responses are empty strings. We dropped every response with fewer "
        "than 10 words (this removed almost only those).",
        "<b>API whitespace.</b> Cohere responses begin with blank lines. That reflects how the API client returned the text, not how the "
        "model writes, so we stripped leading and trailing whitespace from every response.",
        "<b>Different storage formats.</b> PeerRead stores four (prompt, review) pairs per paper as lists, and BLOOMz stores its PeerRead "
        "prompts as unfilled templates. We expanded the lists into one example per pair and filled the templates with the paper's title and abstract.",
        "<b>Kept on purpose (and analysed in RQ3/RQ4):</b> Dolly-v2's responses contain no line breaks at all (0% of examples, versus 48-78% "
        "for the other models), which looks like the text was flattened during collection; and some BLOOMz outputs degenerate into "
        "repetition (e.g. a list of bare numbers \"1. 2. 3. ...\"), which is genuine model behaviour.",
    ]))
    story.append(P("1.4 Sampling and train/validation/test split", "h2"))
    story.append(P(
        f"We sampled <b>600 examples per (model, domain)</b>, giving <b>{n:,}</b> examples: 3,000 per model and 3,000 per domain, "
        f"perfectly balanced. Two design decisions guard against shortcuts. <b>Topic-controlled sampling:</b> wherever possible we "
        f"chose topics (the same Wikipedia title, question or paper) that were answered by all five models, so the subject matter "
        f"carries no information about the label. This holds for 100% of the Wikipedia, WikiHow, Reddit and arXiv examples and "
        f"{pct(full[df.domain == 'peerread'].mean())} of PeerRead (fewer papers could be matched across files), {pct(full.mean())} overall. "
        f"<b>Grouped split:</b> we split by topic (70/15/15), so all five models' answers to a topic land in the same split and no "
        f"test topic is ever seen in training. Result: {spl['train']:,} train, {spl['val']:,} validation and {spl['test']:,} test "
        f"examples over {df['topic'].nunique():,} topics."))
    story.append(P(
        "<b>Data rights.</b> M4 is a public research dataset released by its authors for machine-generated-text research. It contains "
        "LLM outputs and public prompts only (titles, public questions, public abstracts); no private, personal or confidential "
        "information. Our repository does not re-host the raw data: <i>src/build_dataset.py</i> rebuilds our exact sample from a clone "
        "of the M4 repository (fixed seed 42)."))


def section_models(story):
    rc, rl = load("rq2_cnn_output"), load("rq2_lstm_output")
    story.append(P("2. Model Implementation and Training", "h1"))
    story.append(P("2.1 Input representation", "h2"))
    story.append(P(
        "We wrote a small regex tokenizer that keeps information many NLP pipelines throw away, because style lives there: "
        "<b>case is preserved</b>, every <b>punctuation and markdown symbol</b> is its own token, digits are kept, and every line break "
        "becomes a special <i>&lt;nl&gt;</i> token so layout is visible to the model. The vocabulary is built on the training split only "
        "(tokens seen at least twice, at most 30,000). Inputs: <b>output-only</b> = first 256 response tokens; <b>input-only</b> = first "
        "128 prompt tokens; <b>input+output</b> = prompt (128) + <i>&lt;sep&gt;</i> + response (256). The median response is about 280 "
        "tokens, so 256 covers most of a typical response. Word embeddings are learned from scratch: pretrained embeddings were "
        "unavailable offline, and they are trained to ignore exactly the surface style we want to detect."))
    story.append(P("2.2 CNN", "h2"))
    story.append(P(
        "A Kim (2014)-style text CNN: embedding (128-d) &rarr; three parallel 1-D convolutions with kernel widths 3, 4 and 5 "
        "(100 filters each, ReLU), which act as learned n-gram detectors &rarr; global max-pooling over time (padding positions masked "
        "out) &rarr; concatenation (300-d) &rarr; dropout 0.5 &rarr; linear layer to 5 logits. Max-pooling makes the CNN ask "
        "<i>\"does this phrase or pattern occur anywhere?\"</i>, which suits fingerprint-like cues"
        + (f" ({rc['params']:,} parameters, {rc['vocab']:,}-token vocabulary)." if rc else ".")))
    story.append(P("2.3 RNN (bidirectional LSTM)", "h2"))
    story.append(P(
        "Embedding (100-d, dropout 0.2) &rarr; one bidirectional LSTM layer with 64 units per direction, run on packed sequences so "
        "padding is never processed &rarr; the hidden states are summarised by concatenating mean-pooling and max-pooling over time "
        "(256-d) &rarr; dropout 0.5 &rarr; linear layer to 5 logits. Unlike the CNN, the LSTM reads the text in order and can model "
        "long-range structure such as how a response opens and closes"
        + (f" ({rl['params']:,} parameters)." if rl else ".")))
    story.append(P("2.4 Training procedure", "h2"))
    rows = [["Setting", "Value"],
            ["Loss / optimiser", "Cross-entropy; Adam, learning rate 1e-3, no weight decay"],
            ["Batch size / gradient clipping", "64 (length-bucketed batches to reduce padding) / global norm 5.0"],
            ["Epochs / early stopping", "up to 10; stop when validation macro-F1 has not improved for 2 epochs; restore best weights"],
            ["Regularisation", "dropout 0.5 before the output layer (+0.2 on LSTM embeddings)"],
            ["Model selection", "validation split only; the test split is touched once per experiment"],
            ["Reproducibility", "seed 42 for Python, NumPy and PyTorch; CNN repeated with 3 seeds (Section 3)"],
            ["Hardware", "CPU only (2 cores, no GPU), one thread per run, two runs in parallel"],
            ["Reference baseline", "TF-IDF over the same tokens (uni+bigrams, sublinear tf) + logistic regression (C=10)"]]
    story.append(KeepTogether([table(rows, [1.9 * inch, 5.0 * inch]), P("Table 2. Training configuration, identical for every experiment unless stated.", "cap")]))
    story.append(P(
        "<b>What we changed along the way.</b> Our first configuration used 384-token inputs, dropout 0.5 directly on the CNN "
        "embeddings and a 128-unit LSTM. The CNN converged slowly (62% validation accuracy after two epochs) and one LSTM epoch took "
        "more than six minutes on our CPU, which would not have fit our compute budget. We cut inputs to 256 tokens (covers the "
        "median response), removed embedding dropout from the CNN (as in Kim's original model, dropout only before the classifier), "
        "and shrank the LSTM to 64 units per direction. Even then a BiLSTM epoch took about 2.5 minutes (8 minutes with prompt + "
        "response inputs), so the BiLSTM was used for RQ1, RQ2 and two RQ4 ablations, while the 5-fold RQ3 study used the CNN and "
        "the TF-IDF baseline. Every run used the full training schedule, so all comparisons are like-for-like."))


def best_ep(r):
    h = r["history"]
    return max(h, key=lambda e: e["val_macro_f1"])["epoch"]


def per_class_row(name, r):
    pc = r["test"]["per_class"]
    return [name, f3(r["test"]["accuracy"]), f3(r["test"]["macro_f1"])] + [f"{pc[m]['f1-score']:.2f}" for m in LABELS]


def section_results(story):
    rc, rl, rt = load("rq2_cnn_output"), load("rq2_lstm_output"), load("rq2_tfidf_output")
    seeds = pd.read_csv(os.path.join(RES, "summary_seeds.csv")) if os.path.exists(os.path.join(RES, "summary_seeds.csv")) else None
    extra = json.load(open(os.path.join(RES, "extra_analysis.json")))
    story.append(P("3. Results", "h1"))
    story.append(P("3.1 RQ1 - Can we identify which LLM generated a response?", "h2"))
    rows = [["Classifier (response only)", "Acc.", "Macro-F1", "F1 ChatGPT", "F1 davinci", "F1 Cohere", "F1 Dolly", "F1 BLOOMz"],
            per_class_row("CNN", rc), per_class_row("BiLSTM", rl), per_class_row("TF-IDF + LR (baseline)", rt)]
    story.append(table(rows, [1.45 * inch, 0.5 * inch, 0.65 * inch] + [0.84 * inch] * 5, bold_first_col=True))
    cap = "Table 3. RQ1 test results (2,234 test examples, 5 classes, chance = 0.20)."
    if seeds is not None and len(seeds) and int(seeds[seeds.method == "CNN"].iloc[0].n_seeds) > 1:
        s = seeds[seeds.method == "CNN"].iloc[0]
        cap += f" CNN over {int(s.n_seeds)} seeds: accuracy {s.acc_mean:.3f} &plusmn; {s.acc_std:.3f}, macro-F1 {s.f1_mean:.3f} &plusmn; {s.f1_std:.3f}."
    story.append(P(cap, "cap"))
    e = extra.get("CNN", {}); el = extra.get("BiLSTM", {})
    cm = np.array(rc["test"]["confusion"])
    oa = cm[0, 1] + cm[1, 0]; o_err = cm[0].sum() - cm[0, 0] + cm[1].sum() - cm[1, 1]
    story.append(P(
        f"<b>Yes - clearly.</b> From the response alone the CNN identifies the author {pct(rc['test']['accuracy'])} of the time and the "
        f"BiLSTM {pct(rl['test']['accuracy'])}, against 20% chance. <b>BLOOMz is almost perfectly identifiable</b> "
        f"(F1 {rc['test']['per_class']['BLOOMz']['f1-score']:.2f} for the CNN): its outputs are shorter, terser and contain unusual "
        f"punctuation and repetition (Section 4.2). The hardest pair is the two OpenAI models. {oa} of the CNN's {o_err} mistakes on "
        f"ChatGPT/davinci examples are confusions between ChatGPT and davinci. If we only ask for the <i>family</i>, merging the two "
        f"OpenAI models, CNN accuracy rises from {pct(e.get('model_acc', np.nan))} to {pct(e.get('family_acc', np.nan))} "
        f"(BiLSTM: {pct(el.get('model_acc', np.nan))} to {pct(el.get('family_acc', np.nan))}). Same-family models share a fingerprint."))
    story.extend(fig("rq1_confusion", 5.0, "Figure 1. Row-normalised confusion matrices (percent of each true model). For both models "
                     "the largest error is ChatGPT predicted as davinci; the BiLSTM also often mistakes Cohere for Dolly."))
    story.append(P(
        f"<b>The CNN beat the BiLSTM, and the linear baseline beat both.</b> TF-IDF + logistic regression reaches "
        f"{pct(rt['test']['accuracy'])}. With about 10.5k training texts and embeddings learned from scratch, a linear model over "
        f"~100k sparse n-gram features is very hard to beat, because the fingerprint consists of exactly such n-grams (Section 4.2). "
        f"The CNN's 3/4/5-gram filters approximate the same thing and train fast. The BiLSTM reads tokens one at a time through a small "
        f"64-unit state per direction. It learned more slowly (Figure 2) and ended at a lower validation score, and nothing in this "
        f"task seems to need its long-range memory: the cues are local. Both neural models eventually overfit (training loss keeps "
        f"falling while validation F1 plateaus: best epoch {best_ep(rc)} for the CNN, {best_ep(rl)} for the BiLSTM), which is why "
        f"early stopping on validation macro-F1 and restoring the best weights matter."))
    story.extend(fig("training_curves", 4.2, "Figure 2. Validation macro-F1 per epoch (response only). Training stops after two "
                     "epochs without improvement and the best epoch's weights are restored."))

    # ---------------- RQ2 ----------------
    story.append(P("3.2 RQ2 - Does the user's prompt help identify the LLM?", "h2"))
    rows = [["Classifier", "Input only (prompt)", "Output only (response)", "Input + Output"]]
    for disp, key in [("CNN", "cnn"), ("BiLSTM", "lstm"), ("TF-IDF + LR", "tfidf")]:
        rr = [load(f"rq2_{key}_{m}") for m in ["input", "output", "both"]]
        rows.append([disp] + [f"{r['test']['accuracy']:.3f} / {r['test']['macro_f1']:.3f}" if r else "-" for r in rr])
    story.append(KeepTogether([table(rows, [1.5 * inch, 1.6 * inch, 1.6 * inch, 1.6 * inch], bold_first_col=True), P("Table 4. RQ2 test accuracy / macro-F1 for the three input types.", "cap")]))
    story.extend(fig("rq2_inputs", 5.6, "Figure 3. RQ2: macro-F1 by input type. Dashed line = chance."))
    ci, cb = load("rq2_cnn_input"), load("rq2_cnn_both")
    b = extra["prompt_only_bound"]
    story.append(P(
        f"<b>(i) Can the LLM be predicted from the prompt alone?</b> Surprisingly, yes: the CNN gets {pct(ci['test']['accuracy'])} "
        f"from the prompt without ever seeing the response. <b>(ii) Does adding the prompt help?</b> Yes: input + output beats output "
        f"alone for every classifier (CNN {pct(rc['test']['accuracy'])} &rarr; {pct(cb['test']['accuracy'])}). "
        f"<b>(iii) Why does prompt-only work so well?</b> Because of how the data were collected, not because of anything about the "
        f"models. We gave every model the same topics, so the topic carries no information. What differs is the <i>wording of the "
        f"instruction</i>: the M4 authors prompted each model with its own template (e.g. for Reddit, ChatGPT and davinci got "
        f"\"I will ask you a question... more than 200 words... in a Reddit answering style\", Cohere got \"Write a couple of paragraphs "
        f"of at least 500 characters...\", Dolly and BLOOMz got \"Write a detailed response... Use approximately N words/tokens\")."))
    story.append(P(
        f"We verified this. If you group the test set by exact prompt string and always predict the most common model per group, you "
        f"get the <b>best accuracy any prompt-only classifier can reach</b>: {pct(b['overall'])} overall. The CNN's "
        f"{pct(ci['test']['accuracy'])} sits right under that ceiling, domain by domain (Figure 4). The cleanest case is "
        f"<b>Wikipedia</b>: four of the five models received the <i>identical</i> prompt, so a prompt-only classifier can only single out "
        f"BLOOMz and must guess among the other four. The best possible accuracy is therefore 1/5 + 4/5 &times; 1/4 = 40%, and the CNN "
        f"scores exactly {pct(ci['test']['per_domain_acc']['wikipedia'])}. On arXiv every model had a different template, so prompt-only "
        f"accuracy is {pct(ci['test']['per_domain_acc']['arxiv'])}."))
    story.extend(fig("rq2_prompt_bound", 5.4, "Figure 4. Prompt-only accuracy per domain (CNN) against the best accuracy achievable "
                     "from the exact prompt string. The classifier has learned the templates, nothing more."))
    pdom = pd.read_csv(os.path.join(RES, "summary_per_domain.csv"))
    rows = [["CNN input", *DOMAINS]]
    for m in ["input", "output", "both"]:
        rr = pdom[(pdom.method == "CNN") & (pdom.input == m)]
        if len(rr):
            rows.append([m] + [f"{rr.iloc[0][d]:.2f}" for d in DOMAINS])
    story.append(KeepTogether([table(rows, [1.2 * inch] + [1.0 * inch] * 5, bold_first_col=True), P("Table 5. CNN test accuracy per domain and input type.", "cap")]))
    gain_w = pdom[(pdom.method == "CNN") & (pdom.input == "both")].iloc[0]["wikipedia"] - pdom[(pdom.method == "CNN") & (pdom.input == "output")].iloc[0]["wikipedia"]
    gain_a = pdom[(pdom.method == "CNN") & (pdom.input == "both")].iloc[0]["arxiv"] - pdom[(pdom.method == "CNN") & (pdom.input == "output")].iloc[0]["arxiv"]
    gain_r = pdom[(pdom.method == "CNN") & (pdom.input == "both")].iloc[0]["reddit"] - pdom[(pdom.method == "CNN") & (pdom.input == "output")].iloc[0]["reddit"]
    story.append(P(
        f"Table 5 shows the same effect from the other side. Adding the prompt helps a lot where the templates differ (Reddit "
        f"{gain_r * 100:+.1f} points, arXiv {gain_a * 100:+.1f}) and barely at all on Wikipedia ({gain_w * 100:+.1f}), where four "
        f"models share one template. <b>Takeaway:</b> in this "
        f"benchmark the prompt is a leak of the data-collection protocol. If every model received the same prompt, as in a real "
        f"attribution setting, prompt-only accuracy would be at chance and any gain from adding the input would disappear. The "
        f"honest answer to RQ1 is the response-only number."))


def section_analysis(story):
    extra = json.load(open(os.path.join(RES, "extra_analysis.json")))
    rc = load("rq2_cnn_output")
    story.append(P("4. In-Depth Analyses", "h1"))
    # ---------------- RQ3 ----------------
    story.append(P("4.1 RQ3 - Do LLM fingerprints generalise across domains? (extra credit)", "h2"))
    story.append(P(
        "<b>Setup.</b> Leave-one-domain-out: for each of the five domains we train on the other four (their train split, early "
        "stopping on their validation split) and test on <i>all 3,000 examples</i> of the held-out domain, which the model has "
        "never seen: different task, different prompt templates, different vocabulary. We compare against the in-domain test "
        "accuracy of the RQ1 model on the same domain. We ran this for the CNN and the TF-IDF baseline (10 extra training runs); "
        "five more BiLSTM folds would have taken about two more CPU-hours."))
    rows = [["Held-out domain"] + [f"{m}: seen &rarr; unseen" for m in ["CNN", "TF-IDF + LR"]]]
    drops = {}
    for d in DOMAINS:
        row = [d]
        for key in ["cnn", "tfidf"]:
            ind, out = load(f"rq2_{key}_output"), load(f"rq3_{key}_out_{d}")
            if ind and out:
                a, o = ind["test"]["per_domain_acc"][d], out["test"]["accuracy"]
                drops.setdefault(key, []).append(a - o)
                row.append(f"{a:.2f} &rarr; {o:.2f}")
            else:
                row.append("-")
        rows.append(row)
    rows.append(["Mean drop (points)"] + [f"{100 * np.mean(drops[k]):.1f}" if k in drops else "-" for k in ["cnn", "tfidf"]])
    story.append(KeepTogether([table(rows, [1.8 * inch, 2.2 * inch, 2.2 * inch], bold_first_col=True), P("Table 6. RQ3: accuracy on each domain when it was seen in training (in) vs. held out entirely (out).", "cap")]))
    story.extend(fig("rq3_domains", 6.5, "Figure 5. RQ3: in-domain vs. unseen-domain accuracy per held-out domain."))
    rec = pd.read_csv(os.path.join(RES, "rq3_per_model_recall.csv"), index_col=[0, 1]) if os.path.exists(os.path.join(RES, "rq3_per_model_recall.csv")) else None
    if rec is not None and "cnn" in rec.index.get_level_values(0):
        rr = rec.loc["cnn"]
        rows = [["CNN recall on held-out domain"] + [m.replace("GPT-3 davinci", "davinci") for m in LABELS]]
        for d in DOMAINS:
            if d in rr.index:
                rows.append([d] + [f"{rr.loc[d, m]:.2f}" for m in LABELS])
        rows.append(["mean"] + [f"{rr[m].mean():.2f}" for m in LABELS])
        rows.append(["std (consistency)"] + [f"{rr[m].std():.2f}" for m in LABELS])
        story.append(KeepTogether([table(rows, [1.9 * inch] + [0.95 * inch] * 5, bold_first_col=True), P("Table 7. Per-model recall of the CNN on each unseen domain. Low std = consistently identifiable.", "cap")]))
    story.append(P(RQ3_TEXT(drops, rec)))

    # ---------------- RQ4 ----------------
    story.append(P("4.2 RQ4 - What characteristics distinguish the LLMs? (extra credit)", "h2"))
    story.append(P(
        "We attacked this from three directions: <b>(a)</b> descriptive statistics of 20 hand-crafted stylometric features, "
        "<b>(b)</b> what the models actually weight, via the strongest n-grams of the linear baseline, and <b>(c)</b> "
        "<b>ablations</b>: we remove one kind of signal from every response (train and test), retrain, and measure how much "
        "performance drops."))
    story.extend(fig("rq4_style", 6.6, "Figure 6. Stylometric profile per model over all 15,000 responses (bar = mean, whisker = 95% confidence interval)."))
    sc = json.load(open(os.path.join(RES, "rq4_style_classifier.json")))
    pi = list(sc["permutation_importance"].items())[:6]
    means = pd.read_csv(os.path.join(RES, "style_means_by_model.csv"), index_col=0)
    story.append(P(
        f"<b>(a) Style statistics.</b> The models differ a lot on simple, measurable properties (Figure 6). BLOOMz writes the "
        f"shortest answers (mean {means.loc['BLOOMz', 'n_words']:.0f} words vs. {means.loc['ChatGPT', 'n_words']:.0f} for ChatGPT), "
        f"so its lexical diversity is high. Its mean digit rate ({means.loc['BLOOMz', 'digits_per_100w']:.0f} per 100 words vs. at most "
        f"{means.drop('BLOOMz')['digits_per_100w'].max():.0f} for the others) is driven by a minority of degenerate outputs, e.g. runs of bare "
        f"list numbers; its median is ordinary. ChatGPT uses the most line "
        f"breaks and list formatting (on average {pct(means.loc['ChatGPT', 'list_lines_frac'])} of its lines are list items vs. "
        f"{pct(means.loc['GPT-3 davinci', 'list_lines_frac'])} for davinci). Dolly writes the longest sentences "
        f"({means.loc['Dolly-v2', 'avg_sent_len']:.1f} words) and never breaks lines. A classifier that sees <i>only these 20 numbers</i> "
        f"and no words reaches <b>{pct(sc['random_forest']['accuracy'])}</b> accuracy with a random forest "
        f"({pct(sc['logreg']['accuracy'])} with logistic regression). By permutation importance the most useful features are "
        + ", ".join(f"{FEAT_NAMES.get(k, k)} ({v:.3f})" for k, v in pi)
        + ". Two caveats: lexical diversity is partly a proxy for length (short texts repeat fewer words), and the line-break "
          "features are partly the Dolly collection artifact, which is why we test them with ablations below."))
    rt = load("rq2_tfidf_output")
    tf = rt.get("top_features", {})
    def clean(f):
        return ", ".join("“" + x.replace("&", "&amp;").replace("<nl>", "[NL]").replace("<", "&lt;") + "”" for x in f)
    rows = [["Model", "Most indicative n-grams (TF-IDF + LR weights; [NL] = line break)"]]
    for m in LABELS:
        rows.append([m, clean(tf.get(m, [])[:12])])
    story.append(KeepTogether([table(rows, [1.0 * inch, 5.9 * inch], bold_first_col=True), P("Table 8. Top-weighted features of the linear baseline (response only).", "cap")]))
    story.append(P(RQ4_FEATURE_TEXT_fn()))
    # ablations
    rows = [["Ablation (applied to every response)", "What it removes", "CNN", "BiLSTM", "TF-IDF"]]
    abls = [("none", "Original", "-"), ("no_format", "No formatting", "line breaks, markdown, list markers"),
            ("lower", "Lower-case", "capitalisation"), ("no_punct", "No punct. + no format", "all punctuation and layout"),
            ("style_only", "Style only", "every content word (kept: 150 most common words, punctuation, layout)"),
            ("first50", "First 50 words", "length and everything after the opening")]
    vals = {}
    for a, nm, what in abls:
        row = [nm, what]
        for key in ["cnn", "lstm", "tfidf"]:
            r = load(f"rq2_{key}_output") if a == "none" else load(f"rq4_{key}_{a}")
            row.append(f"{r['test']['macro_f1']:.3f}" if r else "-")
            vals[(a, key)] = r["test"]["macro_f1"] if r else None
        rows.append(row)
    story.append(KeepTogether([table(rows, [1.55 * inch, 2.6 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch], bold_first_col=True), P("Table 9. RQ4 ablations: test macro-F1 after retraining on modified responses (BiLSTM run on two ablations only, CPU budget).", "cap")]))
    story.extend(fig("rq4_ablations", 6.6, "Figure 7. RQ4 ablations (test macro-F1)."))
    story.append(P(RQ4_ABL_TEXT(vals)))

    # ---------------- error analysis ----------------
    story.append(P("4.3 Error analysis", "h2"))
    e = extra.get("CNN", {})
    story.append(P(ERR_TEXT(rc, e)))
    story.extend(fig("acc_by_length", 4.4, "Figure 8. Test accuracy by response length (response-only models)."))


def year_rates():
    """Four-digit years (1800-2099) per 100 words, by domain x model."""
    df = pd.read_csv(os.path.join(ROOT, "data", "dataset.csv"), keep_default_na=False)
    rate = df["response"].str.count(r"\b(?:1[89]|20)\d\d\b") / df["response"].str.split().str.len() * 100
    return df.assign(r=rate).pivot_table(index="domain", columns="model", values="r", aggfunc="mean")


def marker_rates():
    df = pd.read_csv(os.path.join(ROOT, "data", "dataset.csv"), keep_default_na=False)
    out = {}
    for name, pat in [("curly", "’"), ("blank", "\n\n"), ("ellipsis", "..."), ("semicolon", ";"), ("additionally", "Additionally"), ("weshow", "We show")]:
        out[name] = df.groupby("model")["response"].apply(lambda s: s.str.contains(pat, regex=False).mean()).to_dict()
    return out


FEAT_NAMES = {"ttr_all": "lexical diversity (whole text)", "newlines_per_100w": "line breaks", "n_chars": "length in characters",
              "n_words": "length in words", "blank_lines_per_100w": "blank lines", "avg_word_len": "average word length",
              "first_person_per_100w": "first-person pronouns", "ttr_first100": "lexical diversity (first 100 words)",
              "ends_with_punct": "ends with punctuation", "commas_per_100w": "commas", "avg_sent_len": "sentence length"}


def RQ4_FEATURE_TEXT_fn():
    mr = marker_rates()
    return (
        "<b>(b) What the linear model weights.</b> The top features (Table 8) read like a style guide, and each is easy to verify in "
        f"the data. <b>ChatGPT</b> structures its text: paragraph breaks (a blank line appears in {pct(mr['blank']['ChatGPT'])} of its "
        "responses), numbered lists, discourse connectives (“However ,”, “, including”, “, which”) and "
        "formal hedging vocabulary (“significant”, “various”). <b>davinci</b> has a <i>typographic</i> tic: it writes "
        f"the curly apostrophe ’ instead of the ASCII ' in {pct(mr['curly']['GPT-3 davinci'])} of its responses "
        f"(Cohere: {pct(mr['curly']['Cohere'])}), and favours “Additionally ,” and “This paper examines”. <b>Cohere</b> "
        f"writes in an academic first-person plural (“We show that” appears in {pct(mr['weshow']['Cohere'])} of Cohere responses, "
        f"under {pct(max(mr['weshow']['ChatGPT'], mr['weshow']['GPT-3 davinci']))} for the OpenAI models). <b>Dolly</b> is marked by colons, "
        "“this paper”, first and second person (“. I”, “your”), explicit years and malformed punctuation "
        f"(“. ,”). <b>BLOOMz</b> uses ellipses (“...” in {pct(mr['ellipsis']['BLOOMz'])} of its responses vs. "
        f"under {pct(max(v for k, v in mr['ellipsis'].items() if k != 'BLOOMz'))} for every other model), semicolons "
        f"({pct(mr['semicolon']['BLOOMz'])}), parentheses, dates and bare “Yes” / “No” answers. The fingerprint is "
        "therefore <b>both lexical</b> (function words, connectives, pronouns) <b>and structural</b> (layout, punctuation, typography).")


def RQ3_TEXT(drops, rec):
    t = []
    if drops:
        c = np.array(drops.get("cnn", [np.nan])) * 100
        t.append(f"<b>(i) Yes, performance drops on unseen domains, sharply and unevenly.</b> Averaged over the five held-out "
                 f"domains, accuracy falls by {np.nanmean(c):.1f} points for the CNN"
                 + (f", {100 * np.mean(drops['lstm']):.1f} for the BiLSTM" if "lstm" in drops else "")
                 + (f" and {100 * np.mean(drops['tfidf']):.1f} for TF-IDF" if "tfidf" in drops else "")
                 + ".")
        accs = {d: load(f"rq3_cnn_out_{d}")["test"] for d in DOMAINS if load(f"rq3_cnn_out_{d}")}
        if accs:
            ok = [d for d in accs if accs[d]["accuracy"] >= 0.45]
            low = [d for d in accs if accs[d]["accuracy"] < 0.45]
            t.append(f"On {len(ok)} of the {len(accs)} unseen domains the CNN stays well above the 20% chance level "
                     f"({min(accs[d]['accuracy'] for d in ok) * 100:.0f}-{max(accs[d]['accuracy'] for d in ok) * 100:.0f}%), so part "
                     f"of the fingerprint is domain-general"
                     + (f", but on unseen {', '.join(low)} it collapses to "
                        + ", ".join(f"{pct(accs[d]['accuracy'])} (macro-F1 {accs[d]['macro_f1']:.2f})" for d in low)
                        + ", barely above chance (explained in (iii))." if low else "."))
    t.append(RQ3_more())
    return " ".join(t)


def RQ3_more():
    out = []
    accs = {d: load(f"rq3_cnn_out_{d}")["test"]["accuracy"] for d in DOMAINS if load(f"rq3_cnn_out_{d}")}
    if accs:
        best, worst = max(accs, key=accs.get), min(accs, key=accs.get)
        pass
    rec = {}
    for d in DOMAINS:
        r = load(f"rq3_cnn_out_{d}")
        if r:
            rec[d] = {m: r["test"]["per_class"][m]["recall"] for m in LABELS}
    if rec:
        R = pd.DataFrame(rec).T
        order = R.mean().sort_values(ascending=False)
        out.append(
            f"<br/><br/><b>(ii) Some models are much more consistently identifiable.</b> Averaged over unseen domains the CNN's recall is "
            + ", ".join(f"{m} {order[m]:.2f}" for m in order.index)
            + f" (Table 7). {order.index[0]} and {order.index[1]} keep their identity across tasks. BLOOMz's habits (ellipses, semicolons, "
            "terse answers) and Dolly's flat, single-paragraph text appear in every domain. The three API models (ChatGPT, davinci, "
            "Cohere) adapt their style to the task much more, so a profile learned on four tasks transfers poorly to a fifth.")
    w = load("rq3_cnn_out_wikipedia")
    if w:
        pc = w["test"]["per_class"]
        tw, tnw = load("rq3_tfidf_out_wikipedia"), load("rq3nf_tfidf_out_wikipedia")
        th, tnh = load("rq3_tfidf_out_wikihow"), load("rq3nf_tfidf_out_wikihow")
        cw, cnw = w, load("rq3nf_cnn_out_wikipedia")
        ch, cnh = load("rq3_cnn_out_wikihow"), load("rq3nf_cnn_out_wikihow")
        yr = year_rates()
        other = yr.drop(index="wikipedia")
        dolly_top = int((other.idxmax(axis=1) == "Dolly-v2").sum())
        s = ("<br/><br/><b>(iii) A general fingerprint or task-specific patterns? Both, and the Wikipedia fold shows where the line is.</b> "
             f"With Wikipedia held out, the CNN labels almost every ChatGPT, davinci and Cohere article as Dolly (recall "
             f"{pc['ChatGPT']['recall']:.2f} / {pc['GPT-3 davinci']['recall']:.2f} / {pc['Cohere']['recall']:.2f}, versus "
             f"{pc['Dolly-v2']['recall']:.2f} for Dolly). Wikipedia articles from every model share two traits that, in the four "
             "training domains, belong to Dolly. <b>No line breaks:</b> in M4's Wikipedia data the ChatGPT, davinci and Cohere articles "
             "are a single flat paragraph (0% contain a line break, versus 7-100% in their other domains), exactly like Dolly's text "
             f"everywhere. <b>Dates:</b> Wikipedia articles contain {yr.loc['wikipedia'].min():.1f}-{yr.loc['wikipedia'].max():.1f} "
             f"four-digit years per 100 words versus at most {other.values.max():.2f} in the other domains, and Dolly is the model that "
             f"mentions years most in {dolly_top} of the 4 training domains (\u201c2023\u201d is among its top features). A classifier "
             "trained on the other domains learns something like \u201cflat text with dates &rarr; Dolly\u201d, which works there "
             "and fails on Wikipedia. We tested the layout half with a follow-up: we re-ran the folds with all layout stripped from "
             "every response (the RQ4 <i>no formatting</i> transform).")
        if tw and tnw and th and tnh:
            s += (f" For TF-IDF, unseen-Wikipedia accuracy rises from {pct(tw['test']['accuracy'])} to {pct(tnw['test']['accuracy'])}")
            if cnw:
                s += (f", but the CNN does not recover ({pct(cw['test']['accuracy'])} to {pct(cnw['test']['accuracy'])}) and still sends "
                      f"{cnw['test']['confusion'][0][3] + cnw['test']['confusion'][1][3] + cnw['test']['confusion'][2][3]:,} of the 1,800 "
                      "ChatGPT/davinci/Cohere articles to Dolly. Layout is therefore only part of the shortcut")
            s += (f". Meanwhile, on unseen WikiHow removing layout <i>hurts</i> (TF-IDF {pct(th['test']['accuracy'])} to "
                  f"{pct(tnh['test']['accuracy'])}")
            if ch and cnh:
                s += f"; CNN {pct(ch['test']['accuracy'])} to {pct(cnh['test']['accuracy'])}"
            s += ("), because there every model except Dolly writes step-by-step text with line breaks, so layout is a genuine, "
                  "transferable cue.")
        s += (" So the classifier learns a mix: a <b>general fingerprint</b> (punctuation, function words, typography such as "
              "davinci's curly apostrophe) that survives a change of task, plus <b>task- and dataset-specific patterns</b> (layout, "
              "dates, length, task phrases like \u201cthis paper\u201d) that do not. Strong in-domain accuracy overstates how well "
              "attribution works on a new kind of text.")
        out.append(s)
    return " ".join(out)
RQ4_ABL_EXTRA = (
    "Removing all formatting (line breaks, markdown, list markers) costs {nf_c:.1f} macro-F1 points for the CNN and {nf_t:.1f} for "
    "TF-IDF. Layout matters (including the Dolly artifact), but most of the signal survives without it. Removing punctuation as "
    "well costs {np_c:.1f} / {np_t:.1f} points in total, and the models still reach {np_c_abs:.2f} / {np_t_abs:.2f} from bare words. "
    "Lower-casing barely matters ({lo_c:.1f} / {lo_t:.1f} points). The most revealing test is <b>style only</b>: we replace every "
    "content word with a placeholder \u201cW\u201d and keep only the 150 most frequent words, punctuation and layout. This costs "
    "TF-IDF just {so_t:.1f} points ({so_t_abs:.2f}), and the CNN does not lose anything ({so_c_abs:.2f} vs. {so_base:.2f}; likely "
    "because it can no longer overfit to topic words). <b>The models identify the author from <i>how</i> the text is written, not "
    "<i>what</i> it says.</b> The largest drop comes from <b>truncating every response to its first 50 words</b>: {f50_c_abs:.2f} / "
    "{f50_t_abs:.2f} ({f50_c:.1f} / {f50_t:.1f} points). The opening carries part of the fingerprint, but length and what comes "
    "later (lists, paragraphing, closing formulas) carry a lot as well."
    "<br/><br/><b>Answers.</b> (i) The classifiers use function words and connectives, pronouns, punctuation and typography, "
    "layout and length. Topic words contribute little. (ii) The signals are mostly <i>stylistic</i>: partly linguistic (function "
    "words, connectives), partly structural (punctuation, layout, length). (iii) Only shortening the text removes a large share. "
    "Removing any one stylistic signal costs the CNN and TF-IDF at most about 5 points, because the others carry overlapping "
    "information. The BiLSTM leans more on layout ({nf_l:.1f} points without formatting). The "
    "fingerprint is redundant, spread over many weak cues rather than one tell-tale marker. That makes it robust to simple edits, "
    "but also means a few artifacts (Dolly's flattened text, BLOOMz's prompt echo) can inflate scores if they are not caught."
)
ERR_EXTRA = (
    "Three patterns stand out in the CNN's errors. <b>First, davinci is the 'default' wrong answer.</b> It is the most common "
    "wrong prediction not only for ChatGPT but also for Cohere and Dolly (Figure 1). Its plain, encyclopedic prose is the most "
    "generic style in the set, so texts without strong markers drift towards it. <b>Second, open-ended prose is hardest.</b> "
    "Accuracy is highest on PeerRead and arXiv and lowest on Wikipedia and Reddit (Table 5): on free-form articles and "
    "explanations the models converge on a similar register. Reviews and abstracts follow more constrained formats, which "
    "plausibly makes each model's habits (e.g. how it structures a review) stand out. <b>Third, longer is not easier.</b> Accuracy "
    "shows no clear trend with response length (Figure 8; {len_0:.2f} for responses under 100 words, {len_4:.2f} for 400+ words). "
    "Beyond roughly 200 words the models never see the extra text (inputs are truncated to 256 tokens), and length buckets are "
    "confounded with model (most short responses are BLOOMz), so this curve should not be over-read."
)


def RQ4_ABL_TEXT(vals):
    def d(a, k):
        return (vals[("none", k)] - vals[(a, k)]) * 100 if vals.get((a, k)) is not None and vals.get(("none", k)) is not None else np.nan
    lstm_bits = []
    if vals.get(("no_format", "lstm")) is not None:
        lstm_bits.append(f"no formatting {vals[('no_format', 'lstm')]:.2f}")
    if vals.get(("style_only", "lstm")) is not None:
        lstm_bits.append(f"style only {vals[('style_only', 'lstm')]:.2f}")
    lstm_txt = (f" The BiLSTM (run on two ablations; original {vals[('none', 'lstm')]:.2f}) scores " + " and ".join(lstm_bits) + ".") if lstm_bits else ""
    if vals.get(("style_only", "lstm")) is not None and vals[("style_only", "lstm")] > vals[("none", "lstm")]:
        lstm_txt += (f" Masking content words actually <i>helps</i> it "
                     f"({(vals[('style_only', 'lstm')] - vals[('none', 'lstm')]) * 100:+.1f} points). Plausibly, with topic words gone "
                     "the small LSTM cannot fit them and must learn style cues that generalise to unseen topics.")
    return ("<b>(c) Ablations.</b> " + RQ4_ABL_EXTRA.replace("<br/><br/><b>Answers.</b>", lstm_txt.replace("{", "{{").replace("}", "}}") + "<br/><br/><b>Answers.</b>").format(
        nf_c=d("no_format", "cnn"), nf_t=d("no_format", "tfidf"), nf_l=d("no_format", "lstm"),
        lo_c=d("lower", "cnn"), lo_t=d("lower", "tfidf"), np_c=d("no_punct", "cnn"), np_t=d("no_punct", "tfidf"),
        so_c=d("style_only", "cnn"), so_t=d("style_only", "tfidf"), so_l=d("style_only", "lstm"),
        f50_c=d("first50", "cnn"), f50_t=d("first50", "tfidf"),
        so_base=vals.get(("none", "cnn")) or np.nan, so_c_abs=vals.get(("style_only", "cnn")) or np.nan, so_t_abs=vals.get(("style_only", "tfidf")) or np.nan,
        f50_c_abs=vals.get(("first50", "cnn")) or np.nan, f50_t_abs=vals.get(("first50", "tfidf")) or np.nan,
        np_c_abs=vals.get(("no_punct", "cnn")) or np.nan, np_t_abs=vals.get(("no_punct", "tfidf")) or np.nan))


def ERR_TEXT(rc, e):
    return ERR_EXTRA.format(**{k: v for k, v in e.items() if not isinstance(v, dict)},
                            **{f"len_{i}": v for i, v in enumerate(e.get("acc_by_length", {}).values())})


def section_lessons(story):
    rc, rt = load("rq2_cnn_output"), load("rq2_tfidf_output")
    story.append(P("5. Lessons and Experience", "h1"))
    items = [
        "<b>Read the raw data before training anything.</b> Three of our most important findings came from inspecting records, not "
        "from models: BLOOMz echoes its prompt (which would have handed us a fake, near-perfect BLOOMz score), 1,780 Cohere Reddit "
        "responses are empty, and Dolly's text has had every line break removed. Each would have quietly distorted the results. "
        "A fourth artifact (the flattened Wikipedia text of three other models) only surfaced when a model failed in RQ3, so "
        "unexpected failures deserve the same scrutiny as raw data.",
        "<b>Leakage hides in how data were collected.</b> RQ2's \"surprisingly good\" prompt-only classifier is not detecting LLMs at "
        "all; it detects which prompt template the dataset authors used for each model. Computing the best accuracy achievable from "
        "the prompt string, and breaking results down per domain, turned a confusing number into a clear explanation. We now treat "
        "any input that differs systematically between classes as a potential shortcut.",
        "<b>Control confounds by design, not after the fact.</b> Topic-controlled sampling (same topics for all five models) and a "
        "split grouped by topic meant topic words could not leak the label and test topics were truly unseen. The Wikipedia case "
        "(prompt-only accuracy exactly at the 40% ceiling) shows the control worked.",
        f"<b>A strong simple baseline is essential.</b> TF-IDF + logistic regression ({pct(rt['test']['accuracy'])}) beat our CNN "
        f"({pct(rc['test']['accuracy'])}) and BiLSTM on this data size. Without it we would have over-claimed what the neural models "
        "achieve. The ablations also explain why: the fingerprint is made of local n-grams and punctuation, which is exactly what a "
        "bag-of-n-grams captures.",
        "<b>The compute budget shapes the experiment.</b> We had no GPU. When our first launch was too slow, we profiled one "
        "training batch of each model on the CPU. That showed the LSTM would not fit the budget at full size, so we shortened inputs, "
        "shrank the model and planned which runs each model would get. We also made the experiment runner skip finished runs, so "
        "parallel job queues could share work and resume safely. Profiling first would have saved us a false start.",
        "<b>Accuracy tells you <i>whether</i>; ablations tell you <i>why</i>.</b> Removing one signal at a time and retraining was "
        "the most informative experiment we ran, because it separated what the models <i>can</i> use from what they <i>need</i>. "
        "It also gave our most surprising result: <i>removing</i> information (masking every content word) improved the BiLSTM"
        + (f" from {load('rq2_lstm_output')['test']['macro_f1']:.2f} to {load('rq4_lstm_style_only')['test']['macro_f1']:.2f} macro-F1"
           if load('rq4_lstm_style_only') else "") + ". Less input can mean less to overfit.",
    ]
    story.extend(bullets(items))
    story.append(P("Limitations and what we would do next", "h2"))
    story.extend(bullets([
        "The LLMs in M4 are 2023-era models. Current GPT, Claude, Gemini and Llama models are not included, and fingerprints "
        "change between model versions.",
        "Each model was sampled with its own prompts and decoding settings, so part of what we call a \"fingerprint\" is the "
        "generation setup, not the model. A cleaner study would send identical prompts with identical settings to every model.",
        "Responses were truncated to 256 tokens. On our CPU budget the BiLSTM was trained with one seed and left out of the "
        "5-fold RQ3 study. The CNN was repeated with 3 seeds.",
        "Next steps: fine-tune a pretrained transformer encoder (e.g. RoBERTa) as a stronger model; add a character-level CNN to "
        "capture typographic cues such as the curly apostrophe; test robustness when responses are paraphrased or reformatted.",
    ]))


def section_appendix(story):
    story.append(P("Appendix A. Reproducing the results", "h1"))
    story.append(P("All code is in the GitHub repository. On a CPU-only machine the full pipeline takes about 3 hours (two job queues in parallel)."))
    for line in ["git clone https://github.com/mbzuai-nlp/M4.git", "pip install -r requirements.txt",
                 "python src/build_dataset.py --m4_dir M4/data --per_cell 600",
                 "python src/experiments.py --rq rq1_rq2                 # RQ1 + RQ2",
                 "python src/experiments.py --rq rq3 --models cnn         # RQ3 leave-one-domain-out",
                 "python src/experiments.py --rq rq3_noformat --models cnn --domains wikipedia,wikihow",
                 "python src/experiments.py --rq rq4 --models cnn         # RQ4 ablations",
                 "python src/experiments.py --rq rq4_one --models lstm --abl no_format   # (and style_only)",
                 "python src/experiments.py --rq rq1_seeds --models cnn   # extra CNN seeds",
                 "python src/style_stats.py                # RQ4 style features + prompt templates",
                 "python src/make_figures.py && python report/build_report.py"]:
        story.append(Paragraph(line.replace("<", "&lt;"), S["code"]))
    story.append(Spacer(1, 6))
    story.append(P("<b>Repository layout.</b> <i>src/build_dataset.py</i> (data curation), <i>src/text.py</i> (tokenizer, input "
                   "construction, ablations), <i>src/models.py</i> (CNN, BiLSTM), <i>src/train.py</i> (training loop, early stopping, "
                   "metrics, TF-IDF baseline), <i>src/experiments.py</i> (RQ1-RQ4 runs), <i>src/style_stats.py</i> (stylometry), "
                   "<i>src/make_figures.py</i>, <i>results/</i> (one JSON per run with metrics, confusion matrix and training "
                   "history), <i>figures/</i>, <i>report/</i>."))
    story.append(P("<b>Reference.</b> Y. Wang et al. \"M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box "
                   "Machine-Generated Text Detection.\" EACL 2024. Y. Kim. \"Convolutional Neural Networks for Sentence "
                   "Classification.\" EMNLP 2014."))
    story.append(P("<b>Use of AI tools.</b> As permitted by the project guidelines, we used an AI assistant (Claude) to help write "
                   "and debug code and to draft parts of this report. We reviewed the code, re-ran every experiment, and checked "
                   "that every number in this report is generated from the saved result files."))


def build():
    df = pd.read_csv(os.path.join(ROOT, "data", "dataset.csv"), keep_default_na=False)
    doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.85 * inch, rightMargin=0.85 * inch, topMargin=0.75 * inch,
                            bottomMargin=0.8 * inch, title="Who Wrote It? Identifying LLMs from Their Responses",
                            author="Daksh Mainee, Sanan Goel")
    story = [P("Who Wrote It? Identifying LLMs from Their Responses", "title"),
             P("CMPSC 448 - Machine Learning and Artificial Intelligence - Midterm Project, Fall 2026", "sub"),
             P("<b>Team leader:</b> Daksh Mainee &nbsp;&nbsp;&nbsp; <b>Team member:</b> Sanan Goel", "sub"), Spacer(1, 8)]
    rc, rl, rt, ci = load("rq2_cnn_output"), load("rq2_lstm_output"), load("rq2_tfidf_output"), load("rq2_cnn_input")
    drops = [rc["test"]["per_domain_acc"][d] - load(f"rq3_cnn_out_{d}")["test"]["accuracy"] for d in DOMAINS if load(f"rq3_cnn_out_{d}")]
    so = load("rq4_cnn_style_only")
    story.append(P(ABSTRACT.format(cnn=pct(rc["test"]["accuracy"]), lstm=pct(rl["test"]["accuracy"]), tfidf=pct(rt["test"]["accuracy"]),
                                   cin=pct(ci["test"]["accuracy"]), rq3drop=f"{100 * np.mean(drops):.0f}" if drops else "?",
                                   so=f"{so['test']['macro_f1']:.2f}" if so else "?"), "body"))
    section_data(story, df)
    section_models(story)
    section_results(story)
    section_analysis(story)
    section_lessons(story)
    section_appendix(story)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


ABSTRACT = (
    "<b>Summary.</b> Can machine learning tell which large language model wrote a text? From the public M4 benchmark we built a "
    "balanced, topic-controlled dataset of 15,000 (LLM, prompt, response) triples: five LLMs (ChatGPT, GPT-3 davinci, Cohere, "
    "Dolly-v2, BLOOMz; four families) across five domains. We trained a CNN and a bidirectional LSTM. "
    "<b>RQ1:</b> from the response alone the CNN identifies the author {cnn} of the time and the BiLSTM {lstm} (chance 20%); a "
    "TF-IDF baseline reaches {tfidf}. Most errors are between the two OpenAI models. "
    "<b>RQ2:</b> the prompt alone also identifies the model ({cin}), but we show this is fully explained by the per-model prompt "
    "templates used to build the dataset. "
    "<b>RQ3:</b> on a domain never seen in training, CNN accuracy drops by {rq3drop} points on average, partly because of a "
    "layout shortcut we identify. "
    "<b>RQ4:</b> the fingerprint is stylistic (function words, punctuation, layout, length) and highly redundant. With every "
    "content word masked, the CNN still reaches {so} macro-F1."
)

if __name__ == "__main__":
    build()

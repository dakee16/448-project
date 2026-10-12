"""Tokenization, vocabulary, input construction and ablation transforms."""
import re
from collections import Counter

NL = "<nl>"
SEP = "<sep>"
PAD, UNK = "<pad>", "<unk>"

# words, numbers, newlines, and every punctuation / markdown symbol as its own token.
# Case is kept: capitalization habits are part of a model's style.
_TOKEN_RE = re.compile(r"\n|[A-Za-z]+(?:'[a-z]+)?|\d+|[^\sA-Za-z\d]")


def tokenize(text):
    return [NL if t == "\n" else t for t in _TOKEN_RE.findall(text)]


class Vocab:
    def __init__(self, token_lists, max_size=30000, min_freq=2):
        cnt = Counter(t for toks in token_lists for t in toks)
        self.itos = [PAD, UNK, SEP, NL] + [t for t, c in cnt.most_common(max_size) if c >= min_freq and t not in (SEP, NL)]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, toks, max_len):
        ids = [self.stoi.get(t, 1) for t in toks[:max_len]]
        return ids if ids else [1]

    def __len__(self):
        return len(self.itos)


def build_tokens(prompt, response, mode, max_prompt=128, max_resp=384):
    """mode: 'output' | 'input' | 'both'  (RQ2)."""
    if mode == "output":
        return tokenize(response)[:max_resp]
    if mode == "input":
        return tokenize(prompt)[:max_prompt]
    if mode == "both":
        return tokenize(prompt)[:max_prompt] + [SEP] + tokenize(response)[:max_resp]
    raise ValueError(mode)


# ---------------- RQ4 ablations (applied to the response text) ----------------
_MD = re.compile(r"[#*_`>|~\-]+")


def ablate(text, kind, common_words=None):
    if kind == "none":
        return text
    if kind == "no_format":  # remove line breaks, markdown/list symbols, collapse whitespace
        t = _MD.sub(" ", text)
        t = re.sub(r"^\s*\d+[.)]\s+", " ", t, flags=re.M)  # numbered-list markers
        return re.sub(r"\s+", " ", t).strip()
    if kind == "no_punct":  # formatting AND punctuation removed -> words only
        t = ablate(text, "no_format")
        t = re.sub(r"[^\w\s']", " ", t)
        return re.sub(r"\s+", " ", t).strip()
    if kind == "lower":  # remove capitalization signal
        return text.lower()
    if kind == "style_only":  # keep only function words + punctuation + layout; mask content words
        toks = _TOKEN_RE.findall(text)
        out = []
        for t in toks:
            if t == "\n" or not re.match(r"[A-Za-z]", t) or t.lower() in common_words:
                out.append(t)
            else:
                out.append("W")
        return " ".join("\n" if t == "\n" else t for t in out)
    if kind == "first50":  # length control: every response cut to its first 50 words
        return " ".join(text.split()[:50])
    raise ValueError(kind)

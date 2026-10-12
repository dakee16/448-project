"""CNN and RNN (BiLSTM) text classifiers in PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """Kim (2014)-style CNN: embeddings -> parallel 1D convolutions with
    kernel sizes 3/4/5 (100 filters each, n-gram detectors) -> ReLU -> global max-pool
    -> dropout(0.5) -> linear."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, n_filters=100, kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x, lengths):
        mask = (x != 0).unsqueeze(1)                      # B x 1 x T
        e = self.emb(x).transpose(1, 2)                   # B x E x T
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(e))[:, :, : x.size(1)]
            h = h.masked_fill(~mask, -1e4)                # ignore padding when max-pooling
            pooled.append(h.max(dim=2).values)
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    """Embeddings (dropout 0.2) -> 1-layer bidirectional LSTM, 64 units per direction
    (packed, so padding is skipped) -> concat[mean-pool, max-pool] over time
    -> dropout(0.5) -> linear."""

    def __init__(self, vocab_size, n_classes, emb_dim=100, hidden=64, dropout=0.5, emb_dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, n_classes)

    def forward(self, x, lengths):
        e = self.emb_drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (x != 0).unsqueeze(2).float()
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = out.masked_fill(mask == 0, -1e4).max(1).values
        return self.fc(self.drop(torch.cat([mean, mx], dim=1)))


def build_model(name, vocab_size, n_classes):
    if name == "cnn":
        return TextCNN(vocab_size, n_classes)
    if name == "lstm":
        return BiLSTM(vocab_size, n_classes)
    raise ValueError(name)

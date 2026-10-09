"""DeepSC transceiver (Fig. 2, Table I) and the MI estimation model (eq. (9)).

    s --Embedding--> E --3x Transformer enc--> M --Dense--> X --channel--> Y
      --Dense--> M_hat --3x Transformer dec--> softmax --> s_hat

Shapes (B = batch, L = sentence length incl. <START>/<END>, N = complex
symbols per word, V = d_model):
    E, M      : B x L x V
    X, Y      : B x (L*N) complex  (= B x L x 2N real before reshaping)
"""
import math

import torch
import torch.nn as nn

from .channels import apply_channel, power_normalize


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class DeepSC(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model=128, n_heads=8, n_layers=3, d_ff=512,
                 symbols_per_word=8, dropout=0.1, rx_positional=True, norm_first=False):
        super().__init__()
        self.rx_positional = rx_positional
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.n_sym = symbols_per_word

        # ---- transmitter: semantic encoder S_beta ----
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model, dropout)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True,
                                         norm_first=norm_first)
        self.semantic_encoder = nn.TransformerEncoder(
            enc, n_layers, norm=nn.LayerNorm(d_model) if norm_first else None, enable_nested_tensor=False)
        # ---- transmitter: channel encoder C_alpha (Dense 256 ReLU, Dense 2N) ----
        self.channel_encoder = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 2 * symbols_per_word))
        # ---- receiver: channel decoder C_delta^-1 (Dense 256 ReLU, Dense 128 ReLU) ----
        self.channel_decoder = nn.Sequential(
            nn.Linear(2 * symbols_per_word, 256), nn.ReLU(), nn.Linear(256, d_model), nn.ReLU())
        # ---- receiver: semantic decoder S_chi^-1 ----
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        dec = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True,
                                         norm_first=norm_first)
        self.semantic_decoder = nn.TransformerDecoder(
            dec, n_layers, norm=nn.LayerNorm(d_model) if norm_first else None)
        self.prediction = nn.Linear(d_model, vocab_size)

    # ------------------------------------------------------------------ #
    def semantic_parameters(self):
        mods = [self.embedding, self.semantic_encoder, self.tgt_embedding,
                self.semantic_decoder, self.prediction]
        return [p for m in mods for p in m.parameters()]

    def channel_parameters(self):
        return list(self.channel_encoder.parameters()) + list(self.channel_decoder.parameters())

    # ------------------------------------------------------------------ #
    def transmit(self, src):
        """Transmitter: tokens (B, L) -> unit-power complex symbols (B, L*N), mask."""
        pad_mask = src.eq(self.pad_idx)
        e = self.pos(self.embedding(src) * math.sqrt(self.d_model))
        m = self.semantic_encoder(e, src_key_padding_mask=pad_mask)
        z = self.channel_encoder(m)  # B x L x 2N
        b, seq_len, _ = z.shape
        x = torch.complex(z[..., 0::2], z[..., 1::2]).reshape(b, seq_len * self.n_sym)
        sym_mask = (~pad_mask).repeat_interleave(self.n_sym, dim=1)
        return power_normalize(x, sym_mask), sym_mask

    def receive(self, y, src_pad_mask):
        """Channel decoder: complex symbols (B, L*N) -> M_hat (B, L, V)."""
        b = y.shape[0]
        z = torch.stack([y.real, y.imag], -1).reshape(b, -1, 2 * self.n_sym)
        m = self.channel_decoder(z)
        # The receiver knows which word slot every block of N symbols belongs to,
        # so the slot index is re-injected for free instead of being transmitted.
        return m + self.pos.pe[:, : m.size(1)] if self.rx_positional else m

    def decode_step(self, memory, src_pad_mask, tgt_in):
        t = tgt_in.size(1)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=tgt_in.device), 1)
        h = self.pos(self.tgt_embedding(tgt_in) * math.sqrt(self.d_model))
        h = self.semantic_decoder(h, memory, tgt_mask=causal,
                                  tgt_key_padding_mask=tgt_in.eq(self.pad_idx),
                                  memory_key_padding_mask=src_pad_mask)
        return self.prediction(h)

    def forward(self, src, tgt_in, channel="awgn", snr_db=10.0, **ch_kw):
        x, sym_mask = self.transmit(src)
        y = apply_channel(channel, x, snr_db, **ch_kw) * sym_mask
        pad_mask = src.eq(self.pad_idx)
        memory = self.receive(y, pad_mask)
        logits = self.decode_step(memory, pad_mask, tgt_in)
        return logits, x, y, sym_mask

    @torch.no_grad()
    def greedy_decode(self, src, start_idx, end_idx, channel="awgn", snr_db=10.0, max_len=None, **ch_kw):
        x, sym_mask = self.transmit(src)
        y = apply_channel(channel, x, snr_db, **ch_kw) * sym_mask
        pad_mask = src.eq(self.pad_idx)
        memory = self.receive(y, pad_mask)
        max_len = max_len or src.size(1) + 2
        out = torch.full((src.size(0), 1), start_idx, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            nxt = self.decode_step(memory, pad_mask, out)[:, -1].argmax(-1)
            nxt = nxt.masked_fill(done, self.pad_idx)
            out = torch.cat([out, nxt[:, None]], 1)
            done |= nxt.eq(end_idx)
            if done.all():
                break
        return out


class MINE(nn.Module):
    """Statistics network T of the Donsker-Varadhan bound (eqs. (7)-(9)).

    Input: one transmitted / received complex symbol pair (Re x, Im x, Re y, Im y).
    """

    def __init__(self, in_dim=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x, y):
        return self.net(torch.cat([x, y], -1))


def complex_pairs(x, y, mask, max_samples=4096):
    """Flatten the symbols actually sent into (N, 2) real tensors for MINE."""
    xs, ys = x[mask], y[mask]
    if xs.numel() > max_samples:
        idx = torch.randperm(xs.numel(), device=xs.device)[:max_samples]
        xs, ys = xs[idx], ys[idx]
    return torch.stack([xs.real, xs.imag], -1), torch.stack([ys.real, ys.imag], -1)


def mi_lower_bound(mine, x, y):
    """L_MI = E_p(x,y)[T] - log E_p(x)p(y)[e^T]   (eq. (9)), in nats."""
    t_joint = mine(x, y).mean()
    y_marg = y[torch.randperm(y.size(0), device=y.device)]
    t_marg = mine(x, y_marg).squeeze(-1)
    return t_joint - (torch.logsumexp(t_marg, 0) - math.log(t_marg.numel()))

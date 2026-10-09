"""Traditional separate source / channel coding baselines (Sec. V-A).

    text --source coding--> bits --channel coding--> bits --QAM--> channel
         --demod--> channel decoding --> source decoding --> text

Source coding : Huffman (character level) or fixed-length 5-bit coding.
Channel coding: Reed-Solomon RS(n, k) over GF(2^8) (hard decision), or a
                rate-1/3 turbo code (two RSC(1, 5/7) encoders, random
                interleaver) decoded with iterative log-MAP (5 iterations).
Modulation    : Gray-mapped square M-QAM with unit average power.
Channel       : same AWGN / Rayleigh block-fading channel as DeepSC
                (perfect CSI, zero-forcing equalisation).
"""
import heapq
import math
from collections import Counter

import numpy as np
from reedsolo import ReedSolomonError, RSCodec

from .data import CHARSET


# ============================ source coding ================================ #
class FixedLengthCoder:
    """5-bit fixed length code for the 32-symbol character set."""

    def __init__(self, charset=CHARSET):
        self.charset = charset
        self.bits = math.ceil(math.log2(len(charset)))
        self.idx = {c: i for i, c in enumerate(charset)}

    def encode(self, text):
        out = []
        for ch in text:
            v = self.idx.get(ch, self.idx[" "])
            out.extend((v >> (self.bits - 1 - j)) & 1 for j in range(self.bits))
        return np.array(out, dtype=np.uint8)

    def decode(self, bits):
        n = len(bits) // self.bits
        b = np.asarray(bits[: n * self.bits]).reshape(n, self.bits)
        vals = b @ (1 << np.arange(self.bits - 1, -1, -1))
        return "".join(self.charset[v] for v in vals)


class HuffmanCoder:
    """Character level Huffman code built from the training corpus."""

    def __init__(self, texts):
        freq = Counter("".join(texts))
        heap = [[w, i, [ch, ""]] for i, (ch, w) in enumerate(freq.items())]
        heapq.heapify(heap)
        cnt = len(heap)
        while len(heap) > 1:
            lo, hi = heapq.heappop(heap), heapq.heappop(heap)
            for pair in lo[2:]:
                pair[1] = "0" + pair[1]
            for pair in hi[2:]:
                pair[1] = "1" + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0], cnt] + lo[2:] + hi[2:])
            cnt += 1
        self.code = {ch: c for ch, c in heap[0][2:]}
        self.inv = {c: ch for ch, c in self.code.items()}
        self.fallback = " "

    def encode(self, text):
        s = "".join(self.code.get(ch, self.code[self.fallback]) for ch in text)
        return np.frombuffer(s.encode(), dtype=np.uint8) - ord("0")

    def decode(self, bits):
        out, cur = [], ""
        for b in bits:
            cur += "1" if b else "0"
            if cur in self.inv:
                out.append(self.inv[cur])
                cur = ""
        return "".join(out)

    def avg_bits_per_char(self, texts):
        freq = Counter("".join(texts))
        return sum(freq[c] * len(self.code.get(c, "")) for c in freq) / sum(freq.values())


# ============================== modulation ================================= #
class QAM:
    """Gray-mapped square M-QAM, unit average symbol energy."""

    def __init__(self, m=64):
        self.m = m
        self.k = int(math.log2(m))
        assert self.k % 2 == 0, "square QAM only"
        side = int(math.sqrt(m))
        kb = self.k // 2
        gray = np.arange(side) ^ (np.arange(side) >> 1)
        levels = np.empty(side)
        levels[gray] = 2 * np.arange(side) - side + 1  # level for every Gray label
        pts = np.empty(m, dtype=complex)
        for idx in range(m):
            pts[idx] = levels[idx >> kb] + 1j * levels[idx & (side - 1)]
        self.points = pts / np.sqrt(np.mean(np.abs(pts) ** 2))
        self.labels = ((np.arange(m)[:, None] >> np.arange(self.k - 1, -1, -1)) & 1).astype(np.uint8)

    def modulate(self, bits):
        bits = np.asarray(bits, dtype=np.int64)
        pad = (-len(bits)) % self.k
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.int64)])
        idx = bits.reshape(-1, self.k) @ (1 << np.arange(self.k - 1, -1, -1))
        return self.points[idx]

    def demod_hard(self, y):
        idx = np.abs(y[:, None] - self.points[None]).argmin(1)
        return self.labels[idx].reshape(-1)

    def demod_llr(self, y, noise_var):
        """Exact LLRs log P(b=1|y)/P(b=0|y); noise_var = E|n|^2 per symbol (scalar or per symbol)."""
        nv = np.broadcast_to(np.asarray(noise_var, dtype=float), y.shape)[:, None]
        metric = -np.abs(y[:, None] - self.points[None]) ** 2 / nv  # (N, M)
        llr = np.empty((len(y), self.k))
        for b in range(self.k):
            one = self.labels[:, b] == 1
            llr[:, b] = _lse(metric[:, one], 1) - _lse(metric[:, ~one], 1)
        return llr.reshape(-1)


def _lse(a, axis):
    m = a.max(axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis, keepdims=True))).squeeze(axis)


# ================================ channel ================================== #
def channel(x, snr_db, kind, rng):
    """x: complex symbols of one sentence -> equalised y, per-symbol noise variance."""
    n0 = 10 ** (-snr_db / 10)  # E|n|^2 for unit-power symbols
    n = np.sqrt(n0 / 2) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    if kind == "awgn":
        return x + n, np.full(x.shape, n0)
    h = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
    return (h * x + n) / h, np.full(x.shape, n0 / abs(h) ** 2)


# ============================ Reed-Solomon ================================= #
def bits_to_bytes(bits):
    bits = np.concatenate([bits, np.zeros((-len(bits)) % 8, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


def bytes_to_bits(b, n_bits):
    return np.unpackbits(np.frombuffer(bytes(b), dtype=np.uint8))[:n_bits]


class ReedSolomon:
    def __init__(self, n, k):
        self.n, self.k = n, k
        self.codec = RSCodec(n - k, nsize=n)

    def encode(self, data: bytes) -> bytes:
        return bytes(self.codec.encode(data))

    def decode(self, data: bytes, msg_len: int) -> bytes:
        out, pos = bytearray(), 0
        while len(out) < msg_len:
            k = min(self.k, msg_len - len(out))
            block = bytes(data[pos: pos + k + self.n - self.k])
            pos += len(block)
            try:
                out += self.codec.decode(block)[0]
            except ReedSolomonError:
                out += block[:k]  # decoding failure: keep the systematic part
        return bytes(out)


# =============================== Turbo code ================================ #
class TurboCode:
    """Rate-1/3 parallel concatenated code with RSC(1, 5/7) constituents.

    Sentences of one batch are padded to a common block length T with zero
    bits that are known to the receiver and are not transmitted (shortened
    code), which lets the log-MAP decoder run vectorised over the batch.
    """

    BIG = 50.0

    def __init__(self, iterations=5, seed=0):
        self.iterations = iterations
        self.rng = np.random.default_rng(seed)
        # state = (a_{k-1}, a_{k-2}); feedback 1+D+D^2 (7), feed-forward 1+D^2 (5)
        self.next_state = np.zeros((4, 2), dtype=np.int64)
        self.parity = np.zeros((4, 2), dtype=np.int64)
        for s in range(4):
            a1, a2 = s >> 1, s & 1
            for u in range(2):
                a = u ^ a1 ^ a2
                self.parity[s, u] = a ^ a2
                self.next_state[s, u] = (a << 1) | a1

    def _rsc(self, u):  # u: (B, T)
        b, t = u.shape
        s = np.zeros(b, dtype=np.int64)
        p = np.zeros((b, t), dtype=np.uint8)
        for k in range(t):
            p[:, k] = self.parity[s, u[:, k]]
            s = self.next_state[s, u[:, k]]
        return p

    def interleaver(self, t):
        return self.rng.permutation(t)

    def encode(self, msgs, perm):
        """msgs: list of 0/1 arrays -> list of coded bit streams, layout info."""
        t = len(perm)
        u = np.zeros((len(msgs), t), dtype=np.int64)
        for i, m in enumerate(msgs):
            u[i, : len(m)] = m
        p1, p2 = self._rsc(u), self._rsc(u[:, perm])
        coded = []
        for i, m in enumerate(msgs):
            n = len(m)
            keep2 = perm < n
            coded.append(np.concatenate([u[i, :n], p1[i, :n], p2[i, keep2]]).astype(np.uint8))
        return coded

    def _log_map(self, ls, lp, la):
        """One constituent BCJR decoder in the log domain. All inputs (B, T)."""
        b, t = ls.shape
        ns, par = self.next_state, self.parity
        g = np.empty((b, t, 4, 2))
        for s in range(4):
            for u in range(2):
                g[:, :, s, u] = 0.5 * (2 * u - 1) * (ls + la) + 0.5 * (2 * par[s, u] - 1) * lp
        alpha = np.full((b, t + 1, 4), -1e9)
        alpha[:, 0, 0] = 0.0
        for k in range(t):
            cand = np.full((b, 4, 2), -1e9)  # two incoming branches per next state
            fill = np.zeros(4, dtype=np.int64)
            for s in range(4):
                for u in range(2):
                    s2 = ns[s, u]
                    cand[:, s2, fill[s2]] = alpha[:, k, s] + g[:, k, s, u]
                    fill[s2] += 1
            a = _lse(cand, 2)
            alpha[:, k + 1] = a - a.max(1, keepdims=True)
        beta = np.zeros((b, t + 1, 4))  # unterminated trellis
        for k in range(t - 1, -1, -1):
            cand = np.stack([beta[:, k + 1, ns[:, u]] + g[:, k, :, u] for u in range(2)], -1)
            bb = _lse(cand, 2)
            beta[:, k] = bb - bb.max(1, keepdims=True)
        llr = []
        for u in range(2):
            m = alpha[:, :-1, :] + g[:, :, :, u] + beta[:, 1:, ns[:, u]]
            llr.append(_lse(m, 2))
        return llr[1] - llr[0]

    def decode(self, llrs, lengths, perm):
        """llrs: list of channel LLR arrays (one per sentence, same layout as encode)."""
        t, bsz = len(perm), len(llrs)
        ls, lp1, lp2, lk = (np.zeros((bsz, t)) for _ in range(4))
        for i, (llr, n) in enumerate(zip(llrs, lengths)):
            keep2 = perm < n
            ls[i, :n], lp1[i, :n] = llr[:n], llr[n: 2 * n]
            lp2[i, keep2] = llr[2 * n: 2 * n + keep2.sum()]
            lk[i, n:] = -self.BIG  # known zero padding bits
        inv = np.argsort(perm)
        le2 = np.zeros((bsz, t))
        for _ in range(self.iterations):
            la1 = le2[:, inv]
            l1 = self._log_map(ls, lp1, la1 + lk)
            le1 = l1 - la1 - lk - ls
            la2 = le1[:, perm]
            l2 = self._log_map(ls[:, perm], lp2, la2 + lk[:, perm])
            le2 = l2 - la2 - lk[:, perm] - ls[:, perm]
        final = l2[:, inv]
        return [(final[i, :n] > 0).astype(np.uint8) for i, n in enumerate(lengths)]


# =========================== full baseline system ========================== #
class TraditionalSystem:
    """source coder + channel coder + QAM; transmits a batch of sentences."""

    def __init__(self, source, channel_code, qam_order=64, seed=0):
        self.source = source
        self.code = channel_code
        self.qam = QAM(qam_order)
        self.rng = np.random.default_rng(seed)

    def transmit(self, sentences, snr_db, kind="awgn"):
        msgs = [self.source.encode(s) for s in sentences]
        if isinstance(self.code, ReedSolomon):
            out, n_sym = [], []
            for m in msgs:
                raw = bits_to_bytes(m)
                cw = self.code.encode(raw)
                cbits = bytes_to_bits(cw, len(cw) * 8)
                x = self.qam.modulate(cbits)
                y, _ = channel(x, snr_db, kind, self.rng)
                rb = bits_to_bytes(self.qam.demod_hard(y)[: len(cbits)])
                dec = self.code.decode(rb, len(raw))
                out.append(self.source.decode(bytes_to_bits(dec, len(m))))
                n_sym.append(len(x))
            return out, n_sym
        # turbo
        perm = self.code.interleaver(max(len(m) for m in msgs))
        coded = self.code.encode(msgs, perm)
        llrs, n_sym = [], []
        for c in coded:
            x = self.qam.modulate(c)
            y, nv = channel(x, snr_db, kind, self.rng)
            llrs.append(self.qam.demod_llr(y, nv)[: len(c)])
            n_sym.append(len(x))
        dec = self.code.decode(llrs, [len(m) for m in msgs], perm)
        return [self.source.decode(d) for d in dec], n_sym

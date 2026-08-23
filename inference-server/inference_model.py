import os
import pickle
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gpt-from-scratch'))
from model import GPT


def _layer_norm(x, gamma, beta, eps=1e-5):
  mean = x.mean(axis=-1, keepdims=True)
  var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
  return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _softmax(x, axis=-1):
  x = x - np.max(x, axis=axis, keepdims=True)
  e = np.exp(x)
  return e / np.sum(e, axis=axis, keepdims=True)


class InferenceGPT:
  # weights are frozen at serving time, so the
  # training-time Tensor graph would just be dead weight here
  def __init__(self, checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
      ckpt = pickle.load(f)

    self.config = ckpt['config']
    model = GPT(**self.config)
    for p, saved in zip(model.parameters(), ckpt['state']):
      p.data = saved

    self.num_layers = self.config['num_layers']
    self.num_heads = self.config['num_heads']
    self.d_model = self.config['d_model']
    self.d_head = self.d_model // self.num_heads
    self.max_seq_len = self.config['max_seq_len']
    self.vocab_size = self.config['vocab_size']

    self.token_emb = model.token_emb.data
    self.pos_enc = model.pos_enc

    self.blocks = [{
      'wq': b.attn.query.w.data, 'wk': b.attn.key.w.data,
      'wv': b.attn.value.w.data, 'wo': b.attn.out_proj.w.data,
      'ln1_g': b.ln1.gamma.data, 'ln1_b': b.ln1.beta.data,
      'fc1_w': b.ff.fc1.w.data, 'fc1_b': b.ff.fc1.b.data,
      'fc2_w': b.ff.fc2.w.data, 'fc2_b': b.ff.fc2.b.data,
      'ln2_g': b.ln2.gamma.data, 'ln2_b': b.ln2.beta.data,
    } for b in model.blocks]

    self.ln_f_g = model.ln_f.gamma.data
    self.ln_f_b = model.ln_f.beta.data
    self.head_w = model.head.w.data

  def kv_bytes_per_token(self):
    # K and V, every layer, float32
    return self.num_layers * 2 * self.d_model * 4

  def _split_heads(self, x):
    T = x.shape[0]
    x = x.reshape(T, self.num_heads, self.d_head)
    return x.transpose(1, 0, 2)  # (H, T, Dh)

  def _merge_heads(self, x):
    H, T, Dh = x.shape
    return x.transpose(1, 0, 2).reshape(T, H * Dh)

  def _block_forward(self, x, blk, Kh, Vh):
    Qh = self._split_heads(x @ blk['wq'])
    scores = (Qh @ Kh.transpose(0, 2, 1)) / np.sqrt(self.d_head)
    if Qh.shape[1] > 1:
      # only prefill (multi-token query) needs a causal mask -- a single
      # decode step's query can always see the whole cache, itself included
      T = Qh.shape[1]
      mask = np.triu(np.ones((T, T), dtype=bool), k=1)
      scores = np.where(mask, -np.inf, scores)
    attn = _softmax(scores, axis=-1)
    out = self._merge_heads(attn @ Vh) @ blk['wo']

    x = _layer_norm(x + out, blk['ln1_g'], blk['ln1_b'])
    ff = np.maximum(0, x @ blk['fc1_w'] + blk['fc1_b']) @ blk['fc2_w'] + blk['fc2_b']
    x = _layer_norm(x + ff, blk['ln2_g'], blk['ln2_b'])
    return x

  def forward_all(self, token_ids):
    # logits at every position, no cache kept -- used for perplexity eval
    # (quantize.py), not for serving
    T = len(token_ids)
    assert T <= self.max_seq_len, 'sequence longer than max_seq_len'
    x = self.token_emb[token_ids] + self.pos_enc[:T]

    for blk in self.blocks:
      Kh = self._split_heads(x @ blk['wk'])
      Vh = self._split_heads(x @ blk['wv'])
      x = self._block_forward(x, blk, Kh, Vh)

    x = _layer_norm(x, self.ln_f_g, self.ln_f_b)
    return x @ self.head_w

  def prefill(self, token_ids):
    # token_ids: 1D int array (prompt, or prompt + already-generated tokens
    # on a recompute-after-eviction). Returns logits for the last position
    # and a fresh per-layer (K, V) cache covering the whole sequence.
    T = len(token_ids)
    assert T <= self.max_seq_len, 'sequence longer than max_seq_len'
    x = self.token_emb[token_ids] + self.pos_enc[:T]

    cache = []
    for blk in self.blocks:
      Kh = self._split_heads(x @ blk['wk'])
      Vh = self._split_heads(x @ blk['wv'])
      x = self._block_forward(x, blk, Kh, Vh)
      cache.append((Kh, Vh))

    x = _layer_norm(x, self.ln_f_g, self.ln_f_b)
    logits = x[-1] @ self.head_w
    return logits, cache

  def decode_step(self, token_id, cache, position):
    # cache: list of (K, V) per layer from a previous prefill/decode_step,
    # each (H, T_so_far, Dh). Appends this step's K/V and returns the grown cache.
    x = (self.token_emb[token_id] + self.pos_enc[position])[None, :]  # (1, D)

    new_cache = []
    for blk, (K_old, V_old) in zip(self.blocks, cache):
      Kh = np.concatenate([K_old, self._split_heads(x @ blk['wk'])], axis=1)
      Vh = np.concatenate([V_old, self._split_heads(x @ blk['wv'])], axis=1)
      x = self._block_forward(x, blk, Kh, Vh)
      new_cache.append((Kh, Vh))

    x = _layer_norm(x, self.ln_f_g, self.ln_f_b)
    logits = x[0] @ self.head_w
    return logits, new_cache

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'autograd'))
from tensor import Tensor
from attention import MultiHeadAttention, Linear


def sinusoidal_positional_encoding(max_seq_len, d_model):
  pos = np.arange(max_seq_len)[:, None]
  i = np.arange(d_model)[None, :]
  angle_rates = 1.0 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
  angles = pos * angle_rates

  pe = np.zeros((max_seq_len, d_model), dtype=np.float32)
  pe[:, 0::2] = np.sin(angles[:, 0::2])
  pe[:, 1::2] = np.cos(angles[:, 1::2])
  return pe


def embedding_lookup(table, indices):
  out = Tensor(table.data[indices], (table,), 'embedding')

  def _backward():
    np.add.at(table.grad, indices, out.grad)

  out._backward = _backward
  return out


def cross_entropy_loss(logits, targets):
  # logits: Tensor (B, T, vocab), targets: int array (B, T)
  B, T, V = logits.data.shape
  flat_logits = logits.data.reshape(B * T, V)
  flat_targets = targets.reshape(B * T)

  max_logits = np.max(flat_logits, axis=1, keepdims=True)
  exp_logits = np.exp(flat_logits - max_logits)
  probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

  loss_data = -np.log(probs[np.arange(B * T), flat_targets] + 1e-15).mean()
  out = Tensor(loss_data, (logits,), 'cross_entropy')

  def _backward():
    dlogits = probs.copy()
    dlogits[np.arange(B * T), flat_targets] -= 1.0
    dlogits /= (B * T)
    logits.grad += dlogits.reshape(B, T, V) * out.grad

  out._backward = _backward
  return out


def perplexity(loss):
  return float(np.exp(loss))


class LayerNorm:
  def __init__(self, d_model, eps=1e-5):
    self.eps = eps
    self.gamma = Tensor(np.ones(d_model))
    self.beta = Tensor(np.zeros(d_model))

  def __call__(self, x):
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    var = (centered * centered).mean(axis=-1, keepdims=True)
    normed = centered / ((var + self.eps) ** 0.5)
    return normed * self.gamma + self.beta

  def parameters(self):
    return [self.gamma, self.beta]


class FeedForward:
  def __init__(self, d_model, d_ff):
    self.fc1 = Linear(d_model, d_ff)
    self.fc2 = Linear(d_ff, d_model)

  def __call__(self, x):
    return self.fc2(self.fc1(x).relu())

  def parameters(self):
    return self.fc1.parameters() + self.fc2.parameters()


class TransformerBlock:
  def __init__(self, d_model, num_heads, d_ff, causal=True):
    self.attn = MultiHeadAttention(d_model, num_heads, causal=causal)
    self.ln1 = LayerNorm(d_model)
    self.ff = FeedForward(d_model, d_ff)
    self.ln2 = LayerNorm(d_model)

  def __call__(self, x):
    attn_out, attn_weights = self.attn(x)
    x = self.ln1(x + attn_out)
    x = self.ln2(x + self.ff(x))
    return x, attn_weights

  def parameters(self):
    params = self.attn.parameters() + self.ln1.parameters()
    params += self.ff.parameters() + self.ln2.parameters()
    return params


class GPT:
  def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=4, d_ff=512, max_seq_len=256):
    self.max_seq_len = max_seq_len
    self.token_emb = Tensor(np.random.randn(vocab_size, d_model) * 0.02)
    self.pos_enc = sinusoidal_positional_encoding(max_seq_len, d_model)  # constant, not trained

    self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
    self.ln_f = LayerNorm(d_model)
    self.head = Linear(d_model, vocab_size, bias=False)

  def __call__(self, idx):
    # idx: int array (B, T), token ids
    B, T = idx.shape
    assert T <= self.max_seq_len, 'sequence longer than max_seq_len'

    x = embedding_lookup(self.token_emb, idx) + Tensor(self.pos_enc[:T])

    attn_weights = []
    for block in self.blocks:
      x, w = block(x)
      attn_weights.append(w)

    x = self.ln_f(x)
    logits = self.head(x)
    return logits, attn_weights

  def parameters(self):
    params = [self.token_emb]
    for block in self.blocks:
      params += block.parameters()
    params += self.ln_f.parameters() + self.head.parameters()
    return params

  def zero_grad(self):
    for p in self.parameters():
      p.grad = np.zeros_like(p.data)

  def num_parameters(self):
    return sum(p.data.size for p in self.parameters())

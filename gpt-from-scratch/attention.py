import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'autograd'))
from tensor import Tensor

def causal_mask(seq_len):
  # True = block this position (upper triangle, strictly above the diagonal)
  return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

def softmax(t, axis=-1, mask=None):
  x = t.data
  if mask is not None:
    x = np.where(mask, -np.inf, x)

  x_max = np.max(x, axis=axis, keepdims=True)
  exp_x = np.exp(x - x_max)
  if mask is not None:
    exp_x = np.where(mask, 0.0, exp_x)
  probs = exp_x / np.sum(exp_x, axis=axis, keepdims=True)

  out = Tensor(probs, (t,), 'softmax')

  def _backward():
    dout = out.grad
    inner = np.sum(dout * probs, axis=axis, keepdims=True)
    t.grad += probs * (dout - inner)

  out._backward = _backward
  return out


class Linear:
  def __init__(self, nin, nout, bias=True):
    self.w = Tensor(np.random.randn(nin, nout) * np.sqrt(2.0 / nin))
    self.use_bias = bias
    if bias:
      self.b = Tensor(np.zeros((1, nout)))

  def __call__(self, x):
    out = x @ self.w
    if self.use_bias:
      out = out + self.b
    return out

  def parameters(self):
    return [self.w, self.b] if self.use_bias else [self.w]


def scaled_dot_product_attention(Q, K, V, mask=None):
  d_k = Q.data.shape[-1]
  scores = (Q @ K.transpose(-2, -1)) * (1.0 / np.sqrt(d_k))
  attn = softmax(scores, axis=-1, mask=mask)
  out = attn @ V
  return out, attn


class SelfAttention:
  # single-head attention, mainly here for milestone 1 (toy examples, hand-worked checks)
  def __init__(self, d_model, d_k, causal=False):
    self.causal = causal
    self.query = Linear(d_model, d_k, bias=False)
    self.key = Linear(d_model, d_k, bias=False)
    self.value = Linear(d_model, d_k, bias=False)

  def __call__(self, x):
    seq_len = x.data.shape[-2]
    mask = causal_mask(seq_len) if self.causal else None
    Q, K, V = self.query(x), self.key(x), self.value(x)
    return scaled_dot_product_attention(Q, K, V, mask=mask)

  def parameters(self):
    return self.query.parameters() + self.key.parameters() + self.value.parameters()


class MultiHeadAttention:
  def __init__(self, d_model, num_heads, causal=True):
    assert d_model % num_heads == 0, 'd_model must divide evenly into heads'
    self.num_heads = num_heads
    self.d_head = d_model // num_heads
    self.causal = causal

    self.query = Linear(d_model, d_model, bias=False)
    self.key = Linear(d_model, d_model, bias=False)
    self.value = Linear(d_model, d_model, bias=False)
    self.out_proj = Linear(d_model, d_model, bias=False)

  def _split_heads(self, x):
    B, T, D = x.data.shape
    x = x.reshape(B, T, self.num_heads, self.d_head)
    return x.transpose(1, 2)  # (B, num_heads, T, d_head)

  def _merge_heads(self, x):
    B, H, T, Dh = x.data.shape
    x = x.transpose(1, 2)  # (B, T, num_heads, d_head), contiguous concat of heads
    return x.reshape(B, T, H * Dh)

  def __call__(self, x):
    T = x.data.shape[-2]
    mask = causal_mask(T) if self.causal else None

    Q = self._split_heads(self.query(x))
    K = self._split_heads(self.key(x))
    V = self._split_heads(self.value(x))

    out, attn = scaled_dot_product_attention(Q, K, V, mask=mask)
    out = self._merge_heads(out)
    return self.out_proj(out), attn

  def parameters(self):
    params = self.query.parameters() + self.key.parameters()
    params += self.value.parameters() + self.out_proj.parameters()
    return params
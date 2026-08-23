import copy

import numpy as np


def quantize_tensor(w, axis=0):
  # symmetric per-channel int8: one scale per slice along `axis`, zero-point
  # fixed at 0 (weights are ~zero-centered, so no need for an asymmetric scheme)
  max_abs = np.max(np.abs(w), axis=axis, keepdims=True)
  scale = np.where(max_abs == 0, 1e-8, max_abs / 127.0)
  q = np.round(w / scale).astype(np.int8)
  return q, scale


def dequantize_tensor(q, scale):
  return q.astype(np.float32) * scale


# weight matrices worth quantizing -- biases and LayerNorm gain/bias are tiny
# and quantizing them buys almost nothing while adding real error, so they
# stay fp32
_QUANTIZABLE_BLOCK_KEYS = ('wq', 'wk', 'wv', 'wo', 'fc1_w', 'fc2_w')


def quantize_model(model):
  # returns (quantized_model, memory_report). the quantized model is a
  # "fake-quant" copy: weights are stored as int8+scale then immediately
  # dequantized back to fp32 arrays, so inference_model.py's forward code
  # doesn't need a separate int8 code path. this measures the accuracy cost
  # of quantization honestly; it does NOT give a real speedup in plain numpy
  # (that needs actual int8 GEMM kernels), so we report memory footprint and
  # perplexity, not latency, for this comparison -- exactly what the design
  # doc asks for.
  qmodel = copy.deepcopy(model)
  fp32_bytes = 0
  int8_bytes = 0

  for blk_orig, blk_q in zip(model.blocks, qmodel.blocks):
    for key in _QUANTIZABLE_BLOCK_KEYS:
      w = blk_orig[key]
      q, scale = quantize_tensor(w, axis=0)
      blk_q[key] = dequantize_tensor(q, scale)
      fp32_bytes += w.nbytes
      int8_bytes += q.nbytes + scale.nbytes

  for attr in ('head_w', 'token_emb'):
    w = getattr(model, attr)
    q, scale = quantize_tensor(w, axis=0)
    setattr(qmodel, attr, dequantize_tensor(q, scale))
    fp32_bytes += w.nbytes
    int8_bytes += q.nbytes + scale.nbytes

  report = {
    'fp32_bytes': fp32_bytes,
    'int8_bytes': int8_bytes,
    'compression_ratio': fp32_bytes / int8_bytes,
  }
  return qmodel, report


def perplexity_eval(model, token_ids, block_size, num_chunks=20, seed=0):
  rng = np.random.default_rng(seed)
  n = len(token_ids)
  losses = []

  for _ in range(num_chunks):
    start = int(rng.integers(0, n - block_size - 1))
    chunk = token_ids[start:start + block_size]
    targets = token_ids[start + 1:start + block_size + 1]

    logits = model.forward_all(chunk)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    log_probs = logits - max_logits - np.log(np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True))
    losses.append(-log_probs[np.arange(block_size), targets].mean())

  mean_loss = float(np.mean(losses))
  return mean_loss, float(np.exp(mean_loss))

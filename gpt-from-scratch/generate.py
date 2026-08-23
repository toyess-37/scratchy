import argparse
import gc
import pickle
import sys
import numpy as np

from model import GPT
from tokenizer import BPETokenizer, CharTokenizer

def _softmax_np(x):
  x = x - np.max(x)
  e = np.exp(x)
  return e / e.sum()


def sample_token(logits, strategy='greedy', temperature=1.0, top_k=None, top_p=None):
  logits = logits.astype(np.float64)

  if strategy == 'greedy':
    return int(np.argmax(logits))

  logits = logits / max(temperature, 1e-8)

  if strategy == 'top_k' and top_k is not None:
    k = min(top_k, logits.shape[-1])
    kth_val = np.partition(logits, -k)[-k]
    logits = np.where(logits < kth_val, -np.inf, logits)

  probs = _softmax_np(logits)

  if strategy == 'top_p' and top_p is not None:
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative, top_p) + 1
    keep = order[:cutoff]
    mask = np.ones_like(probs, dtype=bool)
    mask[keep] = False
    probs[mask] = 0.0
    probs /= probs.sum()

  return int(np.random.choice(len(probs), p=probs))


def generate(model, idx, max_new_tokens, strategy='greedy', temperature=1.0, top_k=None, top_p=None):
  # idx: (1, T) numpy array of token ids, extended one token at a time
  # no KV cache here on purpose -- that's added in the inference server (part 2)
  for step in range(max_new_tokens):
    context = idx[:, -model.max_seq_len:]
    logits, _ = model(context)
    next_logits = logits.data[0, -1]
    next_id = sample_token(next_logits, strategy=strategy, temperature=temperature, top_k=top_k, top_p=top_p)
    idx = np.concatenate([idx, [[next_id]]], axis=1)
    del logits
    if step % 20 == 0:
      gc.collect()
  return idx


def load_checkpoint(path):
  with open(path, 'rb') as f:
    ckpt = pickle.load(f)

  config = ckpt['config']
  model = GPT(**config)
  for p, saved in zip(model.parameters(), ckpt['state']):
    p.data = saved

  if ckpt['tokenizer_kind'] == 'bpe':
    tokenizer = BPETokenizer()
    tokenizer.merges = ckpt['tokenizer_state']['merges']
    tokenizer.vocab = ckpt['tokenizer_state']['vocab']
  else:
    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.stoi = ckpt['tokenizer_state']['stoi']
    tokenizer.itos = ckpt['tokenizer_state']['itos']
    tokenizer.vocab_size = len(tokenizer.stoi)

  return model, tokenizer


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--prompt', default='\n')
  parser.add_argument('--max_new_tokens', type=int, default=200)
  parser.add_argument('--strategy', choices=['greedy', 'temperature', 'top_k', 'top_p'], default='top_k')
  parser.add_argument('--temperature', type=float, default=0.8)
  parser.add_argument('--top_k', type=int, default=40)
  parser.add_argument('--top_p', type=float, default=0.9)
  args = parser.parse_args()

  sys.stdout.reconfigure(encoding='utf-8', errors='replace')
  model, tokenizer = load_checkpoint(args.checkpoint)
  idx = np.array([tokenizer.encode(args.prompt)])
  out = generate(model, idx, args.max_new_tokens, strategy=args.strategy,
                 temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
  print(tokenizer.decode(out[0].tolist()))


if __name__ == '__main__':
  main()
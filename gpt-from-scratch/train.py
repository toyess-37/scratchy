import argparse
import gc
import pickle
import sys
import time

import numpy as np

from model import GPT, cross_entropy_loss, perplexity
from tokenizer import BPETokenizer, CharTokenizer
from generate import generate, sample_token


class Adam:
  def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8):
    self.params = params
    self.lr = lr
    self.beta1, self.beta2 = betas
    self.eps = eps
    self.m = [np.zeros_like(p.data) for p in params]
    self.v = [np.zeros_like(p.data) for p in params]
    self.t = 0

  def step(self):
    self.t += 1
    for i, p in enumerate(self.params):
      self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
      self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
      m_hat = self.m[i] / (1 - self.beta1 ** self.t)
      v_hat = self.v[i] / (1 - self.beta2 ** self.t)
      p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

  def zero_grad(self):
    for p in self.params:
      p.grad = np.zeros_like(p.data)


def get_batch(data_ids, block_size, batch_size):
  ix = np.random.randint(0, len(data_ids) - block_size - 1, size=batch_size)
  x = np.stack([data_ids[i:i + block_size] for i in ix])
  y = np.stack([data_ids[i + 1:i + 1 + block_size] for i in ix])
  return x, y


def estimate_loss(model, data_ids, block_size, batch_size, eval_iters):
  losses = np.zeros(eval_iters)
  for i in range(eval_iters):
    x, y = get_batch(data_ids, block_size, batch_size)
    logits, _ = model(x)
    losses[i] = cross_entropy_loss(logits, y).data
  gc.collect()
  return float(losses.mean())


def save_checkpoint(path, model, config, tokenizer):
  if isinstance(tokenizer, BPETokenizer):
    tokenizer_kind = 'bpe'
    tokenizer_state = {'merges': tokenizer.merges, 'vocab': tokenizer.vocab}
  else:
    tokenizer_kind = 'char'
    tokenizer_state = {'stoi': tokenizer.stoi, 'itos': tokenizer.itos}

  ckpt = {
    'config': config,
    'state': [p.data for p in model.parameters()],
    'tokenizer_kind': tokenizer_kind,
    'tokenizer_state': tokenizer_state,
  }
  with open(path, 'wb') as f:
    pickle.dump(ckpt, f)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--corpus', default='data/tinyshakespeare.txt')
  parser.add_argument('--tokenizer', choices=['char', 'bpe'], default='char')
  parser.add_argument('--bpe_vocab_size', type=int, default=512)
  parser.add_argument('--block_size', type=int, default=64)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--d_model', type=int, default=128)
  parser.add_argument('--num_heads', type=int, default=4)
  parser.add_argument('--num_layers', type=int, default=4)
  parser.add_argument('--d_ff', type=int, default=512)
  parser.add_argument('--lr', type=float, default=3e-4)
  parser.add_argument('--max_iters', type=int, default=3000)
  parser.add_argument('--eval_interval', type=int, default=250)
  parser.add_argument('--eval_iters', type=int, default=50)
  parser.add_argument('--checkpoint', default='checkpoint.pkl')
  parser.add_argument('--seed', type=int, default=1337)
  args = parser.parse_args()

  sys.stdout.reconfigure(encoding='utf-8', errors='replace')
  np.random.seed(args.seed)
  text = open(args.corpus, encoding='utf-8').read()

  if args.tokenizer == 'bpe':
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=args.bpe_vocab_size)
  else:
    tokenizer = CharTokenizer(text)

  data_ids = np.array(tokenizer.encode(text))
  split = int(0.9 * len(data_ids))
  train_ids, val_ids = data_ids[:split], data_ids[split:]

  config = dict(
    vocab_size=tokenizer.vocab_size,
    d_model=args.d_model,
    num_heads=args.num_heads,
    num_layers=args.num_layers,
    d_ff=args.d_ff,
    max_seq_len=args.block_size,
  )
  model = GPT(**config)
  print(f'model parameters: {model.num_parameters():,}')

  optimizer = Adam(model.parameters(), lr=args.lr)

  start = time.time()
  for it in range(1, args.max_iters + 1):
    x, y = get_batch(train_ids, args.block_size, args.batch_size)

    logits, _ = model(x)
    loss = cross_entropy_loss(logits, y)

    model.zero_grad()
    loss.backward()
    optimizer.step()

    # each forward/backward pass builds a graph full of closures with reference
    # cycles (a Tensor's _backward closes over itself); refcounting alone won't
    # free them, so nudge the cyclic collector along instead of piling up memory
    train_loss = loss.data
    del logits, loss, x, y
    if it % 20 == 0:
      gc.collect()

    if it % args.eval_interval == 0 or it == args.max_iters:
      val_loss = estimate_loss(model, val_ids, args.block_size, args.batch_size, args.eval_iters)
      elapsed = time.time() - start
      print(f'iter {it}/{args.max_iters} | train_loss {train_loss:.4f} | '
            f'val_loss {val_loss:.4f} | val_ppl {perplexity(val_loss):.2f} | {elapsed:.1f}s')

      prompt = np.array([[train_ids[0]]])
      sample_ids = generate(model, prompt, max_new_tokens=100, strategy='top_k', top_k=40)[0].tolist()
      print('sample:', repr(tokenizer.decode(sample_ids)))
      gc.collect()

  save_checkpoint(args.checkpoint, model, config, tokenizer)
  print(f'saved checkpoint to {args.checkpoint}')


if __name__ == '__main__':
  main()

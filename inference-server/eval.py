import argparse
import pickle
import sys

import numpy as np

from inference_model import InferenceGPT
from quantize import perplexity_eval, quantize_model


def load_tokenizer(checkpoint_path):
  sys.path.append('../gpt-from-scratch')
  from tokenizer import BPETokenizer, CharTokenizer

  with open(checkpoint_path, 'rb') as f:
    ckpt = pickle.load(f)

  if ckpt['tokenizer_kind'] == 'bpe':
    tokenizer = BPETokenizer()
    tokenizer.merges = ckpt['tokenizer_state']['merges']
    tokenizer.vocab = ckpt['tokenizer_state']['vocab']
  else:
    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.stoi = ckpt['tokenizer_state']['stoi']
    tokenizer.itos = ckpt['tokenizer_state']['itos']
    tokenizer.vocab_size = len(tokenizer.stoi)

  return tokenizer


def print_report(name, loss, ppl):
  print(f'{name} loss={loss:.4f}  perplexity={ppl:.2f}')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', default='model_ckpt.pkl')
  parser.add_argument('--corpus', default='../gpt-from-scratch/data/tinyshakespeare.txt')
  parser.add_argument('--block_size', type=int, default=64)
  parser.add_argument('--num_chunks', type=int, default=20)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()

  tokenizer = load_tokenizer(args.checkpoint)
  model = InferenceGPT(args.checkpoint)

  text = open(args.corpus, encoding='utf-8').read()
  token_ids = np.array(tokenizer.encode(text))

  block_size = min(args.block_size, model.max_seq_len)

  fp32_loss, fp32_ppl = perplexity_eval(model, token_ids, block_size,
                                         num_chunks=args.num_chunks, seed=args.seed)
  print_report('fp32', fp32_loss, fp32_ppl)

  qmodel, report = quantize_model(model)
  q_loss, q_ppl = perplexity_eval(qmodel, token_ids, block_size,
                                   num_chunks=args.num_chunks, seed=args.seed)
  print_report('int8', q_loss, q_ppl)

  print(f'\ncompression_ratio={report["compression_ratio"]:.2f}x  '
        f'fp32_bytes={report["fp32_bytes"]:,}  int8_bytes={report["int8_bytes"]:,}')
  print(f'perplexity delta (int8 - fp32) = {q_ppl - fp32_ppl:+.2f}')


if __name__ == '__main__':
  main()

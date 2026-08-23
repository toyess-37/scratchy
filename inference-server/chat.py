import argparse
import sys

import numpy as np

from eval import load_tokenizer
from inference_model import InferenceGPT

sys.path.append('../gpt-from-scratch')
from generate import sample_token


def generate_reply(model, prompt_ids, max_new_tokens, strategy, temperature, top_k, top_p):
  # room left in the model's fixed-size positional table -- decode_step
  # indexes pos_enc[position] with no wraparound or windowing, so going past
  # max_seq_len would index out of bounds
  room = model.max_seq_len - len(prompt_ids)
  max_new_tokens = max(0, min(max_new_tokens, room))

  logits, cache = model.prefill(np.array(prompt_ids))
  position = len(prompt_ids)
  generated = []

  for _ in range(max_new_tokens):
    next_id = sample_token(logits, strategy=strategy, temperature=temperature, top_k=top_k, top_p=top_p)
    generated.append(next_id)
    logits, cache = model.decode_step(next_id, cache, position)
    position += 1

  return generated


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', default='model_ckpt.pkl')
  parser.add_argument('--max_new_tokens', type=int, default=200)
  parser.add_argument('--strategy', choices=['greedy', 'temperature', 'top_k', 'top_p'], default='top_k')
  parser.add_argument('--temperature', type=float, default=0.8)
  parser.add_argument('--top_k', type=int, default=40)
  parser.add_argument('--top_p', type=float, default=0.9)
  args = parser.parse_args()

  tokenizer = load_tokenizer(args.checkpoint)
  model = InferenceGPT(args.checkpoint)

  print(f"loaded {args.checkpoint} -- context window is {model.max_seq_len} tokens.")
  print("this is a raw char-level model trained on Shakespeare, not an instruction-tuned")
  print("assistant: it continues whatever text you give it in that style, it doesn't answer")
  print("questions. type 'quit' to exit.\n")

  while True:
    try:
      prompt = input('> ')
    except (EOFError, KeyboardInterrupt):
      print()
      break

    if prompt.strip().lower() in ('quit', 'exit'):
      break

    try:
      prompt_ids = tokenizer.encode(prompt)
    except KeyError as e:
      print(f"(character {e} isn't in this model's vocabulary, try something else)\n")
      continue

    if not prompt_ids:
      print('(type something first)\n')
      continue

    prompt_ids = prompt_ids[-model.max_seq_len:]
    if len(prompt_ids) >= model.max_seq_len:
      print(f"(prompt fills the whole {model.max_seq_len}-token context window -- "
            f"no room left to generate, try a shorter prompt)\n")
      continue

    generated = generate_reply(model, prompt_ids, args.max_new_tokens,
                                args.strategy, args.temperature, args.top_k, args.top_p)
    print(tokenizer.decode(generated))
    print()


if __name__ == '__main__':
  main()

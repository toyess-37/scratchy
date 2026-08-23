import json


class CharTokenizer:
  # starter tokenizer, one token per character
  def __init__(self, text):
    chars = sorted(set(text))
    self.stoi = {ch: i for i, ch in enumerate(chars)}
    self.itos = {i: ch for ch, i in self.stoi.items()}
    self.vocab_size = len(chars)

  def encode(self, text):
    return [self.stoi[c] for c in text]

  def decode(self, ids):
    return ''.join(self.itos[i] for i in ids)


class BPETokenizer:
  # byte-level BPE, GPT-2 style, trained from scratch (no tiktoken)
  def __init__(self):
    self.merges = {}                            # (id, id) -> new_id, in learn order
    self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes

  @property
  def vocab_size(self):
    return len(self.vocab)

  @staticmethod
  def _pair_counts(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
      counts[pair] = counts.get(pair, 0) + 1
    return counts

  @staticmethod
  def _merge(ids, pair, new_id):
    merged = []
    i = 0
    while i < len(ids):
      if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
        merged.append(new_id)
        i += 2
      else:
        merged.append(ids[i])
        i += 1
    return merged

  def train(self, text, vocab_size, verbose=False):
    assert vocab_size >= 256, 'vocab_size must cover the base 256 byte values'
    ids = list(text.encode('utf-8'))
    num_merges = vocab_size - 256

    for i in range(num_merges):
      counts = self._pair_counts(ids)
      if not counts:
        break

      top_pair = max(counts, key=counts.get)
      new_id = 256 + i
      ids = self._merge(ids, top_pair, new_id)
      self.merges[top_pair] = new_id
      self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

      if verbose:
        print(f'merge {i + 1}/{num_merges}: {top_pair} -> {new_id} ({self.vocab[new_id]})')

  def encode(self, text):
    ids = list(text.encode('utf-8'))
    while len(ids) >= 2:
      counts = self._pair_counts(ids)
      pair = min(counts, key=lambda p: self.merges.get(p, float('inf')))
      if pair not in self.merges:
        break
      ids = self._merge(ids, pair, self.merges[pair])
    return ids

  def decode(self, ids):
    raw = b''.join(self.vocab[i] for i in ids)
    return raw.decode('utf-8', errors='replace')

  def save(self, path):
    data = {'merges': [[list(pair), new_id] for pair, new_id in self.merges.items()]}
    with open(path, 'w') as f:
      json.dump(data, f)

  def load(self, path):
    with open(path) as f:
      data = json.load(f)

    self.merges = {tuple(pair): new_id for pair, new_id in data['merges']}
    self.vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in sorted(self.merges.items(), key=lambda kv: kv[1]):
      self.vocab[new_id] = self.vocab[a] + self.vocab[b]

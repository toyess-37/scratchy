import numpy as np


class Agent:
  # one simulated agentic session: a prompt, a total token budget to
  # generate, and a chance of "pausing" after each token (standing in for a
  # tool call / retrieval stall while its KV cache sits idle)
  def __init__(self, agent_id, prompt_ids, total_new_tokens, pause_prob, pause_ticks_range, rng):
    self.id = agent_id
    self.prompt_ids = prompt_ids
    self.generated_ids = []
    self.tokens_remaining = total_new_tokens
    self.pause_prob = pause_prob
    self.pause_ticks_range = pause_ticks_range
    self.rng = rng

    self.arrival_tick = 0
    self.cache_layers = None
    self.position = 0
    self.last_token = None
    self.pause_remaining = 0
    self.evicted = False
    self.recompute_count = 0

    self.arrival_wallclock = None
    self.completion_wallclock = None

  def maybe_pause(self):
    if self.tokens_remaining > 0 and self.rng.random() < self.pause_prob:
      lo, hi = self.pause_ticks_range
      self.pause_remaining = int(self.rng.integers(lo, hi + 1))
      return True
    return False

  def full_context_ids(self):
    return self.prompt_ids + self.generated_ids


def generate_workload(num_agents, arrival_rate_per_tick, vocab_size,
                       prompt_len_range=(4, 12), total_tokens_range=(20, 60),
                       pause_prob=0.15, pause_ticks_range=(3, 15), seed=0,
                       max_context_len=None):
  # arrival_rate_per_tick: probability an inter-arrival gap is short (geometric
  # distribution), i.e. higher = agents show up faster.
  # max_context_len: model's max_seq_len -- a recompute-after-eviction replays
  # prompt + everything generated so far in one shot, so no agent's full
  # context (prompt + total_new_tokens) may exceed it.
  rng = np.random.default_rng(seed)
  agents = []
  tick = 0

  for i in range(num_agents):
    tick += int(rng.geometric(arrival_rate_per_tick))
    prompt_len = int(rng.integers(*prompt_len_range))
    prompt_ids = rng.integers(0, vocab_size, size=prompt_len).tolist()
    total_new_tokens = int(rng.integers(*total_tokens_range))
    if max_context_len is not None:
      total_new_tokens = min(total_new_tokens, max_context_len - prompt_len)

    agent = Agent(
      agent_id=i,
      prompt_ids=prompt_ids,
      total_new_tokens=total_new_tokens,
      pause_prob=pause_prob,
      pause_ticks_range=pause_ticks_range,
      rng=rng,
    )
    agent.arrival_tick = tick
    agents.append(agent)

  return agents


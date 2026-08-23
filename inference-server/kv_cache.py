from collections import OrderedDict


class KVCache:
  # tracks how many token-slots each agent currently occupies, in recency
  # order, so it can answer "who's the LRU entry" for eviction. the actual
  # K/V arrays live on the agent (server.py) -- this is just the ledger.
  def __init__(self, budget_tokens):
    self.budget_tokens = budget_tokens
    self.lengths = OrderedDict()  # agent_id -> token length, LRU-ordered

  def put(self, agent_id, length):
    self.lengths[agent_id] = length
    self.lengths.move_to_end(agent_id)

  def touch(self, agent_id, length=None):
    if length is not None:
      self.lengths[agent_id] = length
    self.lengths.move_to_end(agent_id)

  def drop(self, agent_id):
    self.lengths.pop(agent_id, None)

  def usage_tokens(self):
    return sum(self.lengths.values())

  def usage_ratio(self):
    return self.usage_tokens() / self.budget_tokens

  def evict_if_needed(self, evictable_ids):
    # evict LRU-first among evictable_ids until back under budget (or out of
    # evictable entries). agents mid-generation are never in evictable_ids --
    # evicting an active agent's cache would corrupt its own output.
    evictable = set(evictable_ids)
    evicted = []
    for agent_id in list(self.lengths.keys()):
      if self.usage_tokens() <= self.budget_tokens:
        break
      if agent_id in evictable:
        self.lengths.pop(agent_id)
        evicted.append(agent_id)
    return evicted

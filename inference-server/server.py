import time
from collections import deque

import numpy as np


class InferenceServer:
  # a discrete-tick, continuous-batching-style loop: every tick, every
  # admitted+active agent advances by exactly one generated token (a real
  # forward pass, not a simulated delay), so heavier concurrency genuinely
  # costs more wall-clock time per tick -- the latency/throughput tradeoff
  # this benchmark measures falls out of real compute, not fabricated numbers
  def __init__(self, model, cache, scheduler, hit_rate_window=50):
    self.model = model
    self.cache = cache
    self.scheduler = scheduler

    self.active = {}
    self.paused = {}
    self.pending_queue = deque()
    self.completed = []

    self.tick = 0
    self.resume_events = deque(maxlen=hit_rate_window)  # True = cache still warm, False = had to recompute

  def submit(self, agent):
    if agent.arrival_wallclock is None:
      agent.arrival_wallclock = time.perf_counter()
    self.pending_queue.append(agent)

  def num_in_flight(self):
    return len(self.active) + len(self.paused)

  def _cache_state(self):
    usage = self.cache.usage_ratio()
    hit_rate = (sum(self.resume_events) / len(self.resume_events)) if self.resume_events else 1.0
    return {'usage_ratio': usage, 'hit_rate': hit_rate}

  def _prefill(self, agent):
    logits, cache = self.model.prefill(np.array(agent.prompt_ids))
    agent.cache_layers = cache
    agent.position = len(agent.prompt_ids)
    self._commit_token(agent, logits)
    self.cache.put(agent.id, agent.position)

  def _decode_one(self, agent):
    logits, agent.cache_layers = self.model.decode_step(agent.last_token, agent.cache_layers, agent.position)
    agent.position += 1
    self._commit_token(agent, logits)
    self.cache.touch(agent.id, agent.position)

  def _commit_token(self, agent, logits):
    next_id = int(np.argmax(logits))  # greedy: keeps the benchmark deterministic
    agent.generated_ids.append(next_id)
    agent.last_token = next_id
    agent.tokens_remaining -= 1

  def _resume(self, agent):
    if agent.evicted:
      logits, cache = self.model.prefill(np.array(agent.full_context_ids()))
      agent.cache_layers = cache
      agent.position = len(agent.full_context_ids())
      self._commit_token(agent, logits)
      agent.evicted = False
      agent.recompute_count += 1
      self.cache.put(agent.id, agent.position)
      self.resume_events.append(False)
    else:
      self.cache.touch(agent.id, agent.position)
      self.resume_events.append(True)

  def _finish(self, agent):
    agent.completion_wallclock = time.perf_counter()
    self.completed.append(agent)
    self.cache.drop(agent.id)

  def step(self):
    self.tick += 1
    self.scheduler.on_step(self._cache_state())

    to_decode = list(self.active.keys())
    while self.pending_queue and self.scheduler.admit(self.num_in_flight()):
      agent = self.pending_queue.popleft()
      self._prefill(agent)
      self.active[agent.id] = agent
      if agent.tokens_remaining <= 0:
        self._finish(agent)
        del self.active[agent.id]

    # only decode agents that were already active before this tick's
    # admissions -- prefill already advanced newly admitted agents by one
    # token this tick, so decoding them too would give them two tokens of
    # progress in their first tick instead of one
    for agent_id in to_decode:
      if agent_id not in self.active:
        continue
      agent = self.active[agent_id]
      self._decode_one(agent)
      if agent.tokens_remaining <= 0:
        self._finish(agent)
        del self.active[agent_id]
      elif agent.maybe_pause():
        self.paused[agent_id] = agent
        del self.active[agent_id]

    for agent_id in list(self.paused.keys()):
      agent = self.paused[agent_id]
      agent.pause_remaining -= 1
      if agent.pause_remaining <= 0:
        self._resume(agent)
        del self.paused[agent_id]
        if agent.tokens_remaining <= 0:
          self._finish(agent)
        else:
          self.active[agent_id] = agent

    evicted_ids = self.cache.evict_if_needed(list(self.paused.keys()))
    for agent_id in evicted_ids:
      self.paused[agent_id].evicted = True


# -- thin HTTP wrapper -------------------------------------------------------
# benchmark.py talks to InferenceServer directly (no need to pay HTTP
# overhead for a latency benchmark); this is here to satisfy "a server has
# an HTTP interface" and to let you poke at it by hand with curl.
try:
  from fastapi import FastAPI
  from pydantic import BaseModel

  class SubmitAgentRequest(BaseModel):
    prompt_ids: list[int]
    total_new_tokens: int
    pause_prob: float = 0.15
    pause_ticks_min: int = 3
    pause_ticks_max: int = 15

  def build_app(server: InferenceServer):
    app = FastAPI()

    @app.post('/agents')
    def submit_agent(req: SubmitAgentRequest):
      agent_id = server.tick * 100000 + len(server.pending_queue)
      rng = np.random.default_rng(agent_id)
      from workload_sim import Agent
      pause_ticks_range = (req.pause_ticks_min, req.pause_ticks_max)
      agent = Agent(agent_id, req.prompt_ids, req.total_new_tokens, req.pause_prob, pause_ticks_range, rng)
      server.submit(agent)
      return {'agent_id': agent_id}

    @app.post('/step')
    def step():
      server.step()
      return {'tick': server.tick, **server._cache_state(), 'capacity': server.scheduler.current_capacity()}

    @app.get('/stats')
    def stats():
      return {
        'tick': server.tick,
        'active': len(server.active),
        'paused': len(server.paused),
        'pending': len(server.pending_queue),
        'completed': len(server.completed),
        'cache': server._cache_state(),
        'capacity': server.scheduler.current_capacity(),
      }

    return app

except ImportError:
  build_app = None

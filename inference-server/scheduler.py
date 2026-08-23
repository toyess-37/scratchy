from abc import ABC, abstractmethod


class Scheduler(ABC):
  # small interface on purpose: benchmark.py just swaps the instance and
  # everything else (server, workload, metrics) stays the same

  @abstractmethod
  def admit(self, num_active):
    # num_active: agents currently holding a cache slot (active + paused)
    raise NotImplementedError

  def on_step(self, cache_state):
    # cache_state: {'usage_ratio': float, 'hit_rate': float}, called once per tick
    pass

  def current_capacity(self):
    raise NotImplementedError


class UnboundedScheduler(Scheduler):
  # no admission control at all -- the "uncontrolled" baseline
  def admit(self, num_active):
    return True

  def current_capacity(self):
    return float('inf')


class StaticScheduler(Scheduler):
  # fixed max-concurrency baseline (what most mini vLLM clones do)
  def __init__(self, max_concurrency):
    self.max_concurrency = max_concurrency

  def admit(self, num_active):
    return num_active < self.max_concurrency

  def current_capacity(self):
    return self.max_concurrency


class AIMDScheduler(Scheduler):
  # CONCUR-style control law (arXiv:2601.22705, eq. 1):
  #   W += alpha              if U < u_low                     (additive increase)
  #   W *= beta                if U > u_high and H < h_thresh   (multiplicative decrease)
  #   W unchanged              otherwise
  # U = live KV-cache usage ratio, H = recent cache hit rate (fraction of
  # agent resumes that didn't need a recompute). defaults below are the
  # paper's own reported values.
  def __init__(self, alpha=2.0, beta=0.5, u_low=0.2, u_high=0.5, h_thresh=0.2,
               init_window=4.0, min_window=1.0, max_window=None):
    self.alpha = alpha
    self.beta = beta
    self.u_low = u_low
    self.u_high = u_high
    self.h_thresh = h_thresh
    self.min_window = min_window
    self.max_window = max_window
    self.window = init_window
    self.history = []

  def on_step(self, cache_state):
    u = cache_state['usage_ratio']
    h = cache_state['hit_rate']

    if u < self.u_low:
      self.window += self.alpha
    elif u > self.u_high and h < self.h_thresh:
      self.window *= self.beta

    self.window = max(self.min_window, self.window)
    if self.max_window is not None:
      self.window = min(self.max_window, self.window)
    self.history.append(self.window)

  def admit(self, num_active):
    return num_active < self.window

  def current_capacity(self):
    return self.window

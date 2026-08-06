import time
import tracemalloc
import numpy as np

class Profiler:
  # execution time and peak RAM usage
  def __enter__(self):
    tracemalloc.start()
    self.start_time = time.perf_counter()
    return self

  def __exit__(self, exc_type, exc, tb):
    self.elapsed_time = time.perf_counter() - self.start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    self.peak_ram_mb = peak / (1024 * 1024)

def get_batches(X, y, batch_size=64):
  num_samples = len(X)
  indices = np.arange(num_samples)
  np.random.shuffle(indices)

  for i in range(0, num_samples, batch_size):
    batch_idx = indices[i:i+batch_size]
    yield X[batch_idx], y[batch_idx]
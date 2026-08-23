import argparse
import time

import numpy as np

from inference_model import InferenceGPT
from kv_cache import KVCache
from scheduler import AIMDScheduler, StaticScheduler, UnboundedScheduler
from server import InferenceServer
from workload_sim import generate_workload


def run_simulation(model, scheduler, agents, budget_tokens, max_ticks=20000):
  cache = KVCache(budget_tokens)
  server = InferenceServer(model, cache, scheduler)

  arrivals = sorted(agents, key=lambda a: a.arrival_tick)
  arrival_idx = 0
  usage_history = []
  capacity_history = []

  wall_start = time.perf_counter()
  for tick in range(max_ticks):
    while arrival_idx < len(arrivals) and arrivals[arrival_idx].arrival_tick <= tick:
      server.submit(arrivals[arrival_idx])
      arrival_idx += 1

    server.step()
    usage_history.append(cache.usage_ratio())
    capacity_history.append(scheduler.current_capacity())

    drained = arrival_idx >= len(arrivals) and not server.active and not server.paused and not server.pending_queue
    if drained:
      break
  wall_elapsed = time.perf_counter() - wall_start

  latencies = np.array([a.completion_wallclock - a.arrival_wallclock for a in server.completed])
  recomputes = sum(a.recompute_count for a in server.completed)

  return {
    'completed': len(server.completed),
    'ticks': server.tick,
    'wall_seconds': wall_elapsed,
    'throughput_req_s': len(server.completed) / wall_elapsed if wall_elapsed > 0 else 0.0,
    'latency_p50': float(np.percentile(latencies, 50)) if len(latencies) else float('nan'),
    'latency_p95': float(np.percentile(latencies, 95)) if len(latencies) else float('nan'),
    'latency_p99': float(np.percentile(latencies, 99)) if len(latencies) else float('nan'),
    'recomputes': recomputes,
    'usage_history': usage_history,
    'capacity_history': capacity_history,
  }


def make_workload(model, num_agents, arrival_rate, pause_prob, seed):
  return generate_workload(
    num_agents=num_agents,
    arrival_rate_per_tick=arrival_rate,
    vocab_size=model.vocab_size,
    prompt_len_range=(4, 12),
    total_tokens_range=(20, 60),
    pause_prob=pause_prob,
    pause_ticks_range=(3, 15),
    seed=seed,
    max_context_len=model.max_seq_len,
  )


def print_row(name, result):
  print(f"{name:<22} completed={result['completed']:>4}  "
        f"p50={result['latency_p50']*1000:7.2f}ms  p95={result['latency_p95']*1000:7.2f}ms  "
        f"p99={result['latency_p99']*1000:7.2f}ms  throughput={result['throughput_req_s']:6.2f}/s  "
        f"recomputes={result['recomputes']:>3}")


def compare_schedulers(args):
  model = InferenceGPT(args.checkpoint)

  configs = {
    'uncontrolled': lambda: UnboundedScheduler(),
    'static': lambda: StaticScheduler(max_concurrency=args.static_concurrency),
    'aimd': lambda: AIMDScheduler(alpha=args.alpha, beta=args.beta,
                                   u_low=args.u_low, u_high=args.u_high, h_thresh=args.h_thresh),
  }

  print(f'budget_tokens={args.budget_tokens}  num_agents={args.num_agents}  '
        f'arrival_rate={args.arrival_rate}  pause_prob={args.pause_prob}\n')

  for name, factory in configs.items():
    agents = make_workload(model, args.num_agents, args.arrival_rate, args.pause_prob, args.seed)
    result = run_simulation(model, factory(), agents, args.budget_tokens)
    print_row(name, result)


def sweep_thresholds(args):
  model = InferenceGPT(args.checkpoint)
  print('\nU_low / U_high sensitivity sweep (AIMD):')
  for u_low, u_high in [(0.1, 0.3), (0.2, 0.5), (0.3, 0.6), (0.4, 0.8)]:
    agents = make_workload(model, args.num_agents, args.arrival_rate, args.pause_prob, args.seed)
    sched = AIMDScheduler(alpha=args.alpha, beta=args.beta, u_low=u_low, u_high=u_high, h_thresh=args.h_thresh)
    result = run_simulation(model, sched, agents, args.budget_tokens)
    print_row(f'u_low={u_low} u_high={u_high}', result)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--budget_tokens', type=int, default=200)
  parser.add_argument('--num_agents', type=int, default=60)
  parser.add_argument('--arrival_rate', type=float, default=0.3)
  parser.add_argument('--pause_prob', type=float, default=0.2)
  parser.add_argument('--static_concurrency', type=int, default=4)
  parser.add_argument('--alpha', type=float, default=2.0)
  parser.add_argument('--beta', type=float, default=0.5)
  parser.add_argument('--u_low', type=float, default=0.2)
  parser.add_argument('--u_high', type=float, default=0.5)
  parser.add_argument('--h_thresh', type=float, default=0.2)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--sweep', action='store_true')
  args = parser.parse_args()

  compare_schedulers(args)
  if args.sweep:
    sweep_thresholds(args)


if __name__ == '__main__':
  main()

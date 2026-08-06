from value import Value
from tensor import Tensor
import numpy as np

def softmax_cross_entropy(logits, target):
  # for mnist, 0 to 9 prediction only
  maxi = max(logit.data for logit in logits)
  exp_logits = [(logit - maxi).exp() for logit in logits]
  sum_exp = sum(exp_logits)

  target_exp = exp_logits[target]
  log_prob = target_exp.log() - sum_exp.log()

  return -log_prob

# Test
# logits = [Value(2.0), Value(1.0), Value(0.1)]
# target = 0
#
# loss = softmax_cross_entropy(logits, target)
# loss.backward()
#
# print(f"Loss: {loss.data:.4f}")
# print("Gradient on logits:")
# for i, l in enumerate(logits):
#   print(f"Logit {i} grad: {l.grad:.4f}")

def vectorized_softmax_cross_entropy(logits, targets):
  # logits tensor of shape (batch_size, 10)
  # targets numpy array of shape (batch_size,)

  max_logits = np.max(logits.data, axis=1, keepdims=True)
  exp_logits = np.exp(logits.data - max_logits)
  probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

  batch_size = logits.data.shape[0]
  loss_data = -np.log(probs[np.arange(batch_size), targets]+1e-15).mean()

  out = Tensor(loss_data, (logits,), 'softmax_ce')

  def _backward():
    dlogits = probs.copy()
    dlogits[np.arange(batch_size), targets] -= 1.0
    dlogits /= batch_size
    logits.grad += dlogits * out.grad

  out._backward = _backward
  return out
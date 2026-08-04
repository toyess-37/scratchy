from value import Value

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
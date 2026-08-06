import numpy as np
from tensor import Tensor
from losses import vectorized_softmax_cross_entropy

class Trainer:
  def __init__(self, model, lr=0.1):
    self.model = model
    self.lr = lr

  def train_epoch(self, X_train, y_train, get_batches_fn, batch_size=64):
    total_loss = 0.0
    num_batches = 0.0

    for X_batch, y_batch in get_batches_fn(X_train, y_train, batch_size=batch_size):
      x = Tensor(X_batch)
      logits = self.model(x)
      loss = vectorized_softmax_cross_entropy(logits, y_batch)

      self.model.zero_grad()
      loss.backward()

      for p in self.model.parameters():
        p.data -= self.lr * p.grad

      total_loss += loss.data
      num_batches += 1

    return total_loss / num_batches

  def evaluate(self, X_val,  y_val):
    val_x = Tensor(X_val)
    val_logits = self.model(val_x)
    preds = np.argmax(val_logits.data, axis=1)
    return np.mean(preds == y_val) * 100.0
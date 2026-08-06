from sklearn.datasets import fetch_openml
import numpy as np
import random

# from value import Value
# from nn import MLP
# from losses import softmax_cross_entropy

from tensor import Tensor
from nn_tensor import MLP
from losses import vectorized_softmax_cross_entropy

print("Fetching MNIST:")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_raw, y_raw = mnist.data, mnist.target.astype(int)

X_norm = X_raw / 255.0

def get_batches(X, y, batch_size=32):
  num_samples = len(X)
  indices = list(range(num_samples))
  random.shuffle(indices)

  for i in range(0, num_samples, batch_size):
    batch_idx = indices[i : i+batch_size]
    yield X[batch_idx], y[batch_idx]

# X_sub = X_norm[:50]
# y_sub = y_raw[:50]
#
# batch_gen = get_batches(X_sub, y_sub, batch_size=10)
# images, labels = next(batch_gen)
#
# print(f"Batch images shape: {images.shape}")
# print(f"Batch labels shape: {labels.shape}")
#
# sample_img = images[0].reshape(28, 28)
# print(f"\nLabel: {labels[0]}")
# print("Visualized array:")
# for row in sample_img:
#   print("".join(["#" if pixel > 0.5 else " " for pixel in row]))

X_sub = X_norm[:100]
y_sub = y_raw[:100]

model = MLP(784, [32, 10], nonlin='relu')

# print("\nRunning single mini-batch test:")
print("\nTesting Vectorized Forward and Backward Pass:")

for images, labels in get_batches(X_sub, y_sub, batch_size=16):
  # FOR VALUE (SCALAR):
  # batch_loss = Value(0.0)

  # for img, target in zip(images, labels):
  #   x = [Value(pixel) for pixel in img]
  #   logits = model(x)
  #   loss = softmax_cross_entropy(logits, target)

  #   batch_loss = batch_loss + loss

  # model.zero_grad()
  # batch_loss.backward()

  # print(f"Batch loss: {batch_loss.data:.4f}")
  # print("Backpropagation done.")
  # break

  # ----------------------------------------
  # FOR TENSOR:
  x = Tensor(images) # Tensor shape (16, 784)
  logits = model(x)

  loss = vectorized_softmax_cross_entropy(logits, labels)

  model.zero_grad()
  loss.backward()

  print(f"Input batch shape: {x.data.shape}")
  print(f"Logits shape: {logits.data.shape}")
  print(f"Batch loss: {loss.data:.4f}")
  print(f"Weight gradient shape (Layer 1): {model.layers[0].w.grad.shape}")
  print("Backward pass done.")
  break
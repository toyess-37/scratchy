from sklearn.datasets import fetch_openml
import numpy as np
import random

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

X_sub = X_norm[:50]
y_sub = y_raw[:50]

batch_gen = get_batches(X_sub, y_sub, batch_size=10)
images, labels = next(batch_gen)

print(f"Batch images shape: {images.shape}")
print(f"Batch labels shape: {labels.shape}")

sample_img = images[0].reshape(28, 28)
print(f"\nLabel: {labels[0]}")
print("Visualized array:")
for row in sample_img:
  print("".join(["#" if pixel > 0.5 else " " for pixel in row]))
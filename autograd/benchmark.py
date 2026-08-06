import time
import numpy as np
from sklearn.datasets import fetch_openml

from value import Value
import nn as scalar_nn
from losses import softmax_cross_entropy

from tensor import Tensor
import nn_tensor as tensor_nn
from losses import vectorized_softmax_cross_entropy

print("Loading data:")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_norm = (mnist.data[:200] / 255.0).astype(np.float32)
y_raw = mnist.target[:200].astype(int)

batch_size = 10
hidden_dim = 32
num_samples = len(X_norm)

# SCALAR ENGINE (Value)
print("Running Scalar Engine:")
scalar_model = scalar_nn.MLP(784, [hidden_dim, 10], nonlin='relu')
lr = 0.05

t0 = time.perf_counter()

for i in range(0, num_samples, batch_size):
  batch_x = X_norm[i : i + batch_size]
  batch_y = y_raw[i : i + batch_size]
    
  # Process image by image in pure Python loops
  batch_loss = Value(0.0)
  for img, target in zip(batch_x, batch_y):
    x_val = [Value(pixel) for pixel in img]
    logits = scalar_model(x_val)
    loss = softmax_cross_entropy(logits, target)
    batch_loss = batch_loss + loss.data
    
    batch_loss = batch_loss * (1.0 / batch_size)
    
    scalar_model.zero_grad()
    batch_loss.backward()
    
    for p in scalar_model.parameters():
      p.data -= lr * p.grad

scalar_time = time.perf_counter() - t0
print(f"Scalar Engine Time (200 images): {scalar_time:.4f} seconds")

# VECTORIZED ENGINE (Tensor)
print("Running Vectorized Engine:")
tensor_model = tensor_nn.MLP(784, [hidden_dim, 10], nonlin='relu')

t0 = time.perf_counter()

for i in range(0, num_samples, batch_size):
  batch_x = X_norm[i : i + batch_size]
  batch_y = y_raw[i : i + batch_size]
  
  x_tensor = Tensor(batch_x)
  logits = tensor_model(x_tensor)
  loss = vectorized_softmax_cross_entropy(logits, batch_y)
  
  tensor_model.zero_grad()
  loss.backward()
  
  for p in tensor_model.parameters():
    p.data -= lr * p.grad

tensor_time = time.perf_counter() - t0
print(f"Tensor Engine Time (200 images): {tensor_time:.4f} seconds")

speedup = scalar_time / tensor_time
print(f"Speedup Factor: {speedup:.1f}")
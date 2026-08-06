from sklearn.datasets import fetch_openml
import numpy as np
from tensor import Tensor
from nn_tensor import MLP
from losses import vectorized_softmax_cross_entropy
from train_mnist import Trainer
from utils_mnist import Profiler, get_batches

# from value import Value
# from nn import MLP
# from losses import softmax_cross_entropy

print("Fetching MNIST:")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_raw, y_raw = mnist.data, mnist.target.astype(int)
X_norm = (X_raw / 255.0).astype(np.float32)

X_train, y_train = X_norm[:50000], y_raw[:50000]
X_val, y_val = X_norm[50000:60000], y_raw[50000:60000]
X_test, y_test = X_norm[60000:70000], y_raw[60000:70000]

model = MLP(784, [64, 10], nonlin='relu')
trainer = Trainer(model, lr=0.05)

epochs = 10
batch_size = 64

print("Benchmark:")
with Profiler() as p:
  for epoch in range(epochs):
    if epoch == 5:
      trainer.lr *= 0.5
      print(f"reduced learning rate to {trainer.lr}")
    
    loss = trainer.train_epoch(X_train, y_train, get_batches, batch_size=batch_size)
    acc = trainer.evaluate(X_val, y_val)

    print(f"Epoch {epoch+1:02d} | Train loss: {loss:.4f} | Val. Acc.: {acc:.2f}%")

print(f"\nTraining complete.")
print(f"Total time elapsed: {p.elapsed_time:.2f} secs")
print(f"Peak RAM allocated: {p.peak_ram_mb:.2f} MB")
print(f"\nTesting phase:")
test_acc = trainer.evaluate(X_test, y_test)
print(f"Final Test Accuracy: {test_acc:.2f}%")
# output:
# Testing phase:
# Final Test Accuracy: 95.64%


# # --------------------------------------------------
# # OLD STUFF:
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

# # SANITY CHECK PART:
# X_sub = X_norm[:100]
# y_sub = y_raw[:100]
#
# model = MLP(784, [32, 10], nonlin='relu')
#
# # print("\nRunning single mini-batch test:")
# print("\nTesting Vectorized Forward and Backward Pass:")
#
# for images, labels in get_batches(X_sub, y_sub, batch_size=16):
#   # FOR VALUE (SCALAR):
#   # batch_loss = Value(0.0)
#
#   # for img, target in zip(images, labels):
#   #   x = [Value(pixel) for pixel in img]
#   #   logits = model(x)
#   #   loss = softmax_cross_entropy(logits, target)
#
#   #   batch_loss = batch_loss + loss
#
#   # model.zero_grad()
#   # batch_loss.backward()
#
#   # print(f"Batch loss: {batch_loss.data:.4f}")
#   # print("Backpropagation done.")
#   # break
#
#   # ----------------------------------------
#   # FOR TENSOR:
#   x = Tensor(images) # Tensor shape (16, 784)
#   logits = model(x)
#
#   loss = vectorized_softmax_cross_entropy(logits, labels)
#
#   model.zero_grad()
#   loss.backward()
#
#   print(f"Input batch shape: {x.data.shape}")
#   print(f"Logits shape: {logits.data.shape}")
#   print(f"Batch loss: {loss.data:.4f}")
#   print(f"Weight gradient shape (Layer 1): {model.layers[0].w.grad.shape}")
#   print("Backward pass done.")
#   break
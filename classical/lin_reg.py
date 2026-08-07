import numpy as np

# toy dataset
# y = 3x + 2 + noise
np.random.seed(42)
X_raw = np.linspace(-2, 2, 100)
noise = np.random.normal(0, 0.5, size=X_raw.shape)
y = 3*X_raw + 2 + noise

X = np.column_stack([np.ones(X_raw.shape[0]), X_raw])

print(f"Dataset shapes: X -> {X.shape}, y -> {y.shape}")

# Closed-Form Normal Equation
# w = (X^T @ X)^-1 @ X^T @ y

w_normal = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"[Closed-form solution] Weights [Bias, Slope]: {w_normal}")

# Gradient Descent
n = len(y)
w_gd = np.zeros(X.shape[1])
lr = 0.05
epochs = 100

for epoch in range(epochs):
  preds = X @ w_gd
  error = preds - y
  loss = np.sum(error ** 2) / n
  grad = (2/n) * (X.T @ error)
  w_gd -= lr * grad

print(f"[Gradient Descent] Weights [Bias, Slope]: {w_gd}")

print(f"Multicollinearity:")

X_multi = np.column_stack([np.ones(100), X_raw, X_raw + 1e-5 * np.random.randn(100)])

try:
  w_bad = np.linalg.inv(X_multi.T @ X_multi) @ X_multi.T @ y
  print(f"Weights on multicollinear degradation: {w_bad}")
except np.linalg.LinAlgError:
  print(f"Normal method failed: Singular matrix")

w_gd_multi = np.zeros(X_multi.shape[1])
for epoch in range(epochs):
  pred = X_multi @ w_gd_multi
  err = pred - y
  grad = (2/n) * (X_multi.T @ err)
  w_gd_multi -= lr * grad

print(f"Gradient Descent yielded: {w_gd_multi}")
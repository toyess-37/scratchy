import numpy as np
# from sklearn.linear_model import LogisticRegression

def sigmoid(z):
  z = np.clip(z, -500, 500)
  return np.where(z>=0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def compute_loss(y, y_pred):
  eps = 1e-15
  y_pred = np.clip(y_pred, eps, 1-eps)
  return -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

def compute_gradient(X, y, y_pred):
  n = len(y)
  return (1/n) * np.dot(X.T, (y_pred-y))


# toy dataset
X = np.array([
  [1,1], [1,2], [2,1], [2,2],
  [4,4], [4,5], [5,4], [5,5]
])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# sklearn approach:
# model = LogisticRegression()
# model.fit(X,y)

X_b = np.c_[X, np.ones((X.shape[0], 1))]

lr = 0.1
epochs = 1001
weights = np.zeros(X_b.shape[1])

for epoch in range(epochs):
  z = np.dot(X_b, weights)
  y_pred = sigmoid(z)

  loss = compute_loss(y, y_pred)
  grad = compute_gradient(X_b, y, y_pred)

  weights -= lr * grad

  if epoch % 200 == 0:
    print(f"Epoch {epoch}: Loss={loss:.4f}")

print(f"Learned weights: {weights}")
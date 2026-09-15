import numpy as np

def euclidean_dist(x1, x2):
  return np.sqrt(np.sum((x1-x2)**2, axis=-1))

class KNN:
  def __init__(self, k=3):
    self.k = k

  def fit(self, X, y):
    self.X_train = np.array(X)
    self.y_train = np.array(y)

  def predict_point(self, X):
    distances = euclidean_dist(self.X_train, X)
    k_indices = np.argsort(distances)[:self.k]
    k_labels = self.y_train[k_indices]

    labels, counts = np.unique(k_labels, return_counts=True)
    return labels[np.argmax(counts)]

  def predict(self, X):
    X = np.array(X)
    return np.array([self.predict_point(x) for x in X])

X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
y_train = np.array([0, 0, 1, 1, 0, 1])

clf = KNN(k=3)
clf.fit(X_train, y_train)

X_test = np.array([[2, 2], [7, 9]])
predictions = clf.predict(X_test)
print(f"Predictions for {X_test.tolist()}: {predictions}")
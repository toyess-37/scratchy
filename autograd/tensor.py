import numpy as np

class Tensor:
  def __init__(self, data, _children=(), _op=''):
    self.data = np.array(data, dtype=np.float32)
    self.grad = np.zeros_like(self.data)
    self._prev = set(_children)
    self._backward = lambda: None

  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += self._unbroadcast(out.grad, self.data.shape)
      other.grad += self._unbroadcast(out.grad, other.data.shape)

    out._backward = _backward
    return out

  def __radd__(self, other):
    return self + other

  def __sub__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return self + (-other)

  def __rsub__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return other + (-self)

  def __neg__(self):
    out = Tensor(-self.data, (self,), 'neg')

    def _backward():
      self.grad += -out.grad

    out._backward = _backward
    return out

  def __matmul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data @ other.data, (self, other), '@')

    def _backward():
      # d (A @ B)/dA = grad @ B.T
      # d(A @ B)/dB = A.T @ grad

      self.grad += out.grad @ other.data.T
      other.grad += self.data.T @ out.grad

    out._backward = _backward
    return out

  def relu(self):
    out = Tensor(np.maximum(0, self.data), (self,), 'relu')

    def _backward():
      self.grad += (self.data > 0) * out.grad

    out._backward = _backward
    return out

  def exp(self):
    out = Tensor(np.exp(self.data), (self,), 'exp')

    def _backward():
      self.grad += out.data * out.grad

    out._backward = _backward
    return out
  
  def log(self):
    out = Tensor(np.log(self.data + 1e-15), (self,), 'log')

    def _backward():
      self.grad += (1.0 / (self.data + 1e-15)) * out.grad
    
    out._backward = _backward
    return out
  
  def sum(self, axis=None, keepdims=False):
    out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')

    def _backward():
      grad = out.grad
      if axis is not None and not keepdims:
        grad = np.expand_dims(grad, axis=axis)
      self.grad += np.broadcast_to(grad, self.data.shape)

    out._backward = _backward
    return out

  def _unbroadcast(self, grad, target_shape):
    grad_dim = grad.ndim
    target_dim = len(target_shape)

    for _ in range(grad_dim - target_dim):
      grad = grad.sum(axis=0)

    for i, dim in enumerate(target_shape):
      if dim == 1:
        grad = grad.sum(axis=i, keepdims=True)

    return grad

  def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)

    build_topo(self)

    self.grad = np.ones_like(self.data)
    for node in reversed(topo):
      node._backward()
  

    

  
  

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
      # swapaxes(-1, -2) generalizes .T to batched (B, ..., T, D) tensors
      self.grad += self._unbroadcast(out.grad @ other.data.swapaxes(-1, -2), self.data.shape)
      other.grad += self._unbroadcast(self.data.swapaxes(-1, -2) @ out.grad, other.data.shape)

    out._backward = _backward
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += self._unbroadcast(other.data * out.grad, self.data.shape)
      other.grad += self._unbroadcast(self.data * out.grad, other.data.shape)

    out._backward = _backward
    return out

  def __rmul__(self, other):
    return self * other

  def __truediv__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data / other.data, (self, other), '/')

    def _backward():
      self.grad += self._unbroadcast(out.grad / other.data, self.data.shape)
      other.grad += self._unbroadcast(-out.grad * self.data / (other.data ** 2), other.data.shape)

    out._backward = _backward
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), 'only scalar powers supported'
    out = Tensor(self.data ** other, (self,), f'**{other}')

    def _backward():
      self.grad += other * (self.data ** (other - 1)) * out.grad

    out._backward = _backward
    return out

  def transpose(self, axis1=-2, axis2=-1):
    out = Tensor(np.swapaxes(self.data, axis1, axis2), (self,), 'transpose')

    def _backward():
      self.grad += np.swapaxes(out.grad, axis1, axis2)

    out._backward = _backward
    return out

  def reshape(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
      shape = shape[0]
    out = Tensor(self.data.reshape(shape), (self,), 'reshape')

    def _backward():
      self.grad += out.grad.reshape(self.data.shape)

    out._backward = _backward
    return out

  def mean(self, axis=None, keepdims=False):
    n = self.data.size if axis is None else np.prod([self.data.shape[a] for a in np.atleast_1d(axis)])
    return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

  @staticmethod
  def cat(tensors, axis=-1):
    out = Tensor(np.concatenate([t.data for t in tensors], axis=axis), tuple(tensors), 'cat')
    sizes = [t.data.shape[axis] for t in tensors]

    def _backward():
      splits = np.cumsum(sizes)[:-1]
      grads = np.split(out.grad, splits, axis=axis)
      for t, g in zip(tensors, grads):
        t.grad += g

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
  

    

  
  

import numpy as np
from tensor import Tensor

class Layer:
  def __init__(self, nin, nout, nonlin='relu'):
    # weights initialized with gaussian distribution
    self.w = Tensor(np.random.randn(nin, nout)*np.sqrt(2.0/nin))
    self.b = Tensor(np.zeros(1, nout))
    self.nonlin = nonlin

  def __call__(self, x):
    act = x @ self.w + self.b
    if self.nonlin == 'relu':
      return act.relu()
    return act

  def parameters(self):
    return [self.w, self.b]

class MLP:
  def __init__(self, nin, nouts, nonlin='relu'):
    sz = [nin] + nouts
    self.layers = []

    for i in range(len(nouts)):
      layer_nonlin = nonlin if i < len(nouts)-1 else 'none'
      self.layers.append(Layer(sz[i], sz[i+1], nonlin=layer_nonlin))

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

  def zero_grad(self):
    for p in self.parameters():
      p.grad = np.zeros_like(p.data)
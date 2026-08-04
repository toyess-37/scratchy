from value import Value, draw_dot
import random

class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0.0

  def parameters(self):
    return []

class Neuron(Module):
  def __init__(self, nin, nonlin='tanh'):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))
    self.nonlin = nonlin

  def __call__(self, x):
    # w*x + b
    act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
    if self.nonlin == 'tanh':
      return act.tanh()
    elif self.nonlin == 'relu':
      return act.relu()
    elif self.nonlin == 'none' or self.nonlin is None:
      return act
    else:
      raise ValueError(f"Unknown activation: {self.nonlin}")

  def parameters(self):
    return self.w + [self.b]

class Layer(Module):
  def __init__(self, nin, nout, nonlin='tanh'):
    self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP(Module):
  def __init__(self, nin, nouts, nonlin='tanh'):
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

# x = [2.0, 3.0, -1.0]
# n = MLP(3, [4, 4, 1])

# out = n(x)
# print("Forward pass output:", out)

# dot = draw_dot(out)
# dot.render('mlp_graph_relu', view=True)
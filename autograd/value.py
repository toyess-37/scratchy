from graphviz import Digraph
import os
import math
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

class Value:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += 1 * out.grad
      other.grad += 1 * out.grad

    out._backward = _backward
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    
    out._backward = _backward
    return out

  def __neg__(self):
    out = Value(-self.data, (self,), '-op')

    def _backward():
      self.grad += -1 * out.grad

    out._backward = _backward
    return out

  def __sub__(self, other):
    return self + (-other)

  def __radd__(self, other):
    return self + other

  def __rmul__(self, other):
    return self * other

  def __rsub__(self, other):
    return Value(other) + (-self)

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only int/float powers"
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
      self.grad += other * (self.data ** (other-1)) * out.grad

    out._backward = _backward
    return out

  def tanh(self):
    x = self.data
    t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
    out = Value(t, (self,), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad

    out._backward = _backward
    return out

  def relu(self):
    out = Value(max(0, self.data), (self,), 'relu')

    def _backward():
      self.grad += (out.data > 0) * out.grad

    out._backward = _backward
    return out

  def exp(self):
    out = Value(math.exp(self.data), (self,), 'exp')
    def _backward():
      self.grad += out.data * out.grad

    out._backward = _backward
    return out

  def log(self):
    out = Value(math.log(self.data), (self,), 'log')
    def _backward():
      self.grad += (1.0/self.data) * out.grad

    out._backward = _backward
    return out

  def __truediv__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return self * (other ** -1)

  def __rtruediv__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return other / self

  def backward(self):
    self.grad = 1.0

    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)

    build_topo(self)

    for node in reversed(topo):
      node._backward()

  def zero_grad(self):
    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)

    build_topo(self)

    for node in topo:
      node.grad = 0.0

def trace(root):
  # builds set of all nodes and edges in the graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='pdf', graph_attr={'rankdir': 'LR'},  
                node_attr={'fontname': 'Inconsolata'}, 
                edge_attr={'fontname': 'Inconsolata'})

  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    dot.node(name = uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      op_uid = uid + "_op_" + n._op
      dot.node(name = op_uid, label = n._op)
      dot.edge(op_uid, uid)

  for n1, n2 in edges:
    op_uid = str(id(n2)) + "_op_" + n2._op
    dot.edge(str(id(n1)), op_uid)

  return dot

def check_gradient(f, var, eps=1e-6):
  var.data += eps
  out_plus = f().data

  var.data -= 2*eps
  out_minus = f().data

  var.data += eps

  numerical_grad = (out_plus - out_minus) / (2*eps)
  return numerical_grad


a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

# def lol():
#   return (a.exp() + b.exp()).log() / c

# L = lol()
# L.backward()
# print(f"{check_gradient(lol, a)} | a.grad = {a.grad}")
# print(f"{check_gradient(lol, b)} | b.grad = {b.grad}")
# print(f"{check_gradient(lol, c)} | c.grad = {c.grad}")
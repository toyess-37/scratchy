# Autograd Engine

This is the first component in a series of my preparation of understanding and implementing DL concepts.

The tl;dr is - inspired from [Andrej Karpathy's Micrograd](https://www.youtube.com/watch?v=PaCmpygFfXo) and his buildmore series, I decided to start implementing it. 

1. I started off with a `Value` class in my file [value.py](./value.py).
  - The class represents a number, but the target is to maintain a DAG for any expression. So, additionally, for each node, I store its "children" _which are essentially the nodes which point to it_ (should've named parents but in CS typically one parent gives off many children so...), i.e. if `c = a+b` then `children(c) = (a, b)`. Also, if that node represents an 

  - I wrote several functions `__add__`, `__mul__`, `__sub__` etc. (and their corresponding reverse versions too so that both 2+Value(2) and Value(2)+2 are respected).

  - `__repr__` prints what the Value object is.

  - `_backward` function is used to calculate the gradient effect of that operation.

  - `tanh` and `relu` activation functions for introducing non-linearity in the perceptron.

2. For the Backpropagation, I used DFS:
```py
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
```
Then, I need to call the `zero_grad()` that I need to call before calling `backward()` otherwise previously done values will be used and addition will be done on them.

More details to be added.
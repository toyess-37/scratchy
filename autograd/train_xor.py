from value import Value
from nn import MLP

def mse_loss(preds, targets):
  loss = sum((pred-Value(target))**2 for pred, target in zip(preds, targets))
  return loss

def train_model(model, inputs, targets, loss_fn, lr=0.05, epochs=100, print_every=10):
  # SGD
  print(f"Training starts now (epochs: {epochs}, learning rate: {lr}):")

  for epoch in range(epochs):
    preds = [model(x) for x in inputs]
    loss = loss_fn(preds, targets)
    model.zero_grad()
    loss.backward()

    for p in model.parameters():
      p.data -= lr * p.grad

    if epoch % print_every == 0 or epoch == epochs - 1:
      print(f"Epoch {epoch:3d} | Loss: {loss.data:.6f}")

  return loss.data

def evaluate_model(model, inputs, targets, threshold=0.5):
  print("Results:")
  for x, target in zip(inputs, targets):
    pred = model(x)
    input_vals = [i.data if hasattr(i, 'data') else i for i in x]

    prediction = 1.0 if pred.data >= threshold else 0.0
    status = "PASS" if prediction == target else "FAIL"
    print(f"Input: {input_vals} -> Pred: {pred.data: 6.3f} | Target: {target: 3.1f} [{status}]")

# training for XOR
inputs_relu = [
  [Value(1.0), Value(1.0)],
  [Value(1.0), Value(0.0)],
  [Value(0.0), Value(1.0)],
  [Value(0.0), Value(0.0)],
]
targets_relu = [0.0, 1.0, 1.0, 0.0]

mlp_relu = MLP(2, [4, 1], nonlin='relu')

train_model(
  model=mlp_relu,
  inputs=inputs_relu,
  targets=targets_relu,
  loss_fn=mse_loss,
  lr=0.05,
  epochs=100
)

evaluate_model(mlp_relu, inputs_relu, targets_relu, threshold=0.5)
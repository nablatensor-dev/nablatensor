# MNIST MLP in Java on the NablaTensor tensor API

*Keywords: java neural network mnist, java gpu tensor library, jvm deep learning example, java matmul gpu*

A plain **784 → 256 → 10** multilayer perceptron — sigmoid hidden layer,
softmax cross-entropy output — trained with mini-batch SGD. It exists to show
the `Tensor` side of NablaTensor end to end: `matmul`, elementwise ops, a
numerically-stable softmax, and **hand-written backprop** (NablaTensor does not
ship autodiff-over-tensors — that is a deliberate non-goal). No CNN, no
autograd, ~230 lines. Reaches **~98%** held-out accuracy in 30 epochs.

Source: [`nablatensor-examples/.../MnistMlp.java`](../../nablatensor-examples/src/main/java/com/nablatensor/examples/MnistMlp.java)

## Backend

The `Tensor` API runs only on a GPU `ComputeBackend` (Vulkan / ROCm / CUDA);
there is no CPU tensor backend. The example picks
`NablaTensors.defaultDevice()` and prints a notice and exits if no device is
present. (The adjoint-AD engine — `Nabla` / `SDouble` — is the part that
runs on a GPU-less laptop; this example is the tensor library.)

## The step

```java
// init: each weight scaled by 1/sqrt(fan-in) so the sigmoids start in their
// linear middle, not saturated flat — the difference between ~98% and stalling at ~94%.

// forward
Tensor a1  = x.matmul(w1).add(b1).sigmoid();
Tensor z2  = a1.matmul(w2).add(b2);
Tensor p   = softmax(z2);                 // z2 - rowmax, exp, / rowsum

// backward (manual)
Tensor dZ2 = p.sub(y).mul(1.0 / batch);
Tensor dW2 = a1.transpose().matmul(dZ2);
Tensor dA1 = dZ2.matmul(w2.transpose());
Tensor dZ1 = dA1.mul(a1.mul(a1.neg().add(1.0)));   // sigmoid'
Tensor dW1 = x.transpose().matmul(dZ1);

// SGD: param <- param - lr * grad
```

Every `Tensor` is `close()`d as soon as it is dead — buffers are freed
deterministically, not on GC.

## Run it

Get the Kaggle-format `mnist_train.csv` (`label,pixel0,...,pixel783` per row,
60,000 rows) and pass its path:

```bash
mvn -o -q install
mvn -o -q -pl nablatensor-examples exec:java \
  -Dexec.mainClass=com.nablatensor.examples.MnistMlp \
  -Dexec.args="data/mnist/mnist_train.csv"
```

Knobs (defaults shown): `-Depochs=30 -Dbatch=100 -Dhidden=256 -Dlr=1.5`.

With **no CSV in reach** the example falls back to a small synthetic 10-blob
dataset so the pipeline still runs (this is also what the smoke test exercises).

## Output

Real MNIST, fixed 80/20 split on row order, ROCm on a Radeon 780M iGPU:

```
nablatensor devices: [rocm:0, vulkan:0] -> training on rocm:0
data/mnist/mnist_train.csv: 60000 samples in 513 ms
split: 48000 train / 12000 test (80/20 fixed)
epoch 1/30  avg loss 0.5420
epoch 5/30  avg loss 0.0973
epoch 10/30  avg loss 0.0510
epoch 20/30  avg loss 0.0177
epoch 30/30  avg loss 0.0074
training: 30 epochs x 48000 samples in 12034 ms (119661 img/s)
final train accuracy: 99.98%, test accuracy: 97.76%
```

~98% on held-out digits with two matmuls, a sigmoid, a softmax and a reverse
sweep — no layers library, no optimizer, no BLAS. The point here is the tensor
API and the manual gradient, not chasing the last fraction of a percent.

## What to change

- **Wider / deeper:** raise `-Dhidden`, or add a second `w`/`b` pair and one
  more sigmoid + transposed-matmul in the backward pass.
- **ReLU hidden:** swap `sigmoid()` for `relu()` and the `sigmoid'` line for
  `reluBackward`.
- **Another backend:** `-Dnablatensor.tensor.backend=vulkan`.

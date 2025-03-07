## Level 1: Manual Differentiation

This example implements a simple Multilayer Perceptron (MLP) to train on the MNIST database.

The code is written in 100% Odin, has no dependencies other than the core library, and does no parallel optimization.

The MLP uses He initialization and ADAM optimization, and should be able to reach around 97% validation accuracy in around a minute or so, and generally tops out around 97.5%.

### Downsides

* The gradient calculations are done manually, which is not sustainable for larger and more complicated model architectures.
* There are no parallel optimizations, which means it could be much faster.
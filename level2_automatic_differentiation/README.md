## Level 2: Automatic Differentiation

This example implements a simple Linear layer to train on the MNIST database.

Like the level 1 example, the code is written in 100% Odin, has no dependencies other than the core library, and does no parallel optimization. It also uses He initialization and ADAM optimization.

The difference between this and level 1, is that instead of manually calculating the gradients, that process is done automatically by building a computational graph.

The accuracy should cap out around 92.75% after several minutes.

### Downsides

* This is extremely slow, and for that reason I didn't implement a full MLP.
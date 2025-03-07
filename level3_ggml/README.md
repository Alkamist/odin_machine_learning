## Level 3: ggml

This example implements a Multilayer Perceptron (MLP) using a custom Odin wrapper around [ggml](https://github.com/ggml-org/ggml).

Ggml is a machine learning library, which greatly optimizes the operations done during machine learning through parallel processing and utilizing the GPU, while also automatically differentiating to calculate gradients.

This binding to ggml is incomplete, and the wrapper I made is very thin; this is an educational example only. It will also currently only work on Windows using CUDA, but ggml is designed to work on many architectures.

You will need to build ggml yourself, and drop "ggml.lib", "ggml-base.lib", "ggml-cpu.lib", "ggml-cuda.lib" next to ggml.odin.

To get up and running:

* Download and install the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit).
* Clone or download the [ggml](https://github.com/ggml-org/ggml) repository.
* Open the ggml repository with Visual Studio Command Prompt.
* `mkdir build`
* `cd build`
* For CPU only: `cmake .. -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release`
* Or with CUDA: `cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release`
* If you have a different CUDA version you need to modify the path above.
* cmake --build . --config Release
* When you build the code with Odin, you need to pass in `-define:CUDA_VERSION='12.8'` but change 12.8 to your version.

With my NVIDIA GeForce RTX 3090 Ti I am able to train the MLP to around 97.5% validation accuracy on the MNIST database in roughly 8 seconds.

### Downsides

* Greatly increased complexity due to parallel processing and having to build and rely on a 3rd party library in a different language.
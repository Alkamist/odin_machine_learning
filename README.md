# Machine Learning in Pure Odin

This repository contains a simple machine learning library written in pure [Odin](https://odin-lang.org/), along with some examples using it. 

The goal behind this is to explore and understand concepts of machine learning from the ground up, with the hopes of applying them to game development.

I've always been interested in machine learning, but it has always felt hard to approach due to a number of reasons:
* Most of the theory is taught in mathematical notation, which can sometimes be difficult to translate to code.
* You basically have to use Python and import magical libraries that abstract all the details away from you and are difficult to dig into.
* The operations are heavily parallelized and offloaded to the GPU, which makes things even harder to follow and understand.

Basically, one day I asked the question: is it possible to do machine learning in straight up code without any libraries? What would be a better way to learn than to just try to do it yourself?

This is the result of tugging on that thread. Hopefully someone else finds it useful.

As a good place to start, I'd recommend going into the freeplay example and trying to play CartPole yourself for a while, and see how high of a score you can get. Then go into the ppo example, comment out train(), uncomment play(), and switch MODEL_FILE to cartpole.json. That will have a pretrained model play CartPole at a high level.

Something to keep in mind is that this library is currently CPU only, and not very optimized. Some things are mildly parallelized but there is a ton of room for improvement I think, even on the CPU.

Here are some of the resources that helped me a lot in my research:
* [3Blue1Brown Neural Network Series](https://www.youtube.com/watch?v=aircAruvnKk)
* [Crash Course in Deep Learning](https://gpuopen.com/learn/deep_learning_crash_course/)
* [llm.c](https://github.com/karpathy/llm.c)
* [nanoGPT](https://github.com/karpathy/nanoGPT)
* [ggml](https://github.com/ggml-org/ggml)
* [Claude](https://claude.ai/) when it doesn't lie to me
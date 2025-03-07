## What is this?

This repository catalogues my journey into learning about machine learning. 

My goal in making it is to collect meaningful experiences I've had into one place, to be able to reference them myself, and to serve as learning material for any others who are interested in machine learning.

I use the [Odin](https://odin-lang.org/) programming language, a no-nonsense low-level manual memory management-based language; visit their website to get started if you want to compile the examples I have provided.

## Learning about machine learning can be difficult

Machine learning is a very complicated topic to learn about for many reasons: 

* You are immediately presented with high level mathematics and symbols which abstract away the details of what's actually happening behind the scenes if you are not familiar with these concepts.
* On top of that, Python is the dominant language in this domain, so you are also immediately presented with dependency hell, virtual environments, and a disconnect between the code you write and the low level code happening under the hood.
* If that wasn't bad enough, because of the nature of the math behind machine learning, the code can be heavily optimized by running in parallel. This is great, but has the unfortunate side effect of making the code much harder to understand, forcing you to deal with concepts like sharing data between the CPU and GPU before you've even started learning.

I want to try to explain the very basics in a simpler way, and show you that machine learning may be more accessible than you might think.

## So what is machine learning?

Machine learning is basically a different method to solving a problem than trying to write code to do it.

Imagine you wanted to write a program that can look at a picture of a number between 0 and 9, and tell you which number it is. When you think about it, that's a really hard task. It seems so easy to just look at a picture of a number yourself and know which one it is, but how would you even begin to write a program that does that?

What if you could just show a computer a bunch of pictures of numbers paired with which numbers they are, and have it learn the program instead? That would be pretty nice.

It turns out with some clever math, you actually can do that, and that is the basis of machine learning.

## Well how do you do it?

If you think about it, the task of recognizing a number that I described above could be represented theoretically by a function. The function has input (the picture of the number), and it spits out output (the number it thinks it is).

Instead of trying to directly write the body of that function in code, it is possible write a general framework that is capable of approximating any function.

This general framework consists of these steps:

1. Covert the input into floating point numbers.
2. Multiply those floating point numbers by some other floating point numbers (the weights).
3. Add those resulting numbers together.
4. Add a single floating point to that result (the bias).
5. Apply an "activation" function to that result (sigmoid, relu, etc...).

Those steps describe a "neuron" in a neural network. You can repeat those steps to create a "layer" of neurons, and then use that layer of neurons as the input for another layer, and so on, until you decide to stop.

This network of neurons is capable of approximating functions, and all you need to do is set the values of the weights and biases in such a way that it gives you the output you want.

## How do you set the weights and biases?

Luckily, calculus exists, and you don't have to figure out the values of these weights and biases by hand. 

I won't go into details about the calculus here, but the procedure is basically as follows:

1. Initialize the network randomly.
2. Feed the network an input and receive an output.
3. Evaluate the network on how good or bad it did (the cost).
4. Use calculus on the cost to tell you how to adjust the weights and biases to make it give a slightly better output (the gradient).
5. Repeat steps 2-5 and accumulate the gradient over all of the input, or a random batch, to give a general direction to adjust the weights and biases to generally give better outputs.
6. Apply the accumulated gradient to the weights and biases, and repeat steps 2 and on until the network doesn't improve anymore.

This is called Stochastic Gradient Descent, and is the basic form of improving the weights and biases of a neural network. 

There also exists ADAM Optimization, which builds on those concepts with some extra twists to improve training efficiency.

## And that's it!

Those are the very basics of machine learning, although I'm simplifying some things and glossing over some details.

In reality, this guide only scratches the surface, but hopefully it at least achieves its goal of simply helping anyone who is struggling with understanding machine learning concepts.

## Examples

Now that you know the basics, feel free to take a look at my examples in the repository starting with level 1.

The examples train on the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database), and you will need to unzip it for the code to use it.

I recommend building all of the examples with `-o:speed`

Also, for further details, take a look at these resources that helped me:

* [3Blue1Brown Neural Network Series](https://www.youtube.com/watch?v=aircAruvnKk)
* [Crash Course in Deep Learning](https://gpuopen.com/learn/deep_learning_crash_course/)
* [Automatic Differentiation in C](https://github.com/Janko-dev/autodiff/tree/main)
* [ggml](https://github.com/ggml-org/ggml)
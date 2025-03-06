# 📒 My Notes on Andrej Karpathy's lectures

This repository contains notebooks based on the great Andrej Karpathy's series **"Neural Networks: Zero to Hero"**

## 📚 Notebooks Overview

### 🤏🏂 01-Micrograd
Implementation of [Micrograd](https://github.com/karpathy/micrograd), a compact and didactical deep learning library.
- [**Readme**](karpathings_01_micrograd/README.md)
- [nb📕](karpathings_01_micrograd/micrograd_01.ipynb): Manual backpropagation of a simple function and a neuron.
- [nb📘](karpathings_01_micrograd/micrograd_02.ipynb): Adding more functions to our Micrograd library.
- [nb📗](karpathings_01_micrograd/micrograd_03.ipynb): Training a small neural network on a simple task.

### 🧑‍🤝‍🧑💬 02-Bigram
Implementation of a bigram character-level language model.
- [**Readme**](karpathings_02_bigram/README.md)
- [nb📕](karpathings_02_bigram/makemore_01_bigrams.ipynb): Building a bigram model to predict the next character based on the previous one.

### 🧠👨‍👦‍👦 03-MLP
Implementation of a multi-layer perceptron model, predicting the next character based on the three previous ones.
- [**Readme**](karpathings_03_MLP/README.md)
- [nb📕](karpathings_03_MLP/makemore_02_MLP.ipynb): Developing a multi-layer perceptron model.
  
### 🧠⚡ 04-MLP v2
Further improvements and adjustments to the MLP implementation.
- [**Readme**](karpathings_04_MLP/README.md)
- [nb📕](karpathings_04_MLP/makemore_03.ipynb): Revisiting the MLP implementation with improvements.
- [nb📘](karpathings_04_MLP/makemore_03_ipynb_pytorch.ipynb): Implementing the MLP model adding more layers and imitating PyTorch.

### 🥷🏂 05-MLP v3
Replacing loss.backwards() with a manual backprop implementation
- [**Readme**](karpathings_05_MLP/README.md)
- [nb📕](karpathings_05_MLP/makemore_04.ipynb): Becoming a backprop ninja

### 🌊🎵 06-WaveNet
Replacing loss.backwards() with a manual backprop implementation
- [**Readme**](karpathings_06_wavenet/README.md)
- [nb📕](karpathings_06_wavenet/makemore_05_init.ipynb): Initial notebook, used for comparision if needed
- [nb📘](karpathings_06_wavenet/makemore_05_fixes.ipynb): Bug fixes in previous MLP works.
- [nb📗](karpathings_06_wavenet/makemore_05_wavenet.ipynb): Implementing WaveNet

### 🤖🦶 07-Transformers
Creating a transformer architecture from scratch. Training it with *Monty Python Flying Circus'* scripts.
- [**Readme**](karpathings_07_transformer/README.md)
- [nb📕](karpathings_07_transformer/gpt_developing.ipynb): Data cleaning and initial work.
- [nb📘](karpathings_07_transformer/self_attention.ipynb): Explanation of some of the transformer component.
- 🐍: Code for train and generate text with a bigram model and with a gpt model.

## 🔗 References

- [Neural Networks: Zero to Hero Lecture Series](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

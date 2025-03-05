# 🤖 My Transformers Study Notes

This repository contains notebooks based on [the seventh lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy) of Andrej Karpathy's series **"Neural Networks: Zero to Hero"**. I've made some tiny modifications and added additional comments. 📝

Work is divided into 2 notebooks:

- [1📕](gpt_developing.ipynb): Initial notebook, used for comparision if needed
- [2📘](self_attention.ipynb): Bug fixes in previous MLP works.

And 4 python script:

```
|
bigram/
├── train_bigram.py     # Training script for the bigram model
├── generate_bigram.py  # Text generation script using trained bigram model
|
gpt/
├── train_gpt.py        # Training script for the GPT model
└── generate_gpt.py     # Text generation script using trained GPT model
```

# Implementation

The implementation follows a progressive approach, starting with a simple bigram model and building up to a full transformer-based GPT model. The code implements key concepts like self-attention, positional embeddings, and multi-head attention.

![](assets/decoder.png)

# 📈 Results

## Quantitatve results.

# hyperparameters
```python
batch_size = 64      # Number of parallel sequences
block_size = 256     # Maximum context length for predictions
max_iters = 5000     # Total training iterations
eval_interval = 500  # Evaluation frequency
learning_rate = 3e-4 # Learning rate (crucial for transformer models)
n_embd = 384        # Embedding dimension
n_head = 6          # Number of attention heads
n_layer = 6         # Number of transformer layers
dropout = 0.2       # Dropout rate for regularization
```


We achive a train loss of TL and a val loss of VL after training during x min with a RTX 4070

## Qualitative results

Impressive results achieved.


## 🔗 References

📌 **WaveNet Paper**:  
- [Attention is all you need](https://arxiv.org/pdf/1706.03762)  
  *Ashish Vaswani∗
Google Brain
avaswani@google.com
Noam Shazeer∗
Google Brain
noam@google.com
Niki Parmar∗
Google Research
nikip@google.com
Jakob Uszkoreit∗
Google Research
usz@google.com
Llion Jones∗
Google Research
llion@google.com
Aidan N. Gomez∗ †
University of Toronto
aidan@cs.toronto.edu
Łukasz Kaiser∗
Google Brain
lukaszkaiser@google.com
Illia Polosukhin∗ ‡
illia.polosukhin@gmail.com*

📌 **Residual Connections**:  
  - Deep Residual Learning for Image Recognition
Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun
https://arxiv.org/pdf/1512.03385

📌 **Dropout**:  
Dropout: A Simple Way to Prevent Neural Networks from
Overfitting
Nitish Srivastava nitish@cs.toronto.edu
Geoffrey Hinton hinton@cs.toronto.edu
Alex Krizhevsky kriz@cs.toronto.edu
Ilya Sutskever ilya@cs.toronto.edu
Ruslan Salakhutdinov rsalakhu@cs.toronto.edu
https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf

📌 **Dataset**:
- [Original dataset]https://www.kaggle.com/code/valkling/monty-python-scripts-database-to-text upload in kaggle by Valking
- [Clean dataset]https://www.kaggle.com/code/alvarofg21/monty-python-scripts-text-cleaning upload in kaggle by myself
 
📌 **Karpathings**:
- [Neural Networks: Zero to Hero Lecture Series](https://www.youtube.com/watch?v=VMj-3S1tku0)

📌 **Other stuff**:

Transformers have been a big earthwuake in the machine learning area, therefore the web is full of learnign resources. To dont be so long i will keep it short with wome of wich a found more relevants, wich wont surprise you as a big 3b1b fanboy

Visualizing transformers and attention | Talk for TNG Big Tech Day '24- Grant Sanderson https://www.youtube.com/watch?v=KJtZARuO3JY&ab_channel=GrantSanderson

📌 **3Blue1Brown’s Neural Network Series**

emoji 5.1 **[Breve explicación de los modelos extensos de lenguaje (LLM) | DL5]https://www.youtube.com/watch?v=LPZh9BOjkQs&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=6&ab_channel=3Blue1Brown)** 
emoji 5.2 **[Transformers (how LLMs work) explained visually | DL5]https://www.youtube.com/watch?v=wjZofJX0v4M&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=6&ab_channel=3Blue1Brown)**  
emoji 6 **[Attention in transformers, step-by-step | DL6](https://www.youtube.com/watch?v=eMlx5fFNoYc&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=7&ab_channel=3Blue1Brown)**  
emoji 7 **[Backpropagation calculus | DL7](https://www.youtube.com/watch?v=9-Jl0dxWQs8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=8&ab_channel=3Blue1Brown)**  

# Gen-AI-LLMs
This repo will hold all docs and code w.r.t Gen-AI-LLM course material
# Gen-AI-Keywords
# Gen-AI Lifecycle
# Transformers

LLMs-> more parameters -> more memory ->better models

So it's important to understand difference between Parameters & Hyperparameters

- Parameters are variables that are learned by the model from the data, such as **weights** and **biases**. They allow the model to **learn the rules** from the data. Hence why models with billions of parameters are performing really Good(e.g. GPT(175B),BLOOM(175B)) vs model with millions of parameters(e.g. Bert(110M))

- Hyperparameters are variables that are set manually before training, such as **learning rate, batch size, number of layers**, etc. They control how the model is trained and how the parameters are updated.

Some examples of parameters are the **attention weights, the feed-forward network weights, and the embeddings**. 

Some examples of hyperparameters are the **number of heads, the hidden size, the dropout rate, and the warmup steps**.

 *“[Attention is All You Need](https://arxiv.org/abs/1706.03762)” by Vaswani et al. (2017) was the paper which introduced transformer architecture*

## Transformers model key-highlights
  - Scale efficiently to use multi-core GPU's. 
  - Parallel processing of input data and thus making use of much larger training datasets. 
  - Learn to pay attention to the meaning of the words it's processing

The **Power** of LLM's comes from model architecture which was used to train this kind of models vs old architecture like RNN

The power of the transformer architecture lies in its ability to learn the relevance and context of all of the words in a sentence and not just the neighbors.


![image](https://github.com/ShankarChavan/Gen-AI-LLMs/assets/6409350/33d2d07a-dc68-4db2-8683-7b548116617f)

It applies attention weights to those relationships so that the model learns the relevance of each word to each other words no matter where they are in the input.

![image](https://github.com/ShankarChavan/Gen-AI-LLMs/assets/6409350/504d6151-56e4-4a75-99ea-62300ab30a2e)

Based on this sentence itself model has ability to answer following questions:

 - Who has the book?
 - Who could have the book? 

In a nutshell it has ability to understand the context of document given to it.

![image](https://github.com/ShankarChavan/Gen-AI-LLMs/assets/6409350/9ac47d9c-fd72-4398-9cec-c3622a05e5a9)

In the above diagram we can see that word **book** is strongly connected with or paying attention to the word **teacher** and the word **student**. 

This is called **self-attention** and the ability to learn a attention in this way across the whole input significantly approves the model's ability to encode language. 

Self-attention is the key attributes of the transformer architecture.Let's dive in on transformer architecture diagram

### Transformer
![image](https://github.com/ShankarChavan/Gen-AI-LLMs/assets/6409350/6992c3e4-2d20-4fa6-878d-211c76d0b1b0)

### Simplified Transformer architecture 

![image](https://github.com/ShankarChavan/Gen-AI-LLMs/assets/6409350/552c2f5e-3bdd-439e-91b7-eda8774a54cf)



![image](https://github.com/ShankarChavan/Gen-AI-LLMs/assets/6409350/1e787bd7-8b4b-4825-a7f6-60b3f67d9567)


# prompt engineering and prompting types
# Generative Configurations
# Fine-tuning LLM's with PEFT & LoRA
# Fine-tuning LLM's with RLHF
# LLM's in applications

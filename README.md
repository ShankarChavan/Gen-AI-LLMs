# Gen-AI-LLMs
This repo will hold all docs and code w.r.t Gen-AI-LLM materials
# Gen-AI-Keywords
# Gen-AI Lifecycle
Here is diagram of the Gen-AI Lifecycle

![image](https://github.com/ShankarChavan/Gen-AI-LLMs/assets/6409350/3ab65783-56d5-4c2a-bd2f-f14fda6b7dfb)


- **Scope**: Defining the scope of LLM as accurately and narrowly is very important w.r.t use-case because LLM's are capable of carrying out multiple tasks based on the size and architecture of model.
  Getting really specific about what you need your model to do can save you **time** and **compute cost**

  Examples of Specific tasks can be Q & A bot, Text Summarization, or Named-Entity recognition etc. 
- **Select**: In this stage it's important to decide whether to train our own model from scratch or work with an existing base model.
- **Adapt & Align Model**: With our model in hand, the next step is to assess its performance and carry out additional training if needed for our application.

  **Prompt engineering** can sometimes be enough to get our model to perform well, so we'll likely start by trying in-context learning, using examples suited to our task and use case.

  There are still cases, however, where the model may not perform as well as we need, even with one or a few shot inference, and in that case, we can try **fine-tuning** our model. This supervised learning process of training LLMs.

  As models become more capable, it's becoming increasingly important to ensure that they behave well and in a way that is aligned with human preferences in deployment. An additional fine-tuning technique called **reinforcement learning with human feedback**, which can help to make sure that your model behaves well.

   An important aspect of all of these techniques is evaluation. We will explore some metrics and benchmarks that can be used to determine how well your model is performing or how well aligned it is to our preferences.

  _Note that this adapt and aligned stage of app development can be **highly iterative or repitative process** until we get model performance stable enough for our criteria and needs._ 
- **Application Integration**: At this stage, an important step is to **optimize our model for deployment**. Create front-end apps by using our customized LLMs.

**Limitations**: There are some fundamental limitations of LLMs that can be difficult to overcome through training alone like their tendency to invent information when they don't know an answer, or their limited ability to carry out complex reasoning and mathematics.
  

# Transformers

LLMs-> more parameters -> more memory ->better models

So it's important to understand difference between **Parameters** & **Hyperparameters**

- Parameters are variables that are learned by the model from the data, such as **weights** and **biases**. They allow the model to **learn the rules** from the data. Hence why models with billions of parameters are performing really Good(e.g. GPT(175B),BLOOM(175B)) vs model with millions of parameters(e.g. Bert(110M))

- Hyperparameters are variables that are set manually before training, such as **learning rate, batch size, number of layers**, etc. They control how the model is trained and how the parameters are updated.

Some examples of parameters are the **attention weights, the feed-forward network weights, and the embeddings**. 

Some examples of hyperparameters are the **number of heads, the hidden size, the dropout rate, and the warmup steps**.

Some of LLM's 

![image](https://github.com/ShankarChavan/Gen-AI-LLMs/assets/6409350/61b4fc16-8887-4034-a6da-a5d28a4fb982)


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

## In-depth Understanding of Transformer architecture step by step 

# prompt engineering and prompting types
# Generative Configurations
# Fine-tuning LLM's with PEFT & LoRA
# Fine-tuning LLM's with RLHF
# LLM's in applications




# Gen-AI-LLMs
This repo will hold all docs and code w.r.t Gen-AI-LLM materials


# Gen-AI-Keywords

1.**LLM** - Large Language Models (LLMs) are foundational machine learning models that use deep learning algorithms to process and understand natural language. These models are trained on massive amounts of text data to learn patterns and entity relationships in the language

2.**Tokenizer** - The first step in any NLP task is to convert given text into tokens. LLMs use 2 types — BPE (Byte pair encoding) and Wordpiece. GPT based models use BPE and BERT based models use WordPiece. Some models also use SentencePiece to get sentence encodings. Below is hierarchy of Tokenization methods

![](assets/Tokenization_type.png)

3.**Transformer** - A transformer model is a neural network that learns context and thus meaning by tracking relationships in sequential data like the words in this sentence.

4.**Attention** - In LLM, this concept is applied by teaching the model to focus on certain parts of the input data and disregard others to better solve the task at hand.

5.**GPT** - Generative Pre-trained Transformers, commonly known as GPT, are a family of neural network models that uses the transformer architecture

6.**Prompt** - Prompts are the inputs or queries that a user or a program gives to an LLM AI, to get a relevant response from the model.

7.**Prompt Engineering** - Prompt engineering is the process of enhancing the output of large language models (LLMs) like ChatGPT. Carefully crafting input prompts can help the language model understand the information about the input (context) and your desired output.

8.**Hallucination** - LLMs will also sometimes confidently produce information that isn’t real or true, which are typically called “hallucinations”

9.**Context window** - In GPT models, the context window refers to the amount of preceding text that the model can consider when generating a response.

10.**Parameters** - Parameters are the numerical values that chosen by the model. They are learned from data during the training process. The more parameters a model has, the more the model is.

11. **Hyperparameter** - Hyperparameters are the explicitly specified parameters that control the training process. They are essential for optimizing the model. They are set manually by Data Scientist / Machine learning engineer in beginning of training the model


# Gen-AI Lifecycle
Here is diagram of the Gen-AI Lifecycle

![](/assets/Lifecycle_LLMs.png)


- **Scope**: Defining the scope of LLM as accurately and narrowly is very important w.r.t use-case because LLM's are capable of carrying out multiple tasks based on the size and architecture of model.
  Getting really specific about what you need your model to do can save you **time** and **compute cost**

  Examples of Specific tasks can be Q & A bot, Text Summarization, or Named-Entity recognition etc. 
- **Select**: In this stage it's important to decide whether to train our own model from scratch or work with an existing base model.
- **Adapt & Align Model**: With our model in hand, the next step is to assess its performance and carry out additional training if needed for our application.

  **Prompt engineering** can sometimes be enough to get our model to perform well, so we'll likely start by trying in-context learning, using examples suited to our task and use case.

  There are still cases, however, where the model may not perform as well as we need, even with one or a few shot inference, and in that case, we can try **fine-tuning** our model. This supervised learning process of training LLMs.

  As models become more capable, it's becoming increasingly important to ensure that they behave well and in a way that is aligned with human preferences in deployment. An additional fine-tuning technique called **reinforcement learning with human feedback**, which can help to make sure that your model behaves well.

   An important aspect of all of these techniques is evaluation. We will explore some metrics and benchmarks that can be used to determine how well your model is performing or how well aligned it is to our preferences.

  _Note that this adapt and aligned stage of app development can be **highly iterative or repetitive process** until we get model performance stable enough for our criteria and needs._ 
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

![image](assets/LLM_models.png?raw=true)


 *“[Attention is All You Need](https://arxiv.org/abs/1706.03762)” by Vaswani et al. (2017) was the paper which introduced transformer architecture*

## Transformers model key-highlights
  - Scale efficiently to use multi-core GPU's. 
  - Parallel processing of input data and thus making use of much larger training datasets. 
  - Learn to pay attention to the meaning of the words it's processing

The **Power** of LLM's comes from model architecture which was used to train this kind of models vs old architecture like RNN

The power of the transformer architecture lies in its ability to learn the relevance and context of all of the words in a sentence and not just the neighbors.


![image](assets/sentence_example_sequence.png)

It applies attention weights to those relationships so that the model learns the relevance of each word to each other words no matter where they are in the input.

![image](assets/sentence_example_sequence_attn.png)

Based on this sentence itself model has ability to answer following questions:

 - Who has the book?
 - Who could have the book? 

In a nutshell it has ability to understand the context of document given to it.

![image](assets/sentence_example_sequence_attn_nodes.png)

In the above diagram we can see that word **book** is strongly connected with or paying attention to the word **teacher** and the word **student**. 

This is called **self-attention** and the ability to learn a attention in this way across the whole input significantly approves the model's ability to encode language. 

Self-attention is the key attributes of the transformer architecture.Let's dive in on transformer architecture diagram

### Transformer
![image](assets/transformer-architecture-complex.png)

### Simplified Transformer architecture 

![image](assets/transformer-architecture-simple-1.png)



![image](assets/transformer-architecture-simple-2.png)

# In-depth Understanding of Transformer architecture step by step 

## Types of transformer models

1. **Encoder only(Auto-encoding)**:
    Encoder models use only the encoder of a Transformer model. At each stage, the attention layers **can access all the words in the initial sentence**.

    **Objective**: The pretraining of these models usually revolves around somehow corrupting a given sentence (for instance, by masking random words in it) and tasking the model with finding or reconstructing the initial sentence.


    |Original text|The|teacher|teaches|the|student|
    |:-----|:--- |:------- |:------ |:--- |:------- |
    |MLM(Masked Language modeling)|The|teacher|\<mask>|the|student|
    |Reconstruct text(denoise)|The|teacher|teaches|the|student|
    |Bidirectional context|---|----->|teaches|<--|-------|


    **Use-cases**: 
    - Sentiment Analysis
    - Named Entity Recognition
    - Word Classification
    
    **Models**:
    - BERT
    - DistilBERT
    - RoBERTa
----
2. **Encoder-Decoder(Sequence to Sequence)**:
    Encoder-decoder models (also called sequence-to-sequence models) use both parts of the Transformer architecture. At each stage, the attention layers of the encoder can access all the words in the initial sentence, whereas the attention layers of the **_decoder_ can only access the words positioned before a given word in the input**.

    **Objective**: T5 is pretrained by replacing random spans of text (that can contain several words) with a single mask special word, and the objective is then to predict the text that this mask word replaces.

    |\<span corruption>|The|teacher|\<mask>|\<mask>|student|
    |:-----|:--- |:------- |:------ |:--- |:------- |
    |Sentinel token(mask words)|The|teacher|\<x>||student|
    |Reconstruct span|The|teacher|\<teaches>|\<the>|student|

    **Use-cases**:
    - Translation
    - Text Summarization
    - generative question answering

    **Models**:
    - BART
    - T
----
3. **Decoder only(autoregressive models)**:

    Decoder models use only the decoder of a Transformer model. At each stage, for a given word the **attention layers can only access the words positioned before it in the sentence**. These models are often called auto-regressive models.

    **Objective**:
    The pretraining of decoder models usually revolves around **predicting the next word** in the sentence.

    |Original text|The|teacher|teaches|the|student|
    |:-----|:--- |:------- |:------ |:--- |:------- |
    |Causal Language Modeling|The|teacher|?|?|?|
    |Predict Next word|The|teacher|_teaches_|_the_|_student_|
    |Unidirectional context|---|------|---->


    **Use-Cases**:
    - Text generation

    **Models**:
    - GPT
    - GPT2
    - BLOOM
    - BARD
    - CLAUDE
    - PaLM
    - LLAMA,LLAMA2


----
# History(Evolution tree) of LLM models

![Alt text](assets/image-1.png)

 The evolutionary tree of modern LLMs traces the development of language models in recent years and highlights some of the
most well-known models. Models on the same branch have closer relationships. 

Transformer-based models are shown in non-grey
colors: **decoder-only models in the blue** branch, **encoder-only models in the pink** branch, and **encoder-decoder models in the green**
branch. 

The vertical position of the models on the timeline represents their release dates. Open-source models are represented by
solid squares, while closed-source models are represented by hollow ones.

To view the animated view of [evolution tree click here](https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fraw.githubusercontent.com%2FMooler0410%2FLLMsPracticalGuide%2Fmain%2Fsource%2Ffigure_gif.pptx&wdOrigin=BROWSELINK)

The stacked bar plot in the bottom right corner shows the
number of models from various companies and institutions.

For more details you can refer to this [link](https://arxiv.org/pdf/2304.13712v2.pdf)

[Data-Centric AI](https://towardsdatascience.com/what-are-the-data-centric-ai-concepts-behind-gpt-models-a590071bb727) concepts behind GPT models

Try GPT-2 Transformer out on your own at live this url
https://transformer.huggingface.co/doc/gpt2-large


### Transformer model architecture from Scratch using pytorch [here](transformers-details/transformers_scratch.ipynb)






# prompt engineering and prompting types
# Generative Configurations
# Fine-tuning LLM's with PEFT & LoRA
# Fine-tuning LLM's with RLHF
# LLM's in applications

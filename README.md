<!-- TOC -->

- [Gen-AI Lifecycle](#gen-ai-lifecycle)
- [Transformers](#transformers)
    - [Transformers model key-highlights](#transformers-model-key-highlights)
        - [Transformer](#transformer)
        - [Simplified Transformer architecture](#simplified-transformer-architecture)
- [In-depth Understanding of Transformer architecture step by step](#in-depth-understanding-of-transformer-architecture-step-by-step)
    - [Types of transformer models](#types-of-transformer-models)
- [HistoryEvolution tree of LLM models](#historyevolution-tree-of-llm-models)
- [Transformer model code from Scratch](#transformer-model-code-from-scratch)
- [Prompt engineering and Prompting types](#prompt-engineering-and-prompting-types)
- [Generative AI Configurations](#generative-ai-configurations)
- [Fine-tuning LLM's with PEFT & LoRA](#fine-tuning-llms-with-peft--lora)
- [Fine-tuning LLM's with RLHF](#fine-tuning-llms-with-rlhf)
- [LLM's in applications](#llms-in-applications)

<!-- /TOC -->

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
    - T5
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


# Transformer model code from Scratch 
We will look into the actual implementation of transformer model code and its concept in detail for better understanding.
- [using pytorch](transformers-details/transformers_scratch_pytorch.ipynb)

- [using huggingface](transformers-details/simple_transformer_language_model.ipynb)


# Prompt engineering and Prompting types
  We will learn the following topics in prompt engineering & types
  1. What is [prompting, prompt engineering & In-Context learning?](prompt-engineering/prompting.md)
  
  2. Designing Prompts for different tasks

  * [**Text Summarization**](https://www.promptingguide.ai/introduction/examples#text-summarization)
  * [**Question Answering**](https://www.promptingguide.ai/introduction/examples#question-answering)
  * [**Text Classification**](https://www.promptingguide.ai/introduction/examples#text-classification)
  * [**Code Generation**](https://www.promptingguide.ai/introduction/examples#code-generation)
  * [**Reasoning**](https://www.promptingguide.ai/introduction/examples#reasoning)

  3. Prompt Engineering techniques
 
  * [**Few-shot Prompts**](https://www.promptingguide.ai/techniques/fewshot)
  * [**Chain-of-Thought CoT Prompting**](https://www.promptingguide.ai/techniques/cot)
  * [**Tree of thoughts(TOT)**](https://www.promptingguide.ai/techniques/tot)
  * [**Self-Consistency**](https://www.promptingguide.ai/techniques/consistency)
  * [**Generate Knowledge Prompting**](https://www.promptingguide.ai/techniques/knowledge)
  * [**ReAct**](https://www.promptingguide.ai/techniques/react)

  
  4. Applications
  - Progam-Aided learning (PAL)
  - Generating Data
  - Generating Code
  - prompt functions

  5. Risks
  - prompt Injection
  - prompt leaking
  - Jail-breaking

  6. [Reference guide](https://www.promptingguide.ai/)

[pdf download for prompting](resources/22-promptengg.pdf)

# Generative AI Configurations(or Inference Parameters)
![Alt text](assets/Infrence_config.png)



- **Temperature**: The temperature should be set according to the task  and domain expectations.
  A **higher temperature value of 0.7 to 0.9** may be desired, as it can produce more original and diverse texts.

- **Maximum length or tokens**:
  Set the word count, it makes your responses much cleaner. 
  
  Bear in mind that you can only return **2048 tokens**, or about 300–400 words per response. Anything longer may result in a response being cut off.
  
  Don't worry, just prompt "continue" and it should keep going (you may need to copy and paste the last sentence or two).

- **Top p**: A hyperparameter that controls the cumulative probability of the candidate tokens that the model can choose from. 

  A lower top p means that only the most probable tokens are considered, while a higher top p means that more tokens are considered. 

---
  _**Difference between Top p and Temperature**_

  ![Alt text](assets/top_p_vs_temp.png)


  Temperature and top_p are two parameters that affect the randomness of the output of a language model, such as GPT-3. 
  
  Temperature affects the **confidence of the model** in its top choices, while top_p affects the **number of choices** that the model considers. 
  
  A low temperature makes the output more **deterministic and less diverse**, while a high temperature makes the output more **stochastic and more diverse**. 
  
  Top_p sampling **selects only the tokens that have a cumulative probability mass** above a certain threshold.
  
  ![Alt text](assets/temp_vs_top_p.png)

  [Above table from openAI community blogpost](https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api-a-few-tips-and-tricks-on-controlling-the-creativity-deterministic-output-of-prompt-responses/172683)

---
- **Frequency penalty**: A hyperparameter that controls the **repetition of words or phrases in the generated text**. 

  A higher frequency penalty means less repetition, while a lower frequency penalty means more repetition. 


- **Presence penalty**: A hyperparameter that controls the **novelty of words or phrases in the generated text**. 

  A higher presence penalty means more novelty, while a lower presence penalty means more familiarity.  

  For example, for creative writing, a higher presence penalty value of 0.6 to 0.8 may be desired, as it can encourage the generation of new and original ideas. 
  
  For text summarization, a lower presence penalty value of 0.2 to 0.4 may be preferred, as it can ensure the consistency and relevance of the summaries.

  _**Difference between frequency vs Presence penalties**_
  ![Alt text](assets/freq_vs_pres_penalty.png)

[Article to understand difference between Frequency Vs Presence Penalty](https://medium.com/@KTAsim/frequency-vs-presence-penalty-whats-the-difference-openai-api-51b0c4a7229e) 


# Fine-tuning LLM's with PEFT(Parameter Efficient Fine-Tuning)

### Fine tune = update foundation model weights
(AKA parameter fine tuning)

- Update more layers = better model performance

- Full fine-tuning typically produces one model per task
  - Serve one model per task
  - May forget other pre-trained tasks: catastrophic forgetting

- Full fine-tuning LLMs is expensive. How to avoid it?
  - X-shot learning(we have seen this approach in prompt-engineering)
  - Parameter-efficient fine tuning
----
### Training LLM poses 2 main challenges
  - Increasing compute power
  - Increasing file size of model 

### PEFT and LoRa

PEFT is a method that employs various techniques, including LoRa, to efficiently fine-tune large language models. 

LoRa(Low-Rank Adaptation) focuses on **adding extra weights to the model while freezing most of the pre-trained network’s parameters**. This approach helps _prevent catastrophic forgetting_, a situation where models forgets what they were originally trained on during the full fine-tuning process.

### Background 
The [Research Paper](https://arxiv.org/abs/2106.09685) about LoRa was published by microsoft researchers in 2021.A library named [Loralib](https://github.com/microsoft/LoRA) was also created on github and later in Feb-2023 it was supported by **PEFT** library from _HuggingFace_.


### Background on Fine tuning
![Alt text](assets/finetuning_overview.png)




### SFT Finetuning approaches

Following are the finetuning approaches

![Alt text](assets/finetuning_approaches.png)


## **what is Instruction fine-tuning?**

![Alt text](assets/finetune_1.png)

## What are task specific Instructions or examples

**1. Example classify the review**
![Alt text](assets/finetune_example1.png)

**2. Example Summarize or Translate the sentence**

![Alt text](assets/finetune_example2.png)

For this we first need to Generate data for task specific examples.

We can use [Prompt instruction template](https://github.com/bigscience-workshop/promptsource) to generate the instruction data.


![Alt text](assets/instruction-data-template-example.png)

Let's look at how the instruction data looks like on HuggingFace

- [dolly-data-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

- [dialog-summarization-12.5k](https://huggingface.co/datasets/knkarthick/dialogsum)

### LoRa(Low Rank Adaption)?

![Alt text](assets/Lora_overview.png)

We freeze the existing weights of pretrained model and perform the training on the new instruction data using the LoRa.  

### What is Rank of Matrix?

![Alt text](rank_matrix.png)

[Blog to understand how to calculate Rank of Matrix](https://byjus.com/jee/rank-of-a-matrix-and-special-matrices/)

![Alt text](assets/Lora_matrix_decompose.png)

![Alt text](assets/Lora_performance_vs_parameter.png)

![Alt text](assets/Lora_perf_task_specific.png)

We can see that after **Rank-8** there is not much of an improvement and accuracy remain consistent.

Now we can start with Code-walkthrough in notebook.

- [code for finetuning flan-T5 model](fine-tuning/finetuning_FlanT5.ipynb)

----
As a Next step we can even automate the fine-tuning step even further using [HuggingFace AutoTrian](https://huggingface.co/docs/autotrain/index) library.


```
pip install autotrain-advanced
```

Once we have pip installed the autotrain-advanced and kept our data in `data` folder in local system. Use the following code 

```python
# set the hyperparameter and variables

project_name = 'my_autotrain_llm'
model_name = 'user/llama-2-7b-hf-small-shards' 

learning_rate = 2e-4 
num_epochs = 1 
batch_size = 1 
block_size = 1024 
trainer = "sft" 
warmup_ratio = 0.1 
weight_decay = 0.01 
gradient_accumulation = 4 
use_fp16 = True 
use_peft = True 
use_int4 = True 
lora_r = 16 
lora_alpha = 32 
lora_dropout = 0.05 

# store all parameters in environment variable
import os
os.environ["PROJECT_NAME"] = project_name
os.environ["MODEL_NAME"] = model_name
os.environ["PUSH_TO_HUB"] = str(push_to_hub)
os.environ["HF_TOKEN"] = hf_token
os.environ["REPO_ID"] = repo_id
os.environ["LEARNING_RATE"] = str(learning_rate)
os.environ["NUM_EPOCHS"] = str(num_epochs)
os.environ["BATCH_SIZE"] = str(batch_size)
os.environ["BLOCK_SIZE"] = str(block_size)
os.environ["WARMUP_RATIO"] = str(warmup_ratio)
os.environ["WEIGHT_DECAY"] = str(weight_decay)
os.environ["GRADIENT_ACCUMULATION"] = str(gradient_accumulation)
os.environ["USE_FP16"] = str(use_fp16)
os.environ["USE_PEFT"] = str(use_peft)
os.environ["USE_INT4"] = str(use_int4)
os.environ["LORA_R"] = str(lora_r)
os.environ["LORA_ALPHA"] = str(lora_alpha)
os.environ["LORA_DROPOUT"] = str(lora_dropout)

```

run the below code in shell or notebook

```sh
!autotrain llm \
--train \
--model ${MODEL_NAME} \
--project-name ${PROJECT_NAME} \
--data-path data/ \
--text-column text \
--lr ${LEARNING_RATE} \
--batch-size ${BATCH_SIZE} \
--epochs ${NUM_EPOCHS} \
--block-size ${BLOCK_SIZE} \
--warmup-ratio ${WARMUP_RATIO} \
--lora-r ${LORA_R} \
--lora-alpha ${LORA_ALPHA} \
--lora-dropout ${LORA_DROPOUT} \
--weight-decay ${WEIGHT_DECAY} \
--gradient-accumulation ${GRADIENT_ACCUMULATION} 
```


# Fine-tuning LLM's with RLHF

## Why fine-tune LLM's with RLHF?
Because Fine-tunning LLM further with Human feedback will ensure that model is more aligned with Human Values and not toxic in nature.

In year 2020 below was finding from OpenAI 

![Alt text](assets/RLHF_findings_1.png)


We can observe that fine-tuning with Human feedback was improving model performance when compared with Initial fine-tuning and No-fine-tuning.

Below is complete pipeline view of building fine-tuned LLM with RLHF to create very specific LLM based on Human values. Applications of same can be many like Individualized learning Plan bot, Personal Bot etc. 

![Alt text](assets/RLHF_process_1.png)

## What is Reinforcement Learning?

In RL we have an agent and environment where **Objective** is to maximize the reward received for actions.

$r{_t}$ is a reward that agent gets for taking action $a{_t}$ given the current state $s{_t}$ within the evironment

![Alt text](assets/RL_Intro.png)

Example of RL in tic-tac-toe

![Alt text](assets/RL_tic_tac_toe.png)


![Alt text](assets/RL_fine-tune-LLMs.png)

**Reward Model**: This is custom trained ML model which gives the reward to the agent based on the response generated by LLM and how closely it is matching with expected response.

**Example**: A custom toxicity classification model can be trained on text data which can then be used to evaluate the LLM response for toxicity.

_**Note**: Training a reward model requires us to prepare the data with human feedback for each of the response with rank ideally called as human rank completion pair_.

RL policy or algorithm which was used in RLHF was **PPO(Proximal Policy Optimization)**

Proximal Policy Optimization (PPO) is a popular model-free reinforcement learning algorithm that is used to train agents to perform tasks in an environment. It is an iterative algorithm that improves the policy through **trial and error**.

For detail understanding PPO you can refer to [this link](https://huggingface.co/blog/deep-rl-ppo).

## Reward Model

![Alt text](assets/RL_reward_model.png)

Finally complete flow of LLM model weight update is as follows:


![Alt text](assets/RL_fine-tune-reward-model_flow.png)

![Alt text](assets/RL_fine-tune-reward-model_flow2.png)

![Alt text](assets/RL_fine-tune-reward-model_flow3.png)

![Alt text](assets/RL_fine-tune-reward-model_flow4.png)


We can stop the training by setting max number of iterations or by threshold value of helpful response. 

Usually ~20,000 iterations should be good enough.  

# LLM apps (LLM's in application)
## RAG (Retrieval Augmented Generation)

![Alt text](assets/LLM_embeddings_gen_seq.png)

 ### Ingestion Procress in RAG
  - Take a set of proprietary documents
  - Split them up into smaller chunks
  - Create an embedding for each document

![Alt text](assets/LLM_RAG_Query.png)

### Query Process in RAG
  - Create an embedding for the query
  - Find the most similar documents in the embedding space
  - Pass those documents, along with the original query, into a language model to generate an answer

Let's Look at live examples of Chatbot with using pdf document.

Code is in the folder llm-apps/llama2-using-chainlit 

Tools we are going to leverage
  - Langchain for QARetriever,Pdf,textsplit
  - FAISS for vectorStorage DB
  - HuggingFaceEmbeddings(Sentence_transformers) for creating embeddings of text chunks
  - Chainlit for Chatbot interface




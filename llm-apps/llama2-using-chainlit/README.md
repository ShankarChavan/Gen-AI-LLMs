# Llama2-pdf-Chatbot
This is a pdf Chat bot built using Llama2 and Sentence Transformers. 

The bot is powered by Langchain and Chainlit. The bot runs on a decent CPU machine with a minimum of 16GB of RAM.

To run the code follow the below steps

```
1. git clone repo
2. Download the llama2 7B model from huggingface(llama-2-7b-chat.ggmlv3.q8_0.bin)
3. Install packages using pip install -r requirements.txt
4. Keep a pdf file in the data folder
5. run python ingest.py to create embeddings chunks of pdf file
6. run chainlit run bot.py 
```
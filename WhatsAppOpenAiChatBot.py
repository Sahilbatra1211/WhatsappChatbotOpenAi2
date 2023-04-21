#Note: The openai-python library support for Azure OpenAI is in preview.
import os
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://whatsappopenaichattraining.openai.azure.com/"
from langchain.llms import AzureOpenAI




llm = AzureOpenAI(deployment_name="OpenAiWhatsAppChat", 
                  temperature=0.9, 
                  max_tokens=10,
                  top_p=0.5,
                  frequency_penalty=0,
                  presence_penalty=0,
                  best_of=1)


start_phrase = 'Write a tagline for an ice cream shop.'
print(llm(start_phrase))

#using prompt and chains
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["food"],
    template="What is 5 locations with best {food}")

print(prompt.format(food="dessert"))

print(llm(prompt.format(food="dessert")))

chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("fruit"))

#integrating google results (skipped)
#personalized coversations (history is remembered here)

# using custom data 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import VectorDBQA
from langchain.document_loaders import DirectoryLoader
import magic
import nltk

llm2 = AzureOpenAI(deployment_name="OpenAiWhatsAppChat")
loader = DirectoryLoader('C:/Users/sahilbatra/Downloads/Chats', glob="**/*.txt")
documents = loader.load()
result2=documents[0].page_content
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(chunk_size=1)
docsearch  = Chroma.from_documents(texts, embeddings)
qa = VectorDBQA.from_chain_type(llm=llm2, chain_type="stuff",vectorstore=docsearch)

#prompt2 = PromptTemplate(
#    input_variables=["conversation"],
#    template="Harshita said {conversation}. Consider yourself as sahil, first come up with an answer from your existing knowledge then convert that sentence how sahil would have said looking at the data I have given you. And reply as sahil and sahil usually speaks in hindi and all the chats starting from sahil: are sahils words use them. Never talk in english")

prompt2 = PromptTemplate(
    input_variables=["conversation"],
    template="Analyze the chat conversation and tell ")

result= qa.run("Talk to me like you are harhita in her style.Sahil asked: naah heacdache toh nahi hai")
result
result.replace("/n", "<br/>")


#import openai
#openai.api_type = "azure"
#openai.api_base = "https://whatsappopenaichattraining.openai.azure.com/"
#openai.api_version = "2022-12-01"
#openai.api_key = "0720982d9208430ebad5ce70c4e3e307"

#response = openai.Completion.create(
#  engine="OpenAiWhatsAppChat",
#  prompt=start_phrase,
#  temperature=1,
#  max_tokens=10,
#  top_p=0.5,
#  frequency_penalty=0,
#  presence_penalty=0,
#  best_of=1,
#  stop=None)

#text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
#print(start_phrase+text)
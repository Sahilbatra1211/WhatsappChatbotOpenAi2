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


result= qa.run("Ask your question here")
result
result.replace("/n", "<br/>")
from langchain.chains import StuffDocumentsChain,LLMChain,ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms import ctransformers
from langchain.vectorstores import Chroma
import chromadb
import yaml

with open("config.yml","r") as f:
    config = yaml.safe_load(f)
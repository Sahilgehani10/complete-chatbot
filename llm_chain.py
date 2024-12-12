from langchain.chains import StuffDocumentsChain,LLMChain,ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import yaml
import chromadb
from prompt_templates import memory_prompt_template

with open("config.yml","r") as f:
    config = yaml.safe_load(f)

def create_llm(model_path=config["model_path"]["large"],model_type=config["model_type"],model_config=config["model_config"]):
    llm = CTransformers(model=model_path,model_type=model_type)
    return llm   

def create_embeddings(embeddings_path=config["embeddings_path"]):
    return HuggingFaceBgeEmbeddings(model_name=embeddings_path)

def create_llm_chain(llm,chat_prompt,memory):
    return LLMChain(llm=llm,prompt=chat_prompt,memory=memory)
def create_chat_memory(chat_history):
    return ConversationBufferMemory(memory_key="history",chat_memory=chat_history,k=3)


def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)

def load_normal_chain(chat_history):
    return chatChain(chat_history)

def load_vectordb(embeddings):
    persistent_client=chromadb.PersistentClient("chroma db")

    langchain_chroma=Chroma(
        client=persistent_client,
        collection_name="pdfs",
        embedding_function=embeddings,
    )

    return langchain_chroma

def load_pdf_chat_chain(chat_history):
    return pdfChatChain(chat_history)

def load_retrieval_chain(llm, memory, vector_db):
    return RetrievalQA.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(
            search_kwargs={
                "k": 3,
                "fetch_k": 5,  # Fetch more documents initially
                "maximal_marginal_relevance": True,  # Use MMR for diversity
            }
        ),
        memory=memory,
        return_source_documents=True,  # Optional: return sources
        max_tokens_limit=1024  # Adjust based on your model's context window
    )

class pdfChatChain:

    def __init__(self,chat_history):
        self.memory = create_chat_memory(chat_history)
        self.vector_db=load_vectordb(create_embeddings())
        llm=create_llm()
        chat_prompt=create_prompt_from_template(memory_prompt_template)
        self.llm_chain = load_retrieval_chain(llm,self.memory,self.vector_db)

    def run(self,user_input):
        return self.llm_chain.run(query=user_input,history=self.memory.chat_memory.messages,stop=["Human:"])
        

class chatChain:
    def __init__(self,chat_history):
        self.memory = create_chat_memory(chat_history)
        llm=create_llm()
        chat_prompt=create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm,chat_prompt,self.memory)

    def run(self,user_input):
        return self.llm_chain.run(human_input=user_input,history=self.memory.chat_memory.messages,stop=["Human:"])
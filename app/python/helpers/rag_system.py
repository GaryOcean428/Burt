from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import os

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

class RAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # Update to use the 3072-dimensional model
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.vectorstore = LangchainPinecone.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=os.getenv("PINECONE_NAMESPACE", "")
        )
        self.llm = OpenAI(temperature=0)
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.vectorstore.as_retriever()
        )

    def query(self, question: str) -> str:
        return self.qa.run(question)

    def add_document(self, text: str, metadata: dict = None):
        self.vectorstore.add_texts([text], metadatas=[metadata] if metadata else None)

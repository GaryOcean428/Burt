from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from .perplexity_search import perplexity_search, select_sonar_model
import os
import logging

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        )
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.dimension = int(os.getenv("PINECONE_DIMENSION", 1536))
        self.cloud = os.getenv("PINECONE_CLOUD", "aws")
        self.region = os.getenv("PINECONE_ENVIRONMENT")

        # Initialize Pinecone index if it doesn't exist
        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="euclidean",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )

        self.vectorstore = LangchainPinecone.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=os.getenv("PINECONE_NAMESPACE", ""),
        )
        self.llm = OpenAI(temperature=0)
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.vectorstore.as_retriever()
        )

    def query(self, question: str, use_perplexity: bool = False) -> str:
        try:
            if use_perplexity:
                complexity = self.assess_complexity(question)
                return perplexity_search(question, complexity=complexity)
            return self.qa.invoke(question)
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"

    def add_document(self, text: str, metadata: dict = None):
        try:
            self.vectorstore.add_texts(
                [text], metadatas=[metadata] if metadata else None
            )
            # Note: Perplexity doesn't provide a direct way to update its knowledge base
            # You might want to implement a separate system to keep track of added documents
            # for potential retraining or fine-tuning of the Perplexity model in the future
        except Exception as e:
            logger.error(f"Error adding document to RAG system: {str(e)}")

    def hybrid_query(self, question: str) -> str:
        try:
            # Attempt to use Pinecone-based retrieval first
            pinecone_result = self.qa.invoke(question)

            # If Pinecone result is insufficient, fall back to Perplexity search
            if (
                not pinecone_result or len(pinecone_result) < 50
            ):  # Adjust this threshold as needed
                complexity = self.assess_complexity(question)
                perplexity_result = perplexity_search(
                    question, complexity=complexity, stream=True
                )

                # Combine streaming results into a single string
                perplexity_result = "".join(perplexity_result)

                return f"Pinecone: {pinecone_result}\n\nPerplexity: {perplexity_result}"

            return pinecone_result
        except Exception as e:
            logger.error(f"Error in hybrid query: {str(e)}")
            return f"An error occurred while processing your hybrid query: {str(e)}"

    def assess_complexity(self, question: str) -> float:
        # This is a simple complexity assessment. You might want to implement a more sophisticated method.
        words = question.split()
        word_count = len(words)
        avg_word_length = (
            sum(len(word) for word in words) / word_count if word_count > 0 else 0
        )

        complexity = (word_count / 50) * 0.5 + (avg_word_length / 10) * 0.5
        return min(max(complexity, 0), 1)  # Ensure complexity is between 0 and 1

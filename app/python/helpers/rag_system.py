from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from .perplexity_search import perplexity_search
from .pinecone_db import (
    upsert_vectors,
    query_vectors,
    delete_vectors,
    check_pinecone_health,
)
from .redis_cache import RedisCache
import os
import logging
import json
from typing import List, Dict, Any
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        )
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.dimension = int(os.getenv("PINECONE_DIMENSION", "1536"))
        self.cloud = os.getenv("PINECONE_CLOUD", "aws")
        self.region = os.getenv("PINECONE_ENVIRONMENT")

        if check_pinecone_health():
            self.vectorstore = LangchainPinecone.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=os.getenv("PINECONE_NAMESPACE", ""),
            )
            self.llm = OpenAI(temperature=0)
            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(),
            )
        else:
            logger.error(
                "Pinecone is not available. RAG system may not function properly."
            )

    def query(self, question: str, use_perplexity: bool = False) -> str:
        try:
            # Check cache first
            cache_key = f"rag_query:{question}"
            cached_result = RedisCache.get(cache_key)
            if cached_result:
                logger.info(
                    f"Retrieved cached result for question: {question[:50]}..."
                )
                return cached_result

            if use_perplexity:
                complexity = self.assess_complexity(question)
                perplexity_result = perplexity_search(
                    question, complexity=complexity
                )
                result = str(perplexity_result)
            else:
                qa_result = self.qa.invoke(question)
                result = str(qa_result)

            # Cache the result
            RedisCache.set(
                cache_key, result, expiration=3600
            )  # Cache for 1 hour

            return result
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}", exc_info=True)
            return f"An error occurred while processing your query: {str(e)}"

    def add_document(self, text: str, metadata: dict = None):
        try:
            # Generate embedding
            embedding = self.embeddings.embed_query(text)

            # Prepare vector for upsert
            vector = {
                "id": f"doc_{len(text)}_{hash(text)}",  # Generate a unique ID
                "values": embedding,
                "metadata": metadata or {},
            }

            # Upsert vector to Pinecone
            upsert_vectors([vector])

            logger.info(
                f"Successfully added document to RAG system. Metadata: {metadata}"
            )
        except Exception as e:
            logger.error(
                f"Error adding document to RAG system: {str(e)}", exc_info=True
            )

    def hybrid_query(self, question: str) -> str:
        try:
            # Check cache first
            cache_key = f"hybrid_query:{question}"
            cached_result = RedisCache.get(cache_key)
            if cached_result:
                logger.info(
                    f"Retrieved cached hybrid query result for question: {question[:50]}..."
                )
                return cached_result

            # Attempt to use Pinecone-based retrieval first
            pinecone_result = self.qa.invoke(question)

            # If Pinecone result is insufficient, fall back to Perplexity search
            if (
                not pinecone_result or len(str(pinecone_result)) < 50
            ):  # Adjust this threshold as needed
                complexity = self.assess_complexity(question)
                perplexity_result = perplexity_search(
                    question, complexity=complexity, stream=True
                )

                # Combine streaming results into a single string
                perplexity_result = "".join(perplexity_result)

                result = f"Pinecone: {pinecone_result}\n\nPerplexity: {perplexity_result}"
            else:
                result = str(pinecone_result)

            # Cache the result
            RedisCache.set(
                cache_key, result, expiration=3600
            )  # Cache for 1 hour

            return result
        except Exception as e:
            logger.error(f"Error in hybrid query: {str(e)}", exc_info=True)
            return f"An error occurred while processing your hybrid query: {str(e)}"

    def assess_complexity(self, query: str) -> float:
        # Tokenize the query
        tokens = word_tokenize(query.lower())

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

        # Calculate lexical diversity
        lexical_diversity = len(set(tokens)) / len(tokens) if tokens else 0

        # Calculate average word length
        avg_word_length = (
            sum(len(word) for word in tokens) / len(tokens) if tokens else 0
        )

        # Count special characters and numbers
        special_chars = sum(
            1
            for char in query
            if not char.isalnum() and char not in [" ", ".", ",", "!", "?"]
        )
        numbers = sum(1 for char in query if char.isdigit())

        # Calculate complexity score
        complexity = (
            (len(tokens) / 100) * 0.3  # Length factor
            + lexical_diversity * 0.3  # Vocabulary richness
            + (avg_word_length / 10) * 0.2  # Word complexity
            + (special_chars / len(query)) * 0.1  # Special character density
            + (numbers / len(query)) * 0.1  # Number density
        )

        return min(complexity, 1.0)

    def clear_cache(self):
        # This method can be called periodically to clear the cache
        # Implementation depends on your caching strategy
        pass

import re
from app.agent import Agent
from app.python.helpers.vdb import VectorDB, Document
from app.python.helpers import files
import os
from app.python.helpers.tool import Tool, Response
from app.python.helpers.print_style import PrintStyle
from chromadb.errors import InvalidDimensionException
from app.python.helpers.rag_system import RAGSystem
from app.python.helpers.redis_cache import RedisCache
from app.python.helpers.pinecone_db import upsert_vectors, query_vectors, delete_vectors
from app.python.helpers.mongodb_client import (
    insert_document,
    find_documents,
    delete_document,
)
from typing import List, Dict, Any
import uuid

db: VectorDB | None = None
rag_system = RAGSystem()


class Memory(Tool):
    def execute(self, **kwargs):
        result = ""

        try:
            if "query" in kwargs:
                threshold = float(kwargs.get("threshold", 0.1))
                count = int(kwargs.get("count", 5))
                result = search(self.agent, kwargs["query"], count, threshold)
            elif "memorize" in kwargs:
                result = save(self.agent, kwargs["memorize"])
            elif "forget" in kwargs:
                result = forget(self.agent, kwargs["forget"])
            elif "delete" in kwargs:
                result = delete(self.agent, kwargs["delete"])
            elif "upload" in kwargs:
                result = upload_file(self.agent, kwargs["upload"])
        except InvalidDimensionException as e:
            PrintStyle.hint(
                "If you changed your embedding model, you will need to remove contents "
                "of /memory "
                "directory."
            )
            raise

        return Response(message=result, break_loop=False)


def search(agent: Agent, query: str, count: int = 5, threshold: float = 0.1) -> str:
    vector = agent.get_embedding(query)
    results = query_vectors(vector, top_k=count)

    if not results.matches:
        return files.read_file("./prompts/fw.memories_not_found.md", query=query)

    memories = []
    for match in results.matches:
        if match.score < threshold:
            continue
        doc = find_documents("memories", {"vector_id": match.id})
        if doc:
            memories.append(f"ID: {doc['_id']}, Content: {doc['content']}")

    return "\n".join(memories)


def save(agent: Agent, text: str) -> str:
    vector = agent.get_embedding(text)
    vector_id = str(uuid.uuid4())
    upsert_vectors([{"id": vector_id, "values": vector}])

    document_id = insert_document(
        "memories", {"content": text, "vector_id": vector_id}
    ).inserted_id

    return files.read_file("./prompts/fw.memory_saved.md", memory_id=str(document_id))


def delete(agent: Agent, ids_str: str) -> str:
    ids = extract_guids(ids_str)
    docs = find_documents("memories", {"_id": {"$in": ids}})
    vector_ids = [doc["vector_id"] for doc in docs]

    delete_vectors(vector_ids)
    deleted = delete_document("memories", {"_id": {"$in": ids}}).deleted_count

    return files.read_file("./prompts/fw.memories_deleted.md", memory_count=deleted)


def forget(agent: Agent, query: str):
    initialize(agent)
    deleted = db.delete_documents_by_query(query)
    return files.read_file("./prompts/fw.memories_deleted.md", memory_count=deleted)


def upload_file(agent: Agent, file_path: str) -> str:
    content = files.read_file(file_path)
    vector = agent.get_embedding(content)
    vector_id = str(uuid.uuid4())
    upsert_vectors([{"id": vector_id, "values": vector}])

    document_id = insert_document(
        "memories",
        {"content": content, "vector_id": vector_id, "metadata": {"source": file_path}},
    ).inserted_id

    return files.read_file(
        "./prompts/fw.file_uploaded.md", file_path=file_path, memory_id=str(document_id)
    )


def initialize(agent: Agent):
    global db
    if not db:
        memory_dir = os.path.join("memory", agent.config["MEMORY_SUBDIR"])
        db = VectorDB(agent.config)


def extract_guids(text):
    return re.findall(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text
    )

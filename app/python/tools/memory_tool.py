import re
from app.agent import Agent
from app.python.helpers.vdb import VectorDB, Document
from app.python.helpers import files
import os
from app.python.helpers.tool import Tool, Response
from app.python.helpers.print_style import PrintStyle
from chromadb.errors import InvalidDimensionException

db: VectorDB | None = None


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


def search(agent: Agent, query: str, count: int = 5, threshold: float = 0.1):
    initialize(agent)
    docs = db.search_similarity_threshold(query, count, threshold)
    if len(docs) == 0:
        return files.read_file("./prompts/fw.memories_not_found.md", query=query)
    return "\n".join([f"ID: {doc.id}, Content: {doc.content}" for doc in docs])


def save(agent: Agent, text: str):
    initialize(agent)
    document_id = db.insert_document(text)
    return files.read_file("./prompts/fw.memory_saved.md", memory_id=document_id)


def delete(agent: Agent, ids_str: str):
    initialize(agent)
    ids = extract_guids(ids_str)
    deleted = db.delete_documents_by_ids(ids)
    return files.read_file("./prompts/fw.memories_deleted.md", memory_count=deleted)


def forget(agent: Agent, query: str):
    initialize(agent)
    deleted = db.delete_documents_by_query(query)
    return files.read_file("./prompts/fw.memories_deleted.md", memory_count=deleted)


def upload_file(agent: Agent, file_path: str):
    initialize(agent)
    content = files.read_file(file_path)
    document_id = db.insert_document(content, metadata={"source": file_path})
    return files.read_file(
        "./prompts/fw.file_uploaded.md", file_path=file_path, memory_id=document_id
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

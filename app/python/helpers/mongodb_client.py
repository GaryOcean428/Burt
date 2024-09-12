from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import os
import logging
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME")

client = None
db = None

try:
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DB_NAME]
    # Ping the database to check the connection
    client.admin.command("ismaster")
    logger.info("MongoDB connection established successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")

# Fallback file path
FALLBACK_FILE = "mongodb_fallback.json"

# Retry decorator
mongo_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionFailure, OperationFailure)),
    before_sleep=lambda retry_state: logger.info(
        f"Retrying MongoDB operation: attempt {retry_state.attempt_number}"
    ),
)


def check_mongodb_health():
    try:
        if client:
            client.admin.command("ismaster")
            logger.info("MongoDB health check passed")
            return True
        else:
            logger.error("MongoDB client is not initialized")
            return False
    except Exception as e:
        logger.error(f"MongoDB health check failed: {str(e)}")
        return False


@mongo_retry
def insert_document(collection_name: str, document: dict):
    try:
        if db:
            result = db[collection_name].insert_one(document)
            logger.info(f"Document inserted successfully: {result.inserted_id}")
            return result
        else:
            return fallback_insert(collection_name, document)
    except Exception as e:
        logger.error(f"Error inserting document: {str(e)}")
        return fallback_insert(collection_name, document)


@mongo_retry
def find_documents(collection_name: str, query: dict):
    try:
        if db:
            return list(db[collection_name].find(query))
        else:
            return fallback_find(collection_name, query)
    except Exception as e:
        logger.error(f"Error finding documents: {str(e)}")
        return fallback_find(collection_name, query)


@mongo_retry
def update_document(collection_name: str, query: dict, update: dict):
    try:
        if db:
            result = db[collection_name].update_one(query, {"$set": update})
            logger.info(
                f"Document updated successfully: {result.modified_count} document(s) modified"
            )
            return result
        else:
            return fallback_update(collection_name, query, update)
    except Exception as e:
        logger.error(f"Error updating document: {str(e)}")
        return fallback_update(collection_name, query, update)


@mongo_retry
def delete_document(collection_name: str, query: dict):
    try:
        if db:
            result = db[collection_name].delete_one(query)
            logger.info(
                f"Document deleted successfully: {result.deleted_count} document(s) deleted"
            )
            return result
        else:
            return fallback_delete(collection_name, query)
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return fallback_delete(collection_name, query)


# Fallback methods using local JSON file
def load_fallback_data():
    try:
        with open(FALLBACK_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_fallback_data(data):
    with open(FALLBACK_FILE, "w") as f:
        json.dump(data, f)


def fallback_insert(collection_name: str, document: dict):
    data = load_fallback_data()
    if collection_name not in data:
        data[collection_name] = []
    data[collection_name].append(document)
    save_fallback_data(data)
    logger.info("Document inserted into fallback storage")
    return {"inserted_id": len(data[collection_name]) - 1}


def fallback_find(collection_name: str, query: dict):
    data = load_fallback_data()
    if collection_name not in data:
        return []
    return [
        doc
        for doc in data[collection_name]
        if all(doc.get(k) == v for k, v in query.items())
    ]


def fallback_update(collection_name: str, query: dict, update: dict):
    data = load_fallback_data()
    if collection_name not in data:
        return {"modified_count": 0}
    modified_count = 0
    for doc in data[collection_name]:
        if all(doc.get(k) == v for k, v in query.items()):
            doc.update(update["$set"])
            modified_count += 1
    save_fallback_data(data)
    logger.info(
        f"Document updated in fallback storage: {modified_count} document(s) modified"
    )
    return {"modified_count": modified_count}


def fallback_delete(collection_name: str, query: dict):
    data = load_fallback_data()
    if collection_name not in data:
        return {"deleted_count": 0}
    original_length = len(data[collection_name])
    data[collection_name] = [
        doc
        for doc in data[collection_name]
        if not all(doc.get(k) == v for k, v in query.items())
    ]
    deleted_count = original_length - len(data[collection_name])
    save_fallback_data(data)
    logger.info(
        f"Document deleted from fallback storage: {deleted_count} document(s) deleted"
    )
    return {"deleted_count": deleted_count}


# Initialize fallback storage if MongoDB is not available
if not client:
    logger.warning("MongoDB is not available. Using local fallback storage.")
    if not os.path.exists(FALLBACK_FILE):
        save_fallback_data({})
        logger.info("Initialized empty fallback storage file.")

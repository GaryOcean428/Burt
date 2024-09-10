from pymongo import MongoClient
import os

client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("MONGODB_DB_NAME")]


def insert_document(collection_name: str, document: dict):
    return db[collection_name].insert_one(document)


def find_documents(collection_name: str, query: dict):
    return db[collection_name].find(query)


def update_document(collection_name: str, query: dict, update: dict):
    return db[collection_name].update_one(query, {"$set": update})


def delete_document(collection_name: str, query: dict):
    return db[collection_name].delete_one(query)

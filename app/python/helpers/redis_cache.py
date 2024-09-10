import redis
import os
import json

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    ssl=True
)

class RedisCache:
    @staticmethod
    def set(key: str, value: dict, expiration: int = 3600):
        redis_client.setex(key, expiration, json.dumps(value))

    @staticmethod
    def get(key: str) -> dict:
        value = redis_client.get(key)
        return json.loads(value) if value else None

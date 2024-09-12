import redis
import os
import json
import logging
from typing import Any, Optional
from collections import OrderedDict
import docker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis client
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),
        password=os.getenv("REDIS_PASSWORD"),
        ssl=True,
        socket_timeout=5,
    )
except Exception as e:
    logger.error(f"Failed to initialize Redis client: {str(e)}")
    redis_client = None

# Docker-based Redis fallback
DOCKER_REDIS_PORT = 6379
docker_client = docker.from_env()


def ensure_docker_redis():
    try:
        container = docker_client.containers.get("burton-redis-fallback")
        if container.status != "running":
            container.start()
            logger.info("Docker Redis container started")
    except docker.errors.NotFound:
        docker_client.containers.run(
            "redis:latest",
            name="burton-redis-fallback",
            ports={f"{DOCKER_REDIS_PORT}/tcp": DOCKER_REDIS_PORT},
            detach=True,
        )
        logger.info("Docker Redis container created and started")
    except Exception as e:
        logger.error(f"Error ensuring Docker Redis: {str(e)}")


# Ensure Docker Redis is running
ensure_docker_redis()

# Initialize Docker-based Redis client
try:
    docker_redis_client = redis.Redis(
        host="localhost",
        port=DOCKER_REDIS_PORT,
        socket_timeout=5,
    )
except Exception as e:
    logger.error(f"Failed to initialize Docker-based Redis client: {str(e)}")
    docker_redis_client = None


class RedisCache:
    # Local in-memory cache
    local_cache = OrderedDict()
    MAX_LOCAL_CACHE_SIZE = (
        1000  # Adjust this value based on your memory constraints
    )

    @classmethod
    def set(cls, key: str, value: Any, expiration: int = 3600) -> None:
        try:
            if redis_client and redis_client.ping():
                redis_client.setex(key, expiration, json.dumps(value))
                logger.info(f"Set key '{key}' in Redis cache")
            elif docker_redis_client and docker_redis_client.ping():
                docker_redis_client.setex(key, expiration, json.dumps(value))
                logger.info(f"Set key '{key}' in Docker Redis cache")
            else:
                cls._set_local(key, value)
                logger.info(
                    f"Set key '{key}' in local cache (Redis unavailable)"
                )
        except Exception as e:
            logger.error(f"Error setting cache for key '{key}': {str(e)}")
            cls._set_local(key, value)

    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        try:
            if redis_client and redis_client.ping():
                value = redis_client.get(key)
                if value:
                    logger.info(f"Retrieved key '{key}' from Redis cache")
                    return json.loads(value)
            elif docker_redis_client and docker_redis_client.ping():
                value = docker_redis_client.get(key)
                if value:
                    logger.info(
                        f"Retrieved key '{key}' from Docker Redis cache"
                    )
                    return json.loads(value)
            else:
                logger.warning(
                    "Redis unavailable, falling back to local cache"
                )
        except Exception as e:
            logger.error(
                f"Error retrieving from Redis cache for key '{key}': {str(e)}"
            )

        # Fallback to local cache
        return cls._get_local(key)

    @classmethod
    def _set_local(cls, key: str, value: Any) -> None:
        if len(cls.local_cache) >= cls.MAX_LOCAL_CACHE_SIZE:
            cls.local_cache.popitem(last=False)  # Remove oldest item
        cls.local_cache[key] = value

    @classmethod
    def _get_local(cls, key: str) -> Optional[Any]:
        value = cls.local_cache.get(key)
        if value:
            logger.info(f"Retrieved key '{key}' from local cache")
        else:
            logger.info(f"Key '{key}' not found in local cache")
        return value

    @staticmethod
    def check_redis_health() -> bool:
        try:
            if redis_client and redis_client.ping():
                logger.info("Redis health check passed")
                return True
            elif docker_redis_client and docker_redis_client.ping():
                logger.info("Docker Redis health check passed")
                return True
            else:
                logger.warning("Redis health check failed")
                return False
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return False


# Perform initial health check
if RedisCache.check_redis_health():
    logger.info("Redis connection established successfully")
else:
    logger.warning(
        "Failed to establish Redis connection. Using local cache as fallback."
    )

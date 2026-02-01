"""
Redis Client Module for BeautyAI Distributed Architecture

Provides Redis connection management for:
- Session state storage (cross-server session lookup)
- Server registry (master-slave coordination)
- Request queue (inference load balancing)
- Pub/Sub (heartbeat events, routing signals)

Author: BeautyAI Framework
Date: 2026-01-19
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

# Load .env file from backend directory
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed, rely on system environment

logger = logging.getLogger(__name__)

# Try to import redis, provide fallback if not installed
try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis
    from redis.asyncio.client import PubSub
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None
    Redis = None
    PubSub = None
    logger.warning("redis package not installed. Distributed features will be disabled.")


class ClusterMode(Enum):
    """Cluster operation mode."""
    STANDALONE = "standalone"  # Single server, no clustering
    MASTER = "master"          # Master server, coordinates slaves
    SLAVE = "slave"            # Slave server, registers with master


@dataclass
class ServerInfo:
    """Information about a server in the cluster."""
    server_id: str
    host: str
    port: int
    public_url: str  # Public URL for client redirect (important for cross-network)
    mode: str = "slave"
    capabilities: Dict[str, Any] = field(default_factory=lambda: {
        "llm_slots": 1,
        "stt": True,
        "tts": True
    })
    gpu_memory_total_gb: float = 16.0
    gpu_memory_used_gb: float = 0.0
    active_sessions: int = 0
    llm_active: int = 0
    llm_queued: int = 0
    last_heartbeat: float = 0.0
    status: str = "active"  # active, draining, offline
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerInfo":
        return cls(**data)


@dataclass
class ClusterConfig:
    """Configuration for cluster operation."""
    mode: ClusterMode = ClusterMode.STANDALONE
    redis_url: str = "redis://localhost:6379/0"
    master_url: Optional[str] = None  # For slaves: master's API URL
    server_id: Optional[str] = None   # Unique server identifier
    public_url: Optional[str] = None  # Public URL for client redirects
    heartbeat_interval_s: float = 5.0
    heartbeat_timeout_s: float = 15.0
    session_ttl_s: int = 3600         # 1 hour session TTL
    
    @classmethod
    def from_env(cls) -> "ClusterConfig":
        """Load cluster configuration from environment variables (supports .env file)."""
        mode_str = os.getenv("CLUSTER_MODE", "standalone").lower()
        mode = ClusterMode(mode_str) if mode_str in [m.value for m in ClusterMode] else ClusterMode.STANDALONE
        
        # Helper to get env var, treating empty string as None
        def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
            value = os.getenv(key, default)
            return value if value else default
        
        return cls(
            mode=mode,
            redis_url=get_env("REDIS_URL", "redis://localhost:6379/0"),
            master_url=get_env("MASTER_URL"),
            server_id=get_env("SERVER_ID") or f"server-{os.getpid()}",
            public_url=get_env("PUBLIC_URL"),
            heartbeat_interval_s=float(get_env("HEARTBEAT_INTERVAL_S", "5.0")),
            heartbeat_timeout_s=float(get_env("HEARTBEAT_TIMEOUT_S", "15.0")),
            session_ttl_s=int(get_env("SESSION_TTL_S", "3600")),
        )


class RedisClient:
    """
    Async Redis client for distributed BeautyAI operations.
    
    Handles:
    - Connection management with auto-reconnect
    - Session state storage
    - Server registry for cluster coordination
    - Pub/Sub for real-time events
    """
    
    _instance: Optional["RedisClient"] = None
    _lock = asyncio.Lock()
    
    def __init__(self, config: Optional[ClusterConfig] = None):
        self.config = config or ClusterConfig.from_env()
        self._redis: Optional[Redis] = None
        self._pubsub: Optional[PubSub] = None
        self._connected = False
        self._subscribers: Dict[str, List[Callable]] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    async def get_instance(cls, config: Optional[ClusterConfig] = None) -> "RedisClient":
        """Get singleton instance of RedisClient."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance
    
    async def connect(self) -> bool:
        """
        Connect to Redis server.
        
        Returns:
            bool: True if connection successful
        """
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis not available, running in standalone mode")
            return False
        
        if self.config.mode == ClusterMode.STANDALONE:
            self.logger.info("Running in standalone mode, Redis connection skipped")
            return False
        
        try:
            self._redis = await aioredis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
            )
            
            # Test connection
            await self._redis.ping()
            self._connected = True
            self.logger.info(f"✅ Connected to Redis: {self.config.redis_url}")
            
            # Initialize pub/sub
            self._pubsub = self._redis.pubsub()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Redis: {e}")
            self._connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Redis server."""
        try:
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            if self._pubsub:
                await self._pubsub.close()
            
            if self._redis:
                await self._redis.close()
            
            self._connected = False
            self.logger.info("Disconnected from Redis")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from Redis: {e}")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._connected and self._redis is not None
    
    # ========== Session Store Methods ==========
    
    async def store_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """
        Store session data in Redis.
        
        Args:
            session_id: Unique session identifier
            session_data: Session data to store
            
        Returns:
            bool: True if stored successfully
        """
        if not self.is_connected:
            return False
        
        try:
            key = f"cluster:sessions:{session_id}"
            await self._redis.hset(key, mapping={
                "data": json.dumps(session_data),
                "server_id": self.config.server_id,
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
            })
            await self._redis.expire(key, self.config.session_ttl_s)
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing session {session_id}: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data from Redis.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dictionary or None
        """
        if not self.is_connected:
            return None
        
        try:
            key = f"cluster:sessions:{session_id}"
            data = await self._redis.hgetall(key)
            
            if data and "data" in data:
                return {
                    **json.loads(data["data"]),
                    "_server_id": data.get("server_id"),
                    "_created_at": float(data.get("created_at", 0)),
                    "_updated_at": float(data.get("updated_at", 0)),
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting session {session_id}: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session from Redis."""
        if not self.is_connected:
            return False
        
        try:
            key = f"cluster:sessions:{session_id}"
            await self._redis.delete(key)
            return True
        except Exception as e:
            self.logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    async def update_session_server(self, session_id: str, new_server_id: str) -> bool:
        """Update session's assigned server (for migration)."""
        if not self.is_connected:
            return False
        
        try:
            key = f"cluster:sessions:{session_id}"
            await self._redis.hset(key, mapping={
                "server_id": new_server_id,
                "updated_at": str(time.time()),
            })
            return True
        except Exception as e:
            self.logger.error(f"Error updating session server: {e}")
            return False
    
    # ========== Server Registry Methods ==========
    
    async def register_server(self, server_info: ServerInfo) -> bool:
        """
        Register this server in the cluster registry.
        
        Args:
            server_info: Server information
            
        Returns:
            bool: True if registered successfully
        """
        if not self.is_connected:
            return False
        
        try:
            key = f"cluster:servers:{server_info.server_id}"
            server_info.last_heartbeat = time.time()
            
            await self._redis.hset(key, mapping={
                "data": json.dumps(server_info.to_dict()),
                "last_heartbeat": str(server_info.last_heartbeat),
            })
            
            # Add to active servers set
            await self._redis.sadd("cluster:active_servers", server_info.server_id)
            
            self.logger.info(f"✅ Server registered: {server_info.server_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering server: {e}")
            return False
    
    async def unregister_server(self, server_id: str) -> bool:
        """Remove server from cluster registry."""
        if not self.is_connected:
            return False
        
        try:
            key = f"cluster:servers:{server_id}"
            await self._redis.delete(key)
            await self._redis.srem("cluster:active_servers", server_id)
            self.logger.info(f"Server unregistered: {server_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error unregistering server: {e}")
            return False
    
    async def get_server(self, server_id: str) -> Optional[ServerInfo]:
        """Get server info by ID."""
        if not self.is_connected:
            return None
        
        try:
            key = f"cluster:servers:{server_id}"
            data = await self._redis.hget(key, "data")
            if data:
                return ServerInfo.from_dict(json.loads(data))
            return None
        except Exception as e:
            self.logger.error(f"Error getting server {server_id}: {e}")
            return None
    
    async def get_all_servers(self) -> List[ServerInfo]:
        """Get all registered servers."""
        if not self.is_connected:
            return []
        
        try:
            server_ids = await self._redis.smembers("cluster:active_servers")
            servers = []
            
            for server_id in server_ids:
                server = await self.get_server(server_id)
                if server:
                    servers.append(server)
            
            return servers
            
        except Exception as e:
            self.logger.error(f"Error getting all servers: {e}")
            return []
    
    async def get_available_servers(self) -> List[ServerInfo]:
        """
        Get servers with available capacity, sorted by load.
        
        Returns:
            List of servers with available LLM slots, sorted by load (lowest first)
        """
        servers = await self.get_all_servers()
        
        # Filter active servers with available capacity
        available = [
            s for s in servers
            if s.status == "active" 
            and s.llm_active < s.capabilities.get("llm_slots", 1)
            and (time.time() - s.last_heartbeat) < self.config.heartbeat_timeout_s
        ]
        
        # Sort by load (llm_active + llm_queued)
        available.sort(key=lambda s: s.llm_active + s.llm_queued)
        
        return available
    
    # ========== Heartbeat Methods ==========
    
    async def send_heartbeat(self, server_info: ServerInfo) -> bool:
        """Send heartbeat update for this server."""
        if not self.is_connected:
            return False
        
        try:
            key = f"cluster:servers:{server_info.server_id}"
            server_info.last_heartbeat = time.time()
            
            await self._redis.hset(key, mapping={
                "data": json.dumps(server_info.to_dict()),
                "last_heartbeat": str(server_info.last_heartbeat),
            })
            
            # Publish heartbeat event
            await self._redis.publish(
                "cluster:heartbeats",
                json.dumps({
                    "server_id": server_info.server_id,
                    "timestamp": server_info.last_heartbeat,
                    "llm_active": server_info.llm_active,
                    "llm_queued": server_info.llm_queued,
                    "active_sessions": server_info.active_sessions,
                })
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {e}")
            return False
    
    async def start_heartbeat_loop(self, server_info: ServerInfo, update_callback: Optional[Callable] = None):
        """
        Start background heartbeat loop.
        
        Args:
            server_info: Server info to update
            update_callback: Optional callback to update server_info before each heartbeat
        """
        async def heartbeat_loop():
            while True:
                try:
                    # Call callback to update stats
                    if update_callback:
                        await update_callback(server_info)
                    
                    await self.send_heartbeat(server_info)
                    await asyncio.sleep(self.config.heartbeat_interval_s)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Heartbeat error: {e}")
                    await asyncio.sleep(self.config.heartbeat_interval_s)
        
        self._heartbeat_task = asyncio.create_task(heartbeat_loop())
        self.logger.info("Heartbeat loop started")
    
    # ========== Pub/Sub Methods ==========
    
    async def subscribe(self, channel: str, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Subscribe to a pub/sub channel.
        
        Args:
            channel: Channel name
            callback: Async callback function for messages
        """
        if not self.is_connected or not self._pubsub:
            return False
        
        try:
            await self._pubsub.subscribe(channel)
            
            if channel not in self._subscribers:
                self._subscribers[channel] = []
            self._subscribers[channel].append(callback)
            
            self.logger.info(f"Subscribed to channel: {channel}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error subscribing to {channel}: {e}")
            return False
    
    async def publish(self, channel: str, message: Dict[str, Any]) -> bool:
        """Publish message to channel."""
        if not self.is_connected:
            return False
        
        try:
            await self._redis.publish(channel, json.dumps(message))
            return True
        except Exception as e:
            self.logger.error(f"Error publishing to {channel}: {e}")
            return False
    
    async def listen_messages(self):
        """Listen for pub/sub messages and dispatch to subscribers."""
        if not self._pubsub:
            return
        
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"]
                    data = json.loads(message["data"])
                    
                    for callback in self._subscribers.get(channel, []):
                        try:
                            await callback(data)
                        except Exception as e:
                            self.logger.error(f"Error in subscriber callback: {e}")
                            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in message listener: {e}")
    
    # ========== Request Queue Methods ==========
    
    async def enqueue_inference_request(self, request_id: str, request_data: Dict[str, Any]) -> bool:
        """
        Add inference request to queue.
        
        Args:
            request_id: Unique request identifier
            request_data: Request data including session_id, input, etc.
        """
        if not self.is_connected:
            return False
        
        try:
            await self._redis.lpush(
                "cluster:inference_queue",
                json.dumps({
                    "request_id": request_id,
                    "data": request_data,
                    "timestamp": time.time(),
                })
            )
            return True
        except Exception as e:
            self.logger.error(f"Error enqueuing request: {e}")
            return False
    
    async def dequeue_inference_request(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Get next inference request from queue.
        
        Args:
            timeout: Blocking timeout in seconds
            
        Returns:
            Request data or None if queue empty
        """
        if not self.is_connected:
            return None
        
        try:
            result = await self._redis.brpop("cluster:inference_queue", timeout=timeout)
            if result:
                return json.loads(result[1])
            return None
        except Exception as e:
            self.logger.error(f"Error dequeuing request: {e}")
            return None
    
    async def get_queue_length(self) -> int:
        """Get current inference queue length."""
        if not self.is_connected:
            return 0
        
        try:
            return await self._redis.llen("cluster:inference_queue")
        except Exception as e:
            self.logger.error(f"Error getting queue length: {e}")
            return 0


# Global instance getter
_redis_client: Optional[RedisClient] = None


async def get_redis_client(config: Optional[ClusterConfig] = None) -> RedisClient:
    """Get global Redis client instance."""
    global _redis_client
    
    if _redis_client is None:
        _redis_client = await RedisClient.get_instance(config)
    
    return _redis_client


async def initialize_redis(config: Optional[ClusterConfig] = None) -> bool:
    """Initialize Redis connection."""
    client = await get_redis_client(config)
    return await client.connect()


async def shutdown_redis() -> None:
    """Shutdown Redis connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.disconnect()
        _redis_client = None

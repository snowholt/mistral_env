"""
PersonaPlex Manager - Subprocess Lifecycle Management

Manages the PersonaPlex server as a subprocess, handling:
- Server startup/shutdown
- Health monitoring
- SSL certificate generation
- Session tracking
"""

import asyncio
import logging
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum

import aiohttp

from .constants import (
    PersonaPlexConfig,
    VoiceType,
    VOICE_PROMPTS,
    DEFAULT_TEXT_PROMPTS,
)

logger = logging.getLogger(__name__)


class ServerStatus(str, Enum):
    """PersonaPlex server status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class SessionInfo:
    """Information about an active PersonaPlex session."""
    session_id: str
    voice_prompt: str
    text_prompt: str
    started_at: float
    client_ip: Optional[str] = None


class PersonaPlexManager:
    """
    Singleton manager for PersonaPlex server lifecycle.
    
    PersonaPlex runs as a separate server process that handles WebRTC
    connections directly. This manager handles:
    
    - Starting/stopping the PersonaPlex server subprocess
    - Generating temporary SSL certificates for WebRTC
    - Health monitoring and auto-restart
    - Configuration management
    
    Usage:
        manager = get_personaplex_manager()
        await manager.start_server()
        status = manager.get_status()
        await manager.stop_server()
    """
    
    _instance: Optional["PersonaPlexManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Optional[PersonaPlexConfig] = None):
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config: Optional[PersonaPlexConfig] = None):
        """Initialize the manager."""
        if self._initialized:
            return
        
        self.config = config or PersonaPlexConfig()
        self._process: Optional[subprocess.Popen] = None
        self._status = ServerStatus.STOPPED
        self._ssl_dir: Optional[str] = None
        self._startup_time: Optional[float] = None
        self._error_message: Optional[str] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._active_sessions: Dict[str, SessionInfo] = {}
        
        # Load HF token from environment if not set
        if not self.config.hf_token:
            self.config.hf_token = os.getenv("HF_TOKEN")
        
        self._initialized = True
        logger.info("PersonaPlexManager initialized")
    
    @property
    def status(self) -> ServerStatus:
        """Get current server status."""
        return self._status
    
    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._status == ServerStatus.RUNNING and self._process is not None
    
    @property
    def server_url(self) -> str:
        """Get the server URL."""
        protocol = "https" if self.config.ssl_enabled else "http"
        return f"{protocol}://localhost:{self.config.port}"
    
    @property
    def webui_url(self) -> str:
        """Get the WebUI URL (PersonaPlex's built-in React UI)."""
        return self.server_url
    
    def _check_prerequisites(self) -> tuple[bool, str]:
        """Check if PersonaPlex is installed and ready."""
        
        # Check PersonaPlex path
        personaplex_path = Path(self.config.personaplex_path)
        if not personaplex_path.exists():
            return False, f"PersonaPlex not found at {personaplex_path}. Clone from https://github.com/NVIDIA/personaplex"
        
        # Check moshi module
        moshi_path = personaplex_path / "moshi"
        if not moshi_path.exists():
            return False, f"Moshi module not found at {moshi_path}"
        
        # Check HF token
        if not self.config.hf_token:
            return False, "HF_TOKEN environment variable not set. Required for model download."
        
        return True, "Prerequisites OK"
    
    def _create_ssl_dir(self) -> str:
        """Create temporary directory for SSL certificates."""
        if self._ssl_dir and Path(self._ssl_dir).exists():
            return self._ssl_dir
        
        self._ssl_dir = tempfile.mkdtemp(prefix="personaplex_ssl_")
        logger.info(f"Created SSL directory: {self._ssl_dir}")
        return self._ssl_dir
    
    def _build_command(self) -> List[str]:
        """Build the command to start PersonaPlex server."""
        cmd = [
            self.config.python_executable,
            "-m", "moshi.server",
        ]
        
        # SSL directory
        if self.config.ssl_enabled:
            ssl_dir = self._create_ssl_dir()
            cmd.extend(["--ssl", ssl_dir])
        
        # CPU offload for limited VRAM
        if self.config.cpu_offload:
            cmd.append("--cpu-offload")
        
        return cmd
    
    async def start_server(
        self,
        voice_prompt: Optional[VoiceType] = None,
        text_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start the PersonaPlex server.
        
        Args:
            voice_prompt: Voice to use (default: NATF2)
            text_prompt: Text prompt key or custom prompt
            
        Returns:
            Status dict with server URL and info
        """
        if self._status == ServerStatus.RUNNING:
            return {
                "success": True,
                "message": "Server already running",
                "url": self.server_url,
                "webui_url": self.webui_url,
                "status": self._status.value,
            }
        
        if self._status == ServerStatus.STARTING:
            return {
                "success": False,
                "message": "Server is starting, please wait",
                "status": self._status.value,
            }
        
        # Check prerequisites
        ready, message = self._check_prerequisites()
        if not ready:
            self._status = ServerStatus.ERROR
            self._error_message = message
            logger.error(f"PersonaPlex prerequisites not met: {message}")
            return {
                "success": False,
                "message": message,
                "status": self._status.value,
            }
        
        self._status = ServerStatus.STARTING
        self._error_message = None
        
        try:
            # Build and execute command
            cmd = self._build_command()
            logger.info(f"Starting PersonaPlex: {' '.join(cmd)}")
            
            # Set environment
            env = os.environ.copy()
            env["HF_TOKEN"] = self.config.hf_token
            
            # Change to PersonaPlex directory
            cwd = self.config.personaplex_path
            
            # Start process
            self._process = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            self._startup_time = time.time()
            
            # Start output reader thread
            output_thread = threading.Thread(
                target=self._read_output,
                daemon=True
            )
            output_thread.start()
            
            # Wait for server to be ready
            is_ready = await self._wait_for_ready()
            
            if is_ready:
                self._status = ServerStatus.RUNNING
                logger.info(f"✅ PersonaPlex server started at {self.server_url}")
                
                # Start health check task
                self._health_check_task = asyncio.create_task(self._health_check_loop())
                
                return {
                    "success": True,
                    "message": "PersonaPlex server started successfully",
                    "url": self.server_url,
                    "webui_url": self.webui_url,
                    "status": self._status.value,
                    "pid": self._process.pid,
                    "cpu_offload": self.config.cpu_offload,
                }
            else:
                await self.stop_server()
                self._status = ServerStatus.ERROR
                self._error_message = "Server failed to start within timeout"
                return {
                    "success": False,
                    "message": self._error_message,
                    "status": self._status.value,
                }
                
        except Exception as e:
            self._status = ServerStatus.ERROR
            self._error_message = str(e)
            logger.error(f"❌ Failed to start PersonaPlex: {e}")
            return {
                "success": False,
                "message": str(e),
                "status": self._status.value,
            }
    
    def _read_output(self):
        """Read and log subprocess output."""
        if not self._process or not self._process.stdout:
            return
        
        try:
            for line in self._process.stdout:
                line = line.strip()
                if line:
                    logger.info(f"[PersonaPlex] {line}")
        except Exception as e:
            logger.error(f"Error reading PersonaPlex output: {e}")
    
    async def _wait_for_ready(self) -> bool:
        """Wait for server to be ready."""
        start = time.time()
        timeout = self.config.startup_timeout
        
        while time.time() - start < timeout:
            if self._process and self._process.poll() is not None:
                # Process exited
                logger.error("PersonaPlex process exited during startup")
                return False
            
            # Try to connect
            try:
                async with aiohttp.ClientSession() as session:
                    ssl = False if not self.config.ssl_enabled else False  # Skip SSL verification for self-signed
                    async with session.get(
                        f"{self.server_url}/",
                        ssl=ssl,
                        timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        if resp.status in (200, 404):  # 404 is OK, means server is up
                            return True
            except Exception:
                pass
            
            await asyncio.sleep(1)
        
        return False
    
    async def _health_check_loop(self):
        """Periodically check server health."""
        while self._status == ServerStatus.RUNNING:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                if self._process and self._process.poll() is not None:
                    logger.error("PersonaPlex process died unexpectedly")
                    self._status = ServerStatus.ERROR
                    self._error_message = "Process died unexpectedly"
                    break
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def stop_server(self) -> Dict[str, Any]:
        """Stop the PersonaPlex server."""
        if self._status == ServerStatus.STOPPED:
            return {
                "success": True,
                "message": "Server already stopped",
                "status": self._status.value,
            }
        
        self._status = ServerStatus.STOPPING
        
        try:
            # Cancel health check
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
                self._health_check_task = None
            
            # Stop process
            if self._process:
                logger.info("Stopping PersonaPlex server...")
                
                # Try graceful shutdown first
                self._process.send_signal(signal.SIGTERM)
                
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Graceful shutdown failed, forcing kill")
                    self._process.kill()
                    self._process.wait()
                
                self._process = None
            
            # Cleanup SSL dir
            if self._ssl_dir and Path(self._ssl_dir).exists():
                import shutil
                shutil.rmtree(self._ssl_dir, ignore_errors=True)
                self._ssl_dir = None
            
            self._status = ServerStatus.STOPPED
            self._active_sessions.clear()
            
            logger.info("✅ PersonaPlex server stopped")
            
            return {
                "success": True,
                "message": "Server stopped successfully",
                "status": self._status.value,
            }
            
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
            self._status = ServerStatus.ERROR
            self._error_message = str(e)
            return {
                "success": False,
                "message": str(e),
                "status": self._status.value,
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed server status."""
        uptime = None
        if self._startup_time and self._status == ServerStatus.RUNNING:
            uptime = time.time() - self._startup_time
        
        return {
            "status": self._status.value,
            "is_running": self.is_running,
            "url": self.server_url if self.is_running else None,
            "webui_url": self.webui_url if self.is_running else None,
            "pid": self._process.pid if self._process else None,
            "uptime_seconds": uptime,
            "error": self._error_message,
            "config": self.config.to_dict(),
            "active_sessions": len(self._active_sessions),
        }
    
    def get_voices(self) -> Dict[str, Any]:
        """Get available voice prompts."""
        return {
            "voices": VOICE_PROMPTS,
            "default": self.config.default_voice.value,
            "categories": {
                "natural_female": ["NATF0", "NATF1", "NATF2", "NATF3"],
                "natural_male": ["NATM0", "NATM1", "NATM2", "NATM3"],
                "variety_female": ["VARF0", "VARF1", "VARF2", "VARF3", "VARF4"],
                "variety_male": ["VARM0", "VARM1", "VARM2", "VARM3", "VARM4"],
            },
            "recommended": {
                "female": "NATF2",
                "male": "NATM1",
            }
        }
    
    def get_text_prompts(self) -> Dict[str, Any]:
        """Get available text prompts."""
        return {
            "prompts": DEFAULT_TEXT_PROMPTS,
            "default": self.config.default_prompt,
            "categories": {
                "general": ["assistant", "casual"],
                "customer_service": [
                    "customer_service_bank",
                    "customer_service_medical",
                    "customer_service_restaurant",
                    "customer_service_rental",
                ],
                "roleplay": ["astronaut", "beauty_consultant"],
            }
        }


# Global instance
_personaplex_manager: Optional[PersonaPlexManager] = None


def get_personaplex_manager(config: Optional[PersonaPlexConfig] = None) -> PersonaPlexManager:
    """Get the global PersonaPlex manager instance."""
    global _personaplex_manager
    
    if _personaplex_manager is None:
        _personaplex_manager = PersonaPlexManager(config)
    
    return _personaplex_manager

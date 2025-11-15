#!/usr/bin/env python3
"""
PABX Server
Main entry point for the PABX system
"""

import sys
import signal
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)


def run_api_server(host: str = "0.0.0.0", port: int = 8080):
    """
    Run FastAPI server
    
    Args:
        host: Host address to bind to
        port: Port to listen on
    """
    import uvicorn
    from src.api.server import app
    
    logger.info(f"Starting PABX API server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


def run_sip_only():
    """Run SIP server only (without REST API)"""
    from src.services.call_manager import CallManager
    
    logger.info("Starting PABX SIP server (standalone mode)")
    
    # Create call manager
    call_manager = CallManager()
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        call_manager.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start call manager
    call_manager.start()
    
    logger.info("PABX SIP server running. Press Ctrl+C to stop.")
    
    # Keep running
    signal.pause()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="BeautyAI PABX Server"
    )
    
    parser.add_argument(
        '--mode',
        choices=['api', 'sip'],
        default='api',
        help='Server mode: api (FastAPI + SIP) or sip (SIP only)'
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host address to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='API port (default: 8080)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to config file (default: config/settings.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config(config_file=args.config)
    else:
        config = Config()
    
    # Run in selected mode
    if args.mode == 'api':
        run_api_server(host=args.host, port=args.port)
    else:
        run_sip_only()


if __name__ == '__main__':
    main()

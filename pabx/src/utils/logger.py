"""
Structured logging system for PABX
Supports console logs, JSON logs, session traces, and systemd journal
"""

import os
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from systemd import journal
    SYSTEMD_AVAILABLE = True
except ImportError:
    SYSTEMD_AVAILABLE = False

from .config import get_config


class ColoredFormatter(logging.Formatter):
    """Colored console formatter"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for machine-readable logs"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if available
        if hasattr(record, 'session_id'):
            log_data['session_id'] = record.session_id
        if hasattr(record, 'call_id'):
            log_data['call_id'] = record.call_id
        if hasattr(record, 'device_ip'):
            log_data['device_ip'] = record.device_ip
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class SessionTraceHandler(logging.Handler):
    """Handler for per-session trace logs"""
    
    def __init__(self, base_dir: str):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session_files = {}
    
    def emit(self, record):
        if not hasattr(record, 'session_id'):
            return
        
        session_id = record.session_id
        
        # Create session file if needed
        if session_id not in self.session_files:
            date_str = datetime.now().strftime('%Y%m%d')
            session_dir = self.base_dir / date_str
            session_dir.mkdir(exist_ok=True)
            
            session_file = session_dir / f"{session_id}.log"
            self.session_files[session_id] = open(session_file, 'a')
        
        # Write to session file
        try:
            msg = self.format(record)
            self.session_files[session_id].write(msg + '\n')
            self.session_files[session_id].flush()
        except Exception:
            self.handleError(record)
    
    def close(self):
        """Close all open session files"""
        for f in self.session_files.values():
            f.close()
        super().close()


class SystemdJournalHandler(logging.Handler):
    """Handler for systemd journal"""
    
    PRIORITY_MAP = {
        logging.DEBUG: journal.LOG_DEBUG,
        logging.INFO: journal.LOG_INFO,
        logging.WARNING: journal.LOG_WARNING,
        logging.ERROR: journal.LOG_ERR,
        logging.CRITICAL: journal.LOG_CRIT,
    }
    
    def emit(self, record):
        if not SYSTEMD_AVAILABLE:
            return
        
        try:
            priority = self.PRIORITY_MAP.get(record.levelno, journal.LOG_INFO)
            message = self.format(record)
            
            # Add structured fields
            fields = {
                'SYSLOG_IDENTIFIER': 'pabx',
                'LOGGER': record.name,
                'CODE_MODULE': record.module,
                'CODE_FUNC': record.funcName,
                'CODE_LINE': record.lineno,
            }
            
            if hasattr(record, 'session_id'):
                fields['SESSION_ID'] = record.session_id
            if hasattr(record, 'call_id'):
                fields['CALL_ID'] = record.call_id
            
            journal.send(message, PRIORITY=priority, **fields)
        except Exception:
            self.handleError(record)


def setup_logging():
    """Setup logging system based on configuration"""
    config = get_config()
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.get('logging.level', 'INFO')))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if config.get('logging.console.enabled', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        if config.get('logging.console.colored', True):
            console_formatter = ColoredFormatter(
                config.get('logging.console.format',
                          '%(asctime)s [%(levelname)s] %(name)s: %(message)s')
            )
        else:
            console_formatter = logging.Formatter(
                config.get('logging.console.format',
                          '%(asctime)s [%(levelname)s] %(name)s: %(message)s')
            )
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # JSON handler
    if config.get('logging.json.enabled', True):
        json_file = config.get('logging.json.file')
        if json_file:
            # Create directory if needed
            Path(json_file).parent.mkdir(parents=True, exist_ok=True)
            
            json_handler = logging.handlers.RotatingFileHandler(
                json_file,
                maxBytes=config.get('logging.json.rotate_size', 10485760),
                backupCount=config.get('logging.json.backup_count', 5)
            )
            json_handler.setLevel(logging.DEBUG)
            json_handler.setFormatter(JSONFormatter())
            root_logger.addHandler(json_handler)
    
    # Session trace handler
    if config.get('logging.session.enabled', True):
        session_dir = config.get('logging.session.dir')
        if session_dir:
            session_handler = SessionTraceHandler(session_dir)
            session_handler.setLevel(logging.DEBUG)
            session_formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s: %(message)s'
            )
            session_handler.setFormatter(session_formatter)
            root_logger.addHandler(session_handler)
    
    # Systemd journal handler
    if config.get('logging.journal.enabled', True) and SYSTEMD_AVAILABLE:
        journal_handler = SystemdJournalHandler()
        journal_handler.setLevel(logging.INFO)
        journal_formatter = logging.Formatter('%(message)s')
        journal_handler.setFormatter(journal_formatter)
        root_logger.addHandler(journal_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)


class SessionLogger:
    """Context manager for session-specific logging"""
    
    def __init__(self, session_id: str, logger: Optional[logging.Logger] = None):
        self.session_id = session_id
        self.logger = logger or get_logger('session')
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def _log(self, level: int, msg: str, **kwargs):
        """Internal log method with session context"""
        extra = {'session_id': self.session_id}
        extra.update(kwargs)
        self.logger.log(level, msg, extra=extra)
    
    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        self._log(logging.CRITICAL, msg, **kwargs)

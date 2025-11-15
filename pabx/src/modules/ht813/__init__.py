"""
HT813 device integration module
HTTP API wrapper for Grandstream HT813 ATA
"""

from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from datetime import datetime
import requests
from bs4 import BeautifulSoup

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HT813Status:
    """HT813 device status information"""
    mac_address: str
    firmware_version: str
    uptime: int
    product_model: str
    part_number: str
    
    # Network
    ip_address: str
    subnet_mask: str
    gateway: str
    dns_server: str
    
    # Registration status
    fxs1_registered: bool
    fxs2_registered: bool
    fxs1_status: str
    fxs2_status: str
    
    # Call statistics
    active_calls: int
    total_calls: int


@dataclass
class CallStatistics:
    """Call statistics for FXS port"""
    port_name: str
    total_calls: int
    connected_calls: int
    failed_calls: int
    incoming_calls: int
    outgoing_calls: int


class HT813Device:
    """
    Interface to Grandstream HT813 ATA device
    Provides programmatic access to device status and configuration
    """
    
    def __init__(
        self,
        ip_address: str,
        username: str = "admin",
        password: str = "admin",
        timeout: int = 10
    ):
        """
        Initialize HT813 device interface
        
        Args:
            ip_address: Device IP address
            username: Admin username (default: admin)
            password: Admin password (default: admin)
            timeout: Request timeout in seconds
        """
        self.ip_address = ip_address
        self.username = username
        self.password = password
        self.timeout = timeout
        self.base_url = f"http://{ip_address}"
        
        # Session for cookies
        self.session = requests.Session()
        self.authenticated = False
        
        logger.info(f"Initialized HT813 device interface at {ip_address}")
    
    def authenticate(self) -> bool:
        """
        Authenticate with the device
        
        Returns:
            True if authentication successful
        """
        try:
            url = f"{self.base_url}/cgi-bin/dologin"
            
            data = {
                'username': self.username,
                'password': self.password,
            }
            
            response = self.session.post(
                url,
                data=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                self.authenticated = True
                logger.info(f"Authenticated with HT813 at {self.ip_address}")
                return True
            else:
                logger.error(f"Authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error authenticating with HT813: {e}", exc_info=True)
            return False
    
    def get_status(self) -> Optional[HT813Status]:
        """
        Get device status information
        
        Returns:
            HT813Status object or None if failed
        """
        if not self.authenticated:
            if not self.authenticate():
                return None
        
        try:
            url = f"{self.base_url}/cgi-bin/api-sys_operation"
            params = {'request': 'STATUS'}
            
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get status: {response.status_code}")
                return None
            
            # Parse HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract status fields
            status = self._parse_status_page(soup)
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting HT813 status: {e}", exc_info=True)
            return None
    
    def _parse_status_page(self, soup: BeautifulSoup) -> Optional[HT813Status]:
        """
        Parse status page HTML
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            HT813Status object
        """
        try:
            # Extract text content
            text = soup.get_text()
            
            # Simple field extraction
            # Note: Actual parsing depends on HT813 HTML structure
            # This is a template that needs adjustment based on actual device
            
            status = HT813Status(
                mac_address=self._extract_field(text, "MAC Address"),
                firmware_version=self._extract_field(text, "Firmware Version"),
                uptime=self._parse_uptime(self._extract_field(text, "System Up Time")),
                product_model=self._extract_field(text, "Product Model"),
                part_number=self._extract_field(text, "Part Number"),
                
                ip_address=self._extract_field(text, "IP Address"),
                subnet_mask=self._extract_field(text, "Subnet Mask"),
                gateway=self._extract_field(text, "Gateway"),
                dns_server=self._extract_field(text, "DNS Server"),
                
                fxs1_registered=self._check_registration(text, "FXS1"),
                fxs2_registered=self._check_registration(text, "FXS2"),
                fxs1_status=self._extract_field(text, "FXS1 Status"),
                fxs2_status=self._extract_field(text, "FXS2 Status"),
                
                active_calls=self._parse_int(self._extract_field(text, "Active Calls")),
                total_calls=self._parse_int(self._extract_field(text, "Total Calls"))
            )
            
            return status
            
        except Exception as e:
            logger.error(f"Error parsing status page: {e}", exc_info=True)
            return None
    
    def get_call_statistics(self) -> Optional[List[CallStatistics]]:
        """
        Get call statistics for all ports
        
        Returns:
            List of CallStatistics objects
        """
        if not self.authenticated:
            if not self.authenticate():
                return None
        
        try:
            url = f"{self.base_url}/cgi-bin/api-sys_operation"
            params = {'request': 'CALLSTATS'}
            
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get call stats: {response.status_code}")
                return None
            
            # Parse response
            soup = BeautifulSoup(response.text, 'html.parser')
            stats = self._parse_call_stats(soup)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting call statistics: {e}", exc_info=True)
            return None
    
    def _parse_call_stats(self, soup: BeautifulSoup) -> List[CallStatistics]:
        """
        Parse call statistics page
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of CallStatistics objects
        """
        stats = []
        
        try:
            text = soup.get_text()
            
            # Parse FXS1 stats
            fxs1_stats = CallStatistics(
                port_name="FXS1",
                total_calls=self._parse_int(self._extract_field(text, "FXS1 Total Calls")),
                connected_calls=self._parse_int(self._extract_field(text, "FXS1 Connected")),
                failed_calls=self._parse_int(self._extract_field(text, "FXS1 Failed")),
                incoming_calls=self._parse_int(self._extract_field(text, "FXS1 Incoming")),
                outgoing_calls=self._parse_int(self._extract_field(text, "FXS1 Outgoing"))
            )
            stats.append(fxs1_stats)
            
            # Parse FXS2 stats
            fxs2_stats = CallStatistics(
                port_name="FXS2",
                total_calls=self._parse_int(self._extract_field(text, "FXS2 Total Calls")),
                connected_calls=self._parse_int(self._extract_field(text, "FXS2 Connected")),
                failed_calls=self._parse_int(self._extract_field(text, "FXS2 Failed")),
                incoming_calls=self._parse_int(self._extract_field(text, "FXS2 Incoming")),
                outgoing_calls=self._parse_int(self._extract_field(text, "FXS2 Outgoing"))
            )
            stats.append(fxs2_stats)
            
        except Exception as e:
            logger.error(f"Error parsing call stats: {e}", exc_info=True)
        
        return stats
    
    def reboot(self) -> bool:
        """
        Reboot the device
        
        Returns:
            True if reboot command sent successfully
        """
        if not self.authenticated:
            if not self.authenticate():
                return False
        
        try:
            url = f"{self.base_url}/cgi-bin/api-sys_operation"
            data = {'action': 'reboot'}
            
            response = self.session.post(
                url,
                data=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                logger.info(f"Sent reboot command to HT813 at {self.ip_address}")
                return True
            else:
                logger.error(f"Failed to reboot: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error rebooting HT813: {e}", exc_info=True)
            return False
    
    def reset_call_statistics(self) -> bool:
        """
        Reset call statistics counters
        
        Returns:
            True if reset successful
        """
        if not self.authenticated:
            if not self.authenticate():
                return False
        
        try:
            url = f"{self.base_url}/cgi-bin/api-sys_operation"
            data = {'action': 'reset_call_stats'}
            
            response = self.session.post(
                url,
                data=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                logger.info("Reset call statistics")
                return True
            else:
                logger.error(f"Failed to reset stats: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error resetting call stats: {e}", exc_info=True)
            return False
    
    # Helper methods
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract field value from text"""
        # Simple extraction - adjust based on actual HTML structure
        try:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if field_name in line and i + 1 < len(lines):
                    return lines[i + 1].strip()
        except:
            pass
        return ""
    
    def _parse_int(self, value: str) -> int:
        """Parse integer from string"""
        try:
            return int(value)
        except:
            return 0
    
    def _parse_uptime(self, uptime_str: str) -> int:
        """Parse uptime string to seconds"""
        try:
            # Parse "X days Y hours Z minutes" format
            seconds = 0
            if 'day' in uptime_str:
                days = int(uptime_str.split('day')[0].strip())
                seconds += days * 86400
            if 'hour' in uptime_str:
                hours = int(uptime_str.split('hour')[0].split()[-1])
                seconds += hours * 3600
            if 'minute' in uptime_str:
                minutes = int(uptime_str.split('minute')[0].split()[-1])
                seconds += minutes * 60
            return seconds
        except:
            return 0
    
    def _check_registration(self, text: str, port: str) -> bool:
        """Check if port is registered"""
        field = self._extract_field(text, f"{port} Status")
        return "Registered" in field or "Active" in field

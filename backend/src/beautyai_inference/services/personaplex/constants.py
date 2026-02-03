"""
PersonaPlex Constants and Configuration

Voice prompts and default text prompts for PersonaPlex model.
Based on NVIDIA PersonaPlex documentation:
https://github.com/NVIDIA/personaplex
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path


class VoiceType(str, Enum):
    """Available voice types in PersonaPlex."""
    # Natural female voices
    NATF0 = "NATF0"
    NATF1 = "NATF1"
    NATF2 = "NATF2"
    NATF3 = "NATF3"
    
    # Natural male voices
    NATM0 = "NATM0"
    NATM1 = "NATM1"
    NATM2 = "NATM2"
    NATM3 = "NATM3"
    
    # Variety female voices
    VARF0 = "VARF0"
    VARF1 = "VARF1"
    VARF2 = "VARF2"
    VARF3 = "VARF3"
    VARF4 = "VARF4"
    
    # Variety male voices
    VARM0 = "VARM0"
    VARM1 = "VARM1"
    VARM2 = "VARM2"
    VARM3 = "VARM3"
    VARM4 = "VARM4"


# Voice prompt descriptions for UI
VOICE_PROMPTS: Dict[str, Dict[str, str]] = {
    # Natural voices (more conversational)
    "NATF0": {"name": "Natural Female 0", "category": "Natural", "gender": "female", "description": "Natural conversational female voice"},
    "NATF1": {"name": "Natural Female 1", "category": "Natural", "gender": "female", "description": "Natural conversational female voice"},
    "NATF2": {"name": "Natural Female 2", "category": "Natural", "gender": "female", "description": "Natural conversational female voice (recommended)"},
    "NATF3": {"name": "Natural Female 3", "category": "Natural", "gender": "female", "description": "Natural conversational female voice"},
    
    "NATM0": {"name": "Natural Male 0", "category": "Natural", "gender": "male", "description": "Natural conversational male voice"},
    "NATM1": {"name": "Natural Male 1", "category": "Natural", "gender": "male", "description": "Natural conversational male voice (recommended)"},
    "NATM2": {"name": "Natural Male 2", "category": "Natural", "gender": "male", "description": "Natural conversational male voice"},
    "NATM3": {"name": "Natural Male 3", "category": "Natural", "gender": "male", "description": "Natural conversational male voice"},
    
    # Variety voices (more expressive/unique)
    "VARF0": {"name": "Variety Female 0", "category": "Variety", "gender": "female", "description": "Varied expressive female voice"},
    "VARF1": {"name": "Variety Female 1", "category": "Variety", "gender": "female", "description": "Varied expressive female voice"},
    "VARF2": {"name": "Variety Female 2", "category": "Variety", "gender": "female", "description": "Varied expressive female voice"},
    "VARF3": {"name": "Variety Female 3", "category": "Variety", "gender": "female", "description": "Varied expressive female voice"},
    "VARF4": {"name": "Variety Female 4", "category": "Variety", "gender": "female", "description": "Varied expressive female voice"},
    
    "VARM0": {"name": "Variety Male 0", "category": "Variety", "gender": "male", "description": "Varied expressive male voice"},
    "VARM1": {"name": "Variety Male 1", "category": "Variety", "gender": "male", "description": "Varied expressive male voice"},
    "VARM2": {"name": "Variety Male 2", "category": "Variety", "gender": "male", "description": "Varied expressive male voice"},
    "VARM3": {"name": "Variety Male 3", "category": "Variety", "gender": "male", "description": "Varied expressive male voice"},
    "VARM4": {"name": "Variety Male 4", "category": "Variety", "gender": "male", "description": "Varied expressive male voice"},
}


# Default text prompts for different scenarios
DEFAULT_TEXT_PROMPTS: Dict[str, str] = {
    "assistant": "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
    
    "casual": "You enjoy having a good conversation.",
    
    "customer_service_bank": """You work for First Neuron Bank which is a bank and your name is Sanni Virtanen. 
Information: The customer's transaction for $1,200 at Home Depot was declined. 
Verify customer identity. The transaction was flagged due to an unusual location 
(transaction attempted in Miami, FL; customer normally transacts in Seattle, WA).""",
    
    "customer_service_medical": """You work for Dr. Jones's medical office, and you are receiving calls to record information for new patients. 
Information: Record full name, date of birth, any medication allergies, tobacco smoking history, 
alcohol consumption history, and any prior medical conditions. 
Assure the patient that this information will be confidential, if they ask.""",
    
    "customer_service_restaurant": """You work for Jerusalem Shakshuka which is a restaurant and your name is Owen Foster. 
Information: There are two shakshuka options: Classic (poached eggs, $9.50) and Spicy (scrambled eggs with jalapenos, $10.25). 
Sides include warm pita ($2.50) and Israeli salad ($3). No combo offers. Available for drive-through until 9 PM.""",
    
    "customer_service_rental": """You work for AeroRentals Pro which is a drone rental company and your name is Tomaz Novak. 
Information: AeroRentals Pro has the following availability: PhoenixDrone X ($65/4 hours, $110/8 hours), 
and the premium SpectraDrone 9 ($95/4 hours, $160/8 hours). Deposit required: $150 for standard models, $300 for premium.""",
    
    "astronaut": """You enjoy having a good conversation. Have a technical discussion about fixing a reactor core on a spaceship to Mars. 
You are an astronaut on a Mars mission. Your name is Alex. You are already dealing with a reactor core meltdown on a Mars mission. 
Several ship systems are failing, and continued instability will lead to catastrophic failure. 
You explain what is happening and you urgently ask for help thinking through how to stabilize the reactor.""",
    
    "beauty_consultant": """You are Layla, a friendly and knowledgeable beauty consultant at a high-end skincare clinic in Riyadh. 
You speak warmly and professionally. Help customers with skincare advice, product recommendations, 
and appointment scheduling. You're fluent in Arabic and English.""",
}


@dataclass
class PersonaPlexConfig:
    """Configuration for PersonaPlex server."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8998
    
    # Model settings
    cpu_offload: bool = True  # Required for GPUs with <14GB VRAM
    hf_token: Optional[str] = None
    
    # SSL settings (required for WebRTC)
    ssl_enabled: bool = True
    ssl_dir: Optional[str] = None  # Auto-generated if None
    
    # Default voice and prompt
    default_voice: VoiceType = VoiceType.NATF2
    default_prompt: str = "assistant"
    
    # Paths
    personaplex_path: str = "/home/lumi/personaplex"
    python_executable: str = "python"
    
    # Process management
    startup_timeout: int = 120  # seconds
    health_check_interval: int = 5  # seconds
    
    # Resource limits
    max_concurrent_sessions: int = 1  # Full-duplex is resource-intensive
    
    def get_text_prompt(self) -> str:
        """Get the text prompt for the default prompt key."""
        return DEFAULT_TEXT_PROMPTS.get(self.default_prompt, DEFAULT_TEXT_PROMPTS["assistant"])
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "cpu_offload": self.cpu_offload,
            "ssl_enabled": self.ssl_enabled,
            "default_voice": self.default_voice.value,
            "default_prompt": self.default_prompt,
            "personaplex_path": self.personaplex_path,
            "startup_timeout": self.startup_timeout,
            "max_concurrent_sessions": self.max_concurrent_sessions,
        }

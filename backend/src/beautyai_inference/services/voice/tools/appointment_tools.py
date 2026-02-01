"""
Voice Tools for Customer Service Demo.

Provides callable functions for the LLM during voice conversations:
- check_customer: Look up customer by name
- register_customer: Register new customer
- list_available_slots: Get available appointment times
- book_appointment: Book an appointment
- cancel_appointment: Cancel an existing appointment
- get_customer_appointments: Get customer's appointments

These tools integrate with the VoicePipelineOrchestrator for execution.
"""

import logging
import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories for voice tools."""
    CUSTOMER = "customer"
    APPOINTMENT = "appointment"
    QUERY = "query"


@dataclass
class VoiceTool:
    """Definition of a voice tool callable by the LLM."""
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Any]
    required_params: List[str] = field(default_factory=list)
    allows_interruption: bool = True  # Can user interrupt during execution?
    
    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_params
                }
            }
        }


# ============================================
# Tool Definitions
# ============================================

VOICE_TOOLS: Dict[str, VoiceTool] = {
    "check_customer": VoiceTool(
        name="check_customer",
        description="Check if a customer exists in the system by their first and last name. Use this to verify if a customer is already registered before booking.",
        category=ToolCategory.CUSTOMER,
        parameters={
            "first_name": {
                "type": "string",
                "description": "Customer's first name"
            },
            "last_name": {
                "type": "string",
                "description": "Customer's last name"
            },
            "phone": {
                "type": "string",
                "description": "Customer's phone number (optional)"
            }
        },
        required_params=["first_name", "last_name"],
        allows_interruption=True  # Quick query, can be interrupted
    ),
    
    "register_customer": VoiceTool(
        name="register_customer",
        description="Register a new customer in the system. Use this after confirming the customer is not already registered.",
        category=ToolCategory.CUSTOMER,
        parameters={
            "first_name": {
                "type": "string",
                "description": "Customer's first name"
            },
            "last_name": {
                "type": "string",
                "description": "Customer's last name"
            },
            "phone": {
                "type": "string",
                "description": "Customer's phone number (optional)"
            },
            "email": {
                "type": "string",
                "description": "Customer's email address (optional)"
            },
            "preferred_language": {
                "type": "string",
                "enum": ["ar", "en"],
                "description": "Customer's preferred language (ar=Arabic, en=English)"
            }
        },
        required_params=["first_name", "last_name"],
        allows_interruption=False  # Writing to DB, don't interrupt
    ),
    
    "list_available_slots": VoiceTool(
        name="list_available_slots",
        description="Get a list of available appointment time slots. Can filter by specific date or show slots for the next few days.",
        category=ToolCategory.QUERY,
        parameters={
            "date": {
                "type": "string",
                "description": "Specific date to check (YYYY-MM-DD format). If not provided, shows slots for the next 7 days."
            },
            "days_ahead": {
                "type": "integer",
                "description": "Number of days to look ahead (1-30, default 7)"
            }
        },
        required_params=[],
        allows_interruption=True  # Read-only query
    ),
    
    "book_appointment": VoiceTool(
        name="book_appointment",
        description="Book an appointment for a customer at a specific time slot. Requires customer ID and time slot ID from previous queries.",
        category=ToolCategory.APPOINTMENT,
        parameters={
            "customer_id": {
                "type": "integer",
                "description": "The customer's ID (from check_customer or register_customer)"
            },
            "time_slot_id": {
                "type": "integer",
                "description": "The time slot ID (from list_available_slots)"
            },
            "service_type": {
                "type": "string",
                "description": "Type of service (e.g., 'consultation', 'treatment', 'checkup')"
            },
            "notes": {
                "type": "string",
                "description": "Additional notes about the appointment"
            }
        },
        required_params=["customer_id", "time_slot_id"],
        allows_interruption=False  # Writing to DB, don't interrupt
    ),
    
    "cancel_appointment": VoiceTool(
        name="cancel_appointment",
        description="Cancel an existing appointment. The time slot will become available again.",
        category=ToolCategory.APPOINTMENT,
        parameters={
            "appointment_id": {
                "type": "integer",
                "description": "The appointment ID to cancel"
            }
        },
        required_params=["appointment_id"],
        allows_interruption=False  # Writing to DB, don't interrupt
    ),
    
    "get_customer_appointments": VoiceTool(
        name="get_customer_appointments",
        description="Get all appointments for a specific customer.",
        category=ToolCategory.QUERY,
        parameters={
            "customer_id": {
                "type": "integer",
                "description": "The customer's ID"
            },
            "include_cancelled": {
                "type": "boolean",
                "description": "Include cancelled appointments (default: false)"
            }
        },
        required_params=["customer_id"],
        allows_interruption=True  # Read-only query
    )
}


def get_tools_for_openai() -> List[Dict[str, Any]]:
    """Get all voice tools in OpenAI function calling format."""
    return [tool.to_openai_function() for tool in VOICE_TOOLS.values()]


def get_tool(name: str) -> Optional[VoiceTool]:
    """Get a specific tool by name."""
    return VOICE_TOOLS.get(name)


def tool_allows_interruption(name: str) -> bool:
    """Check if a tool allows user interruption during execution."""
    tool = VOICE_TOOLS.get(name)
    return tool.allows_interruption if tool else True


# ============================================
# Tool Executor
# ============================================

class VoiceToolExecutor:
    """
    Executes voice tools by calling the internal API endpoints.
    
    Used by the VoicePipelineOrchestrator to fulfill tool calls.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_prefix = "/api/v1/demo/appointments"
    
    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a voice tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            session_id: Voice session ID for tracking
            
        Returns:
            Tool execution result
        """
        logger.info(f"Executing tool: {tool_name} with params: {parameters}")
        
        tool = VOICE_TOOLS.get(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if tool_name == "check_customer":
                    return await self._check_customer(client, parameters)
                elif tool_name == "register_customer":
                    return await self._register_customer(client, parameters)
                elif tool_name == "list_available_slots":
                    return await self._list_available_slots(client, parameters)
                elif tool_name == "book_appointment":
                    return await self._book_appointment(client, parameters, session_id)
                elif tool_name == "cancel_appointment":
                    return await self._cancel_appointment(client, parameters)
                elif tool_name == "get_customer_appointments":
                    return await self._get_customer_appointments(client, parameters)
                else:
                    return {"success": False, "error": f"Tool not implemented: {tool_name}"}
                    
        except httpx.RequestError as e:
            logger.error(f"HTTP error executing tool {tool_name}: {e}")
            return {"success": False, "error": f"Network error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _check_customer(
        self,
        client: httpx.AsyncClient,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute check_customer tool."""
        response = await client.post(
            f"{self.base_url}{self.api_prefix}/check-customer",
            json={
                "first_name": params["first_name"],
                "last_name": params["last_name"],
                "phone": params.get("phone")
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "success": True,
            "found": data["found"],
            "customer": data.get("customer"),
            "message": data["message"]
        }
    
    async def _register_customer(
        self,
        client: httpx.AsyncClient,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute register_customer tool."""
        response = await client.post(
            f"{self.base_url}{self.api_prefix}/register-customer",
            json={
                "first_name": params["first_name"],
                "last_name": params["last_name"],
                "phone": params.get("phone"),
                "email": params.get("email"),
                "preferred_language": params.get("preferred_language", "ar")
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "success": data["success"],
            "customer": data["customer"],
            "message": data["message"]
        }
    
    async def _list_available_slots(
        self,
        client: httpx.AsyncClient,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute list_available_slots tool."""
        query_params = {}
        if params.get("date"):
            query_params["date"] = params["date"]
        if params.get("days_ahead"):
            query_params["days_ahead"] = params["days_ahead"]
        
        response = await client.get(
            f"{self.base_url}{self.api_prefix}/slots",
            params=query_params
        )
        response.raise_for_status()
        data = response.json()
        
        # Format slots for natural language response
        slots = data.get("available_slots", [])
        if not slots:
            return {
                "success": True,
                "available_slots": [],
                "message": "No available slots found for the requested period.",
                "formatted": "عذراً، لا توجد مواعيد متاحة في الفترة المطلوبة."
            }
        
        # Group by date for readable output
        slots_by_date = {}
        for slot in slots:
            date = slot["date"]
            if date not in slots_by_date:
                slots_by_date[date] = []
            slots_by_date[date].append(slot)
        
        formatted_lines = []
        for date, date_slots in slots_by_date.items():
            times = [s["start_time"] for s in date_slots]
            formatted_lines.append(f"{date}: {', '.join(times)}")
        
        return {
            "success": True,
            "available_slots": slots,
            "total_count": len(slots),
            "slots_by_date": slots_by_date,
            "formatted": "\n".join(formatted_lines)
        }
    
    async def _book_appointment(
        self,
        client: httpx.AsyncClient,
        params: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute book_appointment tool."""
        response = await client.post(
            f"{self.base_url}{self.api_prefix}/book",
            json={
                "customer_id": params["customer_id"],
                "time_slot_id": params["time_slot_id"],
                "service_type": params.get("service_type", "consultation"),
                "notes": params.get("notes"),
                "voice_session_id": session_id
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "success": data["success"],
            "appointment": data.get("appointment"),
            "message": data["message"]
        }
    
    async def _cancel_appointment(
        self,
        client: httpx.AsyncClient,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute cancel_appointment tool."""
        response = await client.patch(
            f"{self.base_url}{self.api_prefix}/appointments/{params['appointment_id']}/cancel"
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "success": data["success"],
            "appointment": data.get("appointment"),
            "message": data["message"]
        }
    
    async def _get_customer_appointments(
        self,
        client: httpx.AsyncClient,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute get_customer_appointments tool."""
        query_params = {}
        if params.get("include_cancelled"):
            query_params["include_cancelled"] = "true"
        
        response = await client.get(
            f"{self.base_url}{self.api_prefix}/customer/{params['customer_id']}/appointments",
            params=query_params
        )
        response.raise_for_status()
        data = response.json()
        
        appointments = data.get("appointments", [])
        if not appointments:
            return {
                "success": True,
                "customer": data.get("customer"),
                "appointments": [],
                "message": "No appointments found for this customer."
            }
        
        # Format appointments for response
        formatted_lines = []
        for appt in appointments:
            slot = appt.get("time_slot", {})
            formatted_lines.append(
                f"- {slot.get('date', 'N/A')} at {slot.get('start_time', 'N/A')} "
                f"({appt.get('service_type', 'N/A')}) - Status: {appt.get('status', 'N/A')}"
            )
        
        return {
            "success": True,
            "customer": data.get("customer"),
            "appointments": appointments,
            "total_count": len(appointments),
            "formatted": "\n".join(formatted_lines)
        }


# ============================================
# System Prompt for Customer Service Agent
# ============================================

CUSTOMER_SERVICE_SYSTEM_PROMPT = """أنت مساعد صوتي لخدمة العملاء في عيادة كيسي للتجميل. مهمتك هي مساعدة العملاء في:

1. **التحقق من العميل**: عندما يتصل عميل، اسأله عن اسمه الكامل واستخدم أداة check_customer للتحقق من وجوده.

2. **تسجيل العملاء الجدد**: إذا لم يكن العميل مسجلاً، اطلب منه معلوماته واستخدم أداة register_customer لتسجيله.

3. **حجز المواعيد**: 
   - استخدم list_available_slots لعرض الأوقات المتاحة
   - استخدم book_appointment لتأكيد الحجز

4. **إلغاء المواعيد**: استخدم cancel_appointment إذا أراد العميل إلغاء موعده.

5. **الاستفسار عن المواعيد**: استخدم get_customer_appointments لمعرفة مواعيد العميل.

### قواعد مهمة:
- تحدث بالعربية دائماً إلا إذا تحدث العميل بالإنجليزية
- كن ودوداً ومهذباً
- لا تتوقف عن الاستماع أثناء تنفيذ العمليات المهمة مثل الحجز أو التسجيل
- أكد دائماً المعلومات مع العميل قبل تنفيذ أي عملية

---

You are a voice assistant for Kesay Beauty Clinic customer service. Your job is to help customers with:

1. **Customer Verification**: When a customer calls, ask for their full name and use check_customer tool to verify.

2. **New Customer Registration**: If customer is not registered, ask for their information and use register_customer.

3. **Booking Appointments**: 
   - Use list_available_slots to show available times
   - Use book_appointment to confirm the booking

4. **Cancelling Appointments**: Use cancel_appointment if the customer wants to cancel.

5. **Checking Appointments**: Use get_customer_appointments to show customer's appointments.

### Important Rules:
- Speak in Arabic unless the customer speaks English
- Be friendly and polite
- Don't stop listening during important operations like booking or registration
- Always confirm information with the customer before executing any operation
"""


def get_customer_service_system_prompt() -> str:
    """Get the system prompt for customer service voice demo."""
    return CUSTOMER_SERVICE_SYSTEM_PROMPT

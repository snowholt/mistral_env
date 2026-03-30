"""
Demo Appointments API endpoints.

Provides endpoints for the Customer Service Voice Demo:
- Customer lookup and registration
- Time slot availability
- Appointment booking and management

These endpoints are called by voice tools during conversations.
"""

import logging
from typing import Optional, List
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload

from ...database.connection import get_db
from ...database.models import (
    DemoCustomer, DemoTimeSlot, DemoAppointment, AppointmentStatus
)

logger = logging.getLogger(__name__)

demo_appointments_router = APIRouter(prefix="/api/v1/demo/appointments", tags=["demo_appointments"])


# ============================================
# Request/Response Models
# ============================================


class CustomerLookupRequest(BaseModel):
    """Request for customer lookup."""
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    phone: Optional[str] = Field(None, max_length=50)


class CustomerLookupResponse(BaseModel):
    """Response for customer lookup."""
    found: bool
    customer: Optional[dict] = None
    message: str


class CustomerRegisterRequest(BaseModel):
    """Request for customer registration."""
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    phone: Optional[str] = Field(None, max_length=50)
    email: Optional[str] = Field(None, max_length=255)
    preferred_language: str = Field(default="ar", pattern="^(ar|en)$")


class CustomerResponse(BaseModel):
    """Response for customer operations."""
    success: bool
    customer: dict
    message: str


class TimeSlotResponse(BaseModel):
    """Response for time slot."""
    id: int
    date: str
    start_time: str
    end_time: str
    duration_minutes: int
    is_available: bool
    slots_remaining: int


class AvailableSlotsResponse(BaseModel):
    """Response for available slots."""
    available_slots: List[TimeSlotResponse]
    total_count: int
    date_range: dict


class BookAppointmentRequest(BaseModel):
    """Request for booking an appointment."""
    customer_id: int
    time_slot_id: int
    service_type: str = Field(default="consultation", max_length=100)
    notes: Optional[str] = None
    voice_session_id: Optional[str] = None


class AppointmentResponse(BaseModel):
    """Response for appointment operations."""
    success: bool
    appointment: Optional[dict] = None
    message: str


class CustomerAppointmentsResponse(BaseModel):
    """Response for customer appointments."""
    customer: dict
    appointments: List[dict]
    total_count: int


# ============================================
# Customer Endpoints
# ============================================


@demo_appointments_router.post("/check-customer", response_model=CustomerLookupResponse)
async def check_customer(
    request: CustomerLookupRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Check if a customer exists by name and optionally phone.
    Called by voice tool: check_customer
    """
    logger.info(f"Checking customer: {request.first_name} {request.last_name}")
    
    # Build query - case-insensitive name match
    query = select(DemoCustomer).where(
        and_(
            func.lower(DemoCustomer.first_name) == request.first_name.lower(),
            func.lower(DemoCustomer.last_name) == request.last_name.lower()
        )
    )
    
    # If phone provided, also match phone
    if request.phone:
        query = query.where(
            or_(
                DemoCustomer.phone == request.phone,
                DemoCustomer.phone.is_(None)
            )
        )
    
    result = await db.execute(query)
    customer = result.scalar_one_or_none()
    
    if customer:
        logger.info(f"Found customer: {customer.id}")
        return CustomerLookupResponse(
            found=True,
            customer=customer.to_dict(),
            message=f"Welcome back, {customer.first_name}!"
        )
    else:
        logger.info("Customer not found")
        return CustomerLookupResponse(
            found=False,
            customer=None,
            message=f"No customer found with name {request.first_name} {request.last_name}"
        )


@demo_appointments_router.post("/register-customer", response_model=CustomerResponse)
async def register_customer(
    request: CustomerRegisterRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new customer.
    Called by voice tool: register_customer
    """
    logger.info(f"Registering customer: {request.first_name} {request.last_name}")
    
    # Check if customer already exists
    existing_query = select(DemoCustomer).where(
        and_(
            func.lower(DemoCustomer.first_name) == request.first_name.lower(),
            func.lower(DemoCustomer.last_name) == request.last_name.lower()
        )
    )
    result = await db.execute(existing_query)
    existing = result.scalar_one_or_none()
    
    if existing:
        logger.info(f"Customer already exists: {existing.id}")
        return CustomerResponse(
            success=True,
            customer=existing.to_dict(),
            message=f"Customer {existing.full_name()} already registered"
        )
    
    # Create new customer
    new_customer = DemoCustomer(
        first_name=request.first_name,
        last_name=request.last_name,
        phone=request.phone,
        email=request.email,
        preferred_language=request.preferred_language
    )
    
    db.add(new_customer)
    await db.commit()
    await db.refresh(new_customer)
    
    logger.info(f"Created customer: {new_customer.id}")
    return CustomerResponse(
        success=True,
        customer=new_customer.to_dict(),
        message=f"Successfully registered {new_customer.full_name()}"
    )


@demo_appointments_router.get("/customer/{customer_id}", response_model=CustomerResponse)
async def get_customer(
    customer_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get customer by ID."""
    result = await db.execute(
        select(DemoCustomer).where(DemoCustomer.id == customer_id)
    )
    customer = result.scalar_one_or_none()
    
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    return CustomerResponse(
        success=True,
        customer=customer.to_dict(),
        message="Customer found"
    )


# ============================================
# Time Slot Endpoints
# ============================================


@demo_appointments_router.get("/slots", response_model=AvailableSlotsResponse)
async def list_available_slots(
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    days_ahead: int = Query(7, ge=1, le=30, description="Number of days to look ahead"),
    db: AsyncSession = Depends(get_db)
):
    """
    List available time slots.
    Called by voice tool: list_available_slots
    """
    logger.info(f"Listing available slots, date={date}, days_ahead={days_ahead}")
    
    now = datetime.now()
    start_date = now
    end_date = now + timedelta(days=days_ahead)
    
    # If specific date requested
    if date:
        try:
            specific_date = datetime.strptime(date, "%Y-%m-%d")
            start_date = specific_date
            end_date = specific_date + timedelta(days=1)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD"
            )
    
    # Query available slots
    query = select(DemoTimeSlot).where(
        and_(
            DemoTimeSlot.date >= start_date,
            DemoTimeSlot.date < end_date,
            DemoTimeSlot.is_available == True,
            DemoTimeSlot.current_bookings < DemoTimeSlot.max_bookings
        )
    ).order_by(DemoTimeSlot.date, DemoTimeSlot.start_time)
    
    result = await db.execute(query)
    slots = result.scalars().all()
    
    # Filter out past slots for today
    available_slots = []
    for slot in slots:
        if slot.is_bookable():
            available_slots.append(TimeSlotResponse(**slot.to_dict()))
    
    logger.info(f"Found {len(available_slots)} available slots")
    return AvailableSlotsResponse(
        available_slots=available_slots,
        total_count=len(available_slots),
        date_range={
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d")
        }
    )


@demo_appointments_router.post("/slots/generate")
async def generate_time_slots(
    days: int = Query(7, ge=1, le=30, description="Number of days to generate slots for"),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate time slots for the next N days.
    Admin utility endpoint.
    """
    logger.info(f"Generating time slots for {days} days")
    
    created_count = 0
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Generate slots for each day
    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        
        # Skip weekends (Friday and Saturday in Saudi Arabia)
        if current_date.weekday() in [4, 5]:  # Friday=4, Saturday=5
            continue
        
        # Generate slots from 9:00 to 17:00 (30-minute intervals)
        for hour in range(9, 17):
            for minute in [0, 30]:
                start_time = f"{hour:02d}:{minute:02d}"
                end_hour = hour if minute == 30 else hour
                end_minute = 0 if minute == 30 else 30
                if minute == 30:
                    end_hour += 1
                end_time = f"{end_hour:02d}:{end_minute:02d}"
                
                # Check if slot already exists
                existing = await db.execute(
                    select(DemoTimeSlot).where(
                        and_(
                            DemoTimeSlot.date == current_date,
                            DemoTimeSlot.start_time == start_time
                        )
                    )
                )
                
                if not existing.scalar_one_or_none():
                    new_slot = DemoTimeSlot(
                        date=current_date,
                        start_time=start_time,
                        end_time=end_time,
                        duration_minutes=30,
                        max_bookings=1,
                        is_available=True
                    )
                    db.add(new_slot)
                    created_count += 1
    
    await db.commit()
    logger.info(f"Created {created_count} time slots")
    
    return {
        "success": True,
        "message": f"Generated {created_count} time slots",
        "days_generated": days
    }


# ============================================
# Appointment Endpoints
# ============================================


@demo_appointments_router.post("/book", response_model=AppointmentResponse)
async def book_appointment(
    request: BookAppointmentRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Book an appointment for a customer.
    Called by voice tool: book_appointment
    """
    logger.info(f"Booking appointment: customer={request.customer_id}, slot={request.time_slot_id}")
    
    # Verify customer exists
    customer_result = await db.execute(
        select(DemoCustomer).where(DemoCustomer.id == request.customer_id)
    )
    customer = customer_result.scalar_one_or_none()
    
    if not customer:
        logger.warning(f"Customer not found: {request.customer_id}")
        return AppointmentResponse(
            success=False,
            appointment=None,
            message="Customer not found. Please register first."
        )
    
    # Verify time slot exists and is available
    slot_result = await db.execute(
        select(DemoTimeSlot).where(DemoTimeSlot.id == request.time_slot_id)
    )
    time_slot = slot_result.scalar_one_or_none()
    
    if not time_slot:
        logger.warning(f"Time slot not found: {request.time_slot_id}")
        return AppointmentResponse(
            success=False,
            appointment=None,
            message="Time slot not found."
        )
    
    if not time_slot.is_bookable():
        logger.warning(f"Time slot not available: {request.time_slot_id}")
        return AppointmentResponse(
            success=False,
            appointment=None,
            message="Sorry, this time slot is no longer available. Please choose another."
        )
    
    # Check if customer already has an appointment at this slot
    existing_appt = await db.execute(
        select(DemoAppointment).where(
            and_(
                DemoAppointment.customer_id == request.customer_id,
                DemoAppointment.time_slot_id == request.time_slot_id,
                DemoAppointment.status != AppointmentStatus.CANCELLED
            )
        )
    )
    
    if existing_appt.scalar_one_or_none():
        return AppointmentResponse(
            success=False,
            appointment=None,
            message="You already have an appointment at this time."
        )
    
    # Book the slot
    if not time_slot.book():
        return AppointmentResponse(
            success=False,
            appointment=None,
            message="Failed to book the slot. Please try again."
        )
    
    # Create appointment
    appointment = DemoAppointment(
        customer_id=request.customer_id,
        time_slot_id=request.time_slot_id,
        service_type=request.service_type,
        notes=request.notes,
        voice_session_id=request.voice_session_id,
        status=AppointmentStatus.CONFIRMED
    )
    appointment.confirmed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    
    db.add(appointment)
    await db.commit()
    await db.refresh(appointment)
    
    # Load relationships for response
    await db.refresh(appointment, ["customer", "time_slot"])
    
    logger.info(f"Created appointment: {appointment.id}")
    return AppointmentResponse(
        success=True,
        appointment=appointment.to_dict(),
        message=f"Appointment confirmed for {time_slot.date.strftime('%Y-%m-%d')} at {time_slot.start_time}"
    )


@demo_appointments_router.get("/customer/{customer_id}/appointments", response_model=CustomerAppointmentsResponse)
async def get_customer_appointments(
    customer_id: int,
    include_cancelled: bool = Query(False, description="Include cancelled appointments"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all appointments for a customer.
    Called by voice tool: get_customer_appointments
    """
    logger.info(f"Getting appointments for customer: {customer_id}")
    
    # Get customer
    customer_result = await db.execute(
        select(DemoCustomer).where(DemoCustomer.id == customer_id)
    )
    customer = customer_result.scalar_one_or_none()
    
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    # Build appointments query
    query = select(DemoAppointment).where(
        DemoAppointment.customer_id == customer_id
    ).options(
        selectinload(DemoAppointment.time_slot)
    ).order_by(DemoAppointment.created_at.desc())
    
    if not include_cancelled:
        query = query.where(DemoAppointment.status != AppointmentStatus.CANCELLED)
    
    result = await db.execute(query)
    appointments = result.scalars().all()
    
    return CustomerAppointmentsResponse(
        customer=customer.to_dict(),
        appointments=[appt.to_dict() for appt in appointments],
        total_count=len(appointments)
    )


@demo_appointments_router.patch("/appointments/{appointment_id}/cancel", response_model=AppointmentResponse)
async def cancel_appointment(
    appointment_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel an appointment.
    Called by voice tool: cancel_appointment
    """
    logger.info(f"Cancelling appointment: {appointment_id}")
    
    result = await db.execute(
        select(DemoAppointment)
        .where(DemoAppointment.id == appointment_id)
        .options(
            selectinload(DemoAppointment.customer),
            selectinload(DemoAppointment.time_slot)
        )
    )
    appointment = result.scalar_one_or_none()
    
    if not appointment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Appointment not found"
        )
    
    if appointment.status == AppointmentStatus.CANCELLED:
        return AppointmentResponse(
            success=False,
            appointment=appointment.to_dict(),
            message="This appointment is already cancelled."
        )
    
    # Cancel the appointment
    appointment.cancel()
    await db.commit()
    await db.refresh(appointment)
    
    logger.info(f"Cancelled appointment: {appointment_id}")
    return AppointmentResponse(
        success=True,
        appointment=appointment.to_dict(),
        message="Appointment cancelled successfully."
    )


@demo_appointments_router.get("/appointments", response_model=List[dict])
async def list_all_appointments(
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db)
):
    """List all appointments (admin view)."""
    query = select(DemoAppointment).options(
        selectinload(DemoAppointment.customer),
        selectinload(DemoAppointment.time_slot)
    ).order_by(DemoAppointment.created_at.desc()).limit(limit)
    
    if status_filter:
        try:
            status_enum = AppointmentStatus(status_filter)
            query = query.where(DemoAppointment.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Valid values: {[s.value for s in AppointmentStatus]}"
            )
    
    result = await db.execute(query)
    appointments = result.scalars().all()
    
    return [appt.to_dict() for appt in appointments]

import asyncio
from contextlib import asynccontextmanager
from functools import wraps
import json
import logging
from datetime import datetime, date
from enum import Enum
from typing import Optional, Annotated, List, Dict
import uuid
import os
import aiosqlite
from fastapi import FastAPI, HTTPException, Depends, Body, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
import uvicorn

# --- 1. Logging and App Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("business-api-backend")

DB_PATH = "business_schedules.db"

# --- NEW: Lifespan Manager for Startup/Shutdown Events ---
async def init_db():
    """Initializes the database and creates the schedules table if it doesn't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS schedules (
                scheduleRecordId TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT,
                startTime TEXT NOT NULL, endTime TEXT NOT NULL, scheduleType TEXT NOT NULL,
                locationId TEXT NOT NULL, organizationId TEXT NOT NULL, instructorId TEXT,
                isRecurring INTEGER NOT NULL, recurrenceRule TEXT, studentId TEXT, groupClassId TEXT,
                scheduleLocationType TEXT NOT NULL, meetingUrl TEXT, roomId TEXT, customAddress TEXT,
                originalScheduleId TEXT
            )
        """)
        await db.commit()
    logger.info("Database initialized successfully.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    This replaces the deprecated on_event decorator.
    """
    logger.info("Application startup...")
    await init_db()
    yield
    logger.info("Application shutdown.")

app = FastAPI(
    title="Business Scheduling API",
    description="Backend service for managing business schedules, including lessons, classes, and block-offs.",
    version="2.0.0",
    lifespan=lifespan  # Use the new lifespan manager
)

# --- 2. Authentication Placeholder ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
async def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated, Authorization header is missing.")
    return {"token": token}

# --- 3. Pydantic Models & Enums with Business Logic ---
class ScheduleType(str, Enum):
    PRIVATE_LESSON = "PRIVATE_LESSON"
    GROUP_CLASS = "GROUP_CLASS"
    BLOCK_OFF = "BLOCK_OFF"

class ScheduleLocationType(str, Enum):
    VIRTUAL = "VIRTUAL"
    IN_PERSON = "IN_PERSON"
    CUSTOM_ADDRESS = "CUSTOM_ADDRESS"

class RecurringMode(str, Enum):
    SINGLE_OCCURRENCE = "SINGLE_OCCURRENCE"
    THIS_AND_FUTURE = "THIS_AND_FUTURE"
    ENTIRE_SERIES = "ENTIRE_SERIES"

class ScheduleBase(BaseModel):
    title: str = Field(..., description="Title or name of the schedule entry.")
    startTime: datetime
    endTime: datetime
    scheduleType: ScheduleType
    scheduleLocationType: ScheduleLocationType
    description: Optional[str] = None
    instructorId: Optional[str] = None
    isRecurring: bool = False
    recurrenceRule: Optional[str] = None
    studentId: Optional[str] = None
    groupClassId: Optional[str] = None
    meetingUrl: Optional[str] = None
    roomId: Optional[str] = None
    customAddress: Optional[str] = None

class CreateSchedulePayload(ScheduleBase):
    organizationId: str = Field(..., description="The user's organization ID.")
    locationId: str = Field(..., description="The user's selected location ID.")
    
    @model_validator(mode='after')
    def check_conditional_fields(self) -> 'CreateSchedulePayload':
        if self.scheduleType == ScheduleType.PRIVATE_LESSON and not self.studentId:
            raise ValueError("studentId is required for a PRIVATE_LESSON.")
        if self.scheduleType == ScheduleType.GROUP_CLASS and not self.groupClassId:
            raise ValueError("groupClassId is required for a GROUP_CLASS.")
        if self.scheduleLocationType == ScheduleLocationType.VIRTUAL and not self.meetingUrl:
            raise ValueError("meetingUrl is required for a VIRTUAL location.")
        if self.scheduleLocationType == ScheduleLocationType.IN_PERSON and not self.roomId:
            raise ValueError("roomId is required for an IN_PERSON location.")
        if self.scheduleLocationType == ScheduleLocationType.CUSTOM_ADDRESS and not self.customAddress:
            raise ValueError("customAddress is required for a CUSTOM_ADDRESS location.")
        return self

class ScheduleResponse(ScheduleBase):
    scheduleRecordId: str
    organizationId: str
    locationId: str
    originalScheduleId: Optional[str] = None

class UpdateSchedulePayload(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    startTime: Optional[datetime] = None
    endTime: Optional[datetime] = None
    instructorId: Optional[str] = None
    mode: Optional[RecurringMode] = Field(None, description="Required if updating a recurring event.")
    instanceOriginalStartTime: Optional[datetime] = Field(None, description="Original start time for recurring instance.")

class DeleteSchedulePayload(BaseModel):
    scheduleRecordId: str
    mode: Optional[RecurringMode] = Field(None, description="Required if deleting a recurring event.")
    instanceOriginalStartTime: Optional[datetime] = Field(None, description="Original start time for recurring instance.")


# --- 4. API Endpoints ---
@app.exception_handler(RequestValidationError)
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=400, content={"detail": exc.errors()})

@app.post("/api/v1/account/schedule", status_code=201, response_model=ScheduleResponse, dependencies=[Depends(get_current_user)])
async def create_schedule_entry(payload: CreateSchedulePayload):
    schedule_id = str(uuid.uuid4())
    db_payload = payload.model_dump()
    db_payload['scheduleRecordId'] = schedule_id
    db_payload['startTime'] = db_payload['startTime'].isoformat()
    db_payload['endTime'] = db_payload['endTime'].isoformat()
    if payload.isRecurring:
        db_payload['originalScheduleId'] = schedule_id
    
    columns = ', '.join(db_payload.keys())
    placeholders = ', '.join('?' for _ in db_payload)
    
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(f"INSERT INTO schedules ({columns}) VALUES ({placeholders})", list(db_payload.values()))
            await db.commit()
    except Exception as e:
        logger.error(f"DB Error on create: {e}")
        raise HTTPException(status_code=500, detail="Could not create schedule in the database.")
        
    logger.info(f"Created schedule {schedule_id}")
    return ScheduleResponse(**db_payload)

@app.delete("/api/v1/account/schedule", response_model=ScheduleResponse, dependencies=[Depends(get_current_user)])
async def delete_schedule_entry(payload: DeleteSchedulePayload):
    logger.info(f"Attempting to delete schedule: {payload.scheduleRecordId}")
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM schedules WHERE scheduleRecordId = ?", (payload.scheduleRecordId,))
        schedule_to_delete = await cursor.fetchone()

        if not schedule_to_delete:
            raise HTTPException(status_code=404, detail=f"Schedule with ID {payload.scheduleRecordId} not found.")

        if schedule_to_delete["isRecurring"]:
            if not payload.mode:
                raise HTTPException(status_code=400, detail="The 'mode' field is required to delete a recurring event.")
            logger.info(f"Handling recurring delete with mode: {payload.mode.value}")

        await db.execute("DELETE FROM schedules WHERE scheduleRecordId = ?", (payload.scheduleRecordId,))
        await db.commit()
    
    logger.info(f"Deleted schedule {payload.scheduleRecordId}")
    return ScheduleResponse(**schedule_to_delete)

@app.put("/api/v1/account/schedule/{scheduleRecordId}", response_model=ScheduleResponse, dependencies=[Depends(get_current_user)])
async def update_schedule_entry(scheduleRecordId: str, payload: UpdateSchedulePayload):
    logger.info(f"Attempting to update schedule: {scheduleRecordId}")
    
    update_data = payload.model_dump(exclude_none=True)
    update_data.pop("mode", None)
    update_data.pop("instanceOriginalStartTime", None)
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided.")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM schedules WHERE scheduleRecordId = ?", (scheduleRecordId,))
        schedule_to_update = await cursor.fetchone()

        if not schedule_to_update:
            raise HTTPException(status_code=404, detail=f"Schedule with ID {scheduleRecordId} not found.")

        if schedule_to_update["isRecurring"] and not payload.mode:
            raise HTTPException(status_code=400, detail="The 'mode' field is required to update a recurring event.")
        
        for key in ['startTime', 'endTime']:
            if key in update_data:
                update_data[key] = update_data[key].isoformat()

        set_clause = ", ".join(f"{key} = ?" for key in update_data.keys())
        values = list(update_data.values()) + [scheduleRecordId]

        await db.execute(f"UPDATE schedules SET {set_clause} WHERE scheduleRecordId = ?", values)
        await db.commit()
        
        cursor = await db.execute("SELECT * FROM schedules WHERE scheduleRecordId = ?", (scheduleRecordId,))
        updated_row = await cursor.fetchone()
        
    logger.info(f"Updated schedule {scheduleRecordId}")
    return ScheduleResponse(**updated_row)

@app.get("/api/v1/account/schedule", response_model=Dict[str, List[ScheduleResponse]], dependencies=[Depends(get_current_user)])
async def query_schedule_entries(startDate: date, endDate: date, organizationId: str, locationId: str):
    logger.info(f"Querying schedules for org {organizationId} from {startDate} to {endDate}")
    schedules = []
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT * FROM schedules WHERE organizationId = ? AND locationId = ? AND
            date(startTime) BETWEEN ? AND ? ORDER BY startTime
            """, (organizationId, locationId, startDate.isoformat(), endDate.isoformat()))
        
        rows = await cursor.fetchall()
        for row in rows:
            schedules.append(ScheduleResponse(**row))
            
    return {"data": schedules}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)